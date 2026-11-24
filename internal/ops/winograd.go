package ops

import (
	"sync"

	"github.com/nivrrex/trufor-go/internal/tensor"
)

// Winograd F(2×2, 3×3) 路径：把 3×3/stride1/pad1 卷积换成"输入变换 → 16 个点乘 GEMM →
// 输出变换"，乘加量由 9 次/输出降到 4 次/输出（2.25×）。
//
// 数值上不是直接卷积的等价形式：变换矩阵带来额外的舍入放大（F(2,3) 约 2~4 倍 fp32 epsilon），
// 因此该路径只在误差断言通过时使用（见 TestWinogradAccuracy）。
//
// 变换矩阵（Lavin & Gray）：
//
//	B^T = [[1,0,-1,0],[0,1,1,0],[0,-1,1,0],[0,1,0,-1]]
//	G   = [[1,0,0],[.5,.5,.5],[.5,-.5,.5],[0,0,1]]
//	A^T = [[1,1,1,0],[0,1,-1,-1]]
const winogradComponents = 16

// winogradPlan 是某个 Conv2D 的变换后权重（惰性构建）。
type winogradPlan struct {
	// U[comp][ic*OutC + o]：第 comp 个变换分量的权重（Cin×Cout，作为 GEMM 的 B）
	U [winogradComponents][]float32
}

// buildWinogradPlan 把 OIHW 权重变换到 Winograd 域（每个分量一张 Cin×Cout 矩阵）。
func buildWinogradPlan(w []float32, inC, outC int) *winogradPlan {
	p := &winogradPlan{}
	tmp := make([]float32, winogradComponents*3)
	for comp := 0; comp < winogradComponents; comp++ {
		p.U[comp] = make([]float32, inC*outC)
	}
	for ic := 0; ic < inC; ic++ {
		for o := 0; o < outC; o++ {
			g := w[o*inC*9+ic*9 : o*inC*9+ic*9+9]
			// G g G^T：先按列做 G g
			for j := 0; j < 3; j++ {
				g0 := g[0*3+j]
				g1 := g[1*3+j]
				g2 := g[2*3+j]
				tmp[0*3+j] = g0
				tmp[1*3+j] = 0.5 * (g0 + g1 + g2)
				tmp[2*3+j] = 0.5 * (g0 - g1 + g2)
				tmp[3*3+j] = g2
			}
			// 再按行做 (Gg) G^T
			for i := 0; i < 4; i++ {
				t0 := tmp[i*3+0]
				t1 := tmp[i*3+1]
				t2 := tmp[i*3+2]
				vals := [4]float32{
					t0,
					0.5 * (t0 + t1 + t2),
					0.5 * (t0 - t1 + t2),
					t2,
				}
				for j := 0; j < 4; j++ {
					p.U[i*4+j][ic*outC+o] = vals[j]
				}
			}
		}
	}
	return p
}

// winogradBlock 是单次处理的行块（以 tile 行计）参数。
const winogradTileRows = 4

// useWinograd 判断是否走 Winograd 路径：3×3/stride1/pad1、非分组、通道数与尺寸足够大。
func (c *Conv2D) useWinograd(h, w int) bool {
	if c.Groups > 1 || c.KH != 3 || c.KW != 3 || c.Stride != 1 || c.PadH != 1 || c.PadW != 1 {
		return false
	}
	if c.InC < 16 || c.OutC < 16 || h < 8 || w < 8 {
		return false
	}
	return c.winoEnabled
}

// winogradPlan 惰性构建变换后权重。
func (c *Conv2D) winogradPlan() *winogradPlan {
	c.winoOnce.Do(func() {
		c.wino = buildWinogradPlan(c.w, c.InC, c.OutC)
	})
	return c.wino
}

// convWinograd 用 Winograd F(2×2,3×3) 计算整幅卷积输出。
func (c *Conv2D) convWinograd(x, dst *tensor.Tensor, workers int) {
	oh, ow := dst.H, dst.W
	outC := c.OutC
	plan := c.winogradPlan()
	if c.bias != nil {
		for o := 0; o < outC; o++ {
			seg := dst.Data[o*oh*ow : (o+1)*oh*ow]
			b := c.bias[o]
			for i := range seg {
				seg[i] = b
			}
		}
	} else {
		zeroF32(dst.Data)
	}

	tilesW := (ow + 1) / 2
	tileRowsPerBlock := winogradTileRows
	if tileRowsPerBlock > (oh+1)/2 {
		tileRowsPerBlock = (oh + 1) / 2
	}
	if tileRowsPerBlock < 1 {
		tileRowsPerBlock = 1
	}

	blocks := ((oh+1)/2 + tileRowsPerBlock - 1) / tileRowsPerBlock
	sem := make(chan struct{}, workers)
	var wg sync.WaitGroup
	for b := 0; b < blocks; b++ {
		ty0 := b * tileRowsPerBlock
		ty1 := ty0 + tileRowsPerBlock
		if ty1 > (oh+1)/2 {
			ty1 = (oh + 1) / 2
		}
		wg.Add(1)
		sem <- struct{}{}
		go func(ty0, ty1 int) {
			defer wg.Done()
			defer func() { <-sem }()
			c.winogradBlock(x, dst, plan, ty0, ty1, tilesW)
		}(ty0, ty1)
	}
	wg.Wait()
}

// winogradBlock 处理 tile 行区间 [ty0, ty1)。
func (c *Conv2D) winogradBlock(x, dst *tensor.Tensor, plan *winogradPlan, ty0, ty1, tilesW int) {
	inC, outC := c.InC, c.OutC
	oh, ow := dst.H, dst.W
	// 左右各留 padding：左 1 列；右 2 列（ow 为奇数时最后一个 tile 会多读一列）
	rowStride := ow + 3
	rowsNeeded := (ty1-ty0)*2 + 2 // 每个 tile 行需要 4 行、相邻 tile 行重叠 2 行
	firstRow := ty0*2 - 1         // 对应输入行（含 padding 后的坐标：窗口第 0 行 = 该行的上一行）

	// 构造零填充行窗口（InC × rowsNeeded × (ow+2)）
	win := make([]float32, inC*rowsNeeded*rowStride)
	planeIn := x.H * x.W
	for ic := 0; ic < inC; ic++ {
		src := x.Data[ic*planeIn : (ic+1)*planeIn]
		dstPlane := win[ic*rowsNeeded*rowStride : (ic+1)*rowsNeeded*rowStride]
		for r := 0; r < rowsNeeded; r++ {
			iy := firstRow + 1 + r - 1 // 窗口行 r 对应的图像行
			if uint(iy) >= uint(x.H) {
				continue
			}
			copy(dstPlane[r*rowStride+1:r*rowStride+1+ow], src[iy*x.W:iy*x.W+ow])
		}
	}

	tiles := (ty1 - ty0) * tilesW
	// V[comp][tile][ic]
	V := make([]float32, winogradComponents*tiles*inC)
	tileTmp := make([]float32, winogradComponents)
	var d [4][4]float32
	var tmp [4][4]float32
	for t := 0; t < tiles; t++ {
		wy := t / tilesW
		wx := t - wy*tilesW
		baseRow := wy * 2 // 窗口内的起始行
		baseCol := wx * 2 // 窗口内的起始列（含左 padding 列）
		for ic := 0; ic < inC; ic++ {
			plane := win[ic*rowsNeeded*rowStride : (ic+1)*rowsNeeded*rowStride]
			for r := 0; r < 4; r++ {
				off := (baseRow+r)*rowStride + baseCol
				for cc := 0; cc < 4; cc++ {
					d[r][cc] = plane[off+cc]
				}
			}
			winogradInputTransform(&d, &tmp, &tileTmp)
			for comp := 0; comp < winogradComponents; comp++ {
				V[comp*tiles*inC+t*inC+ic] = tileTmp[comp]
			}
		}
	}

	// 16 个分量的 GEMM：V[comp](tiles×inC) · U[comp](inC×outC) → M[comp](tiles×outC)
	M := make([]float32, winogradComponents*tiles*outC)
	for comp := 0; comp < winogradComponents; comp++ {
		GemmStrided(V[comp*tiles*inC:], tiles, inC, plan.U[comp], outC, M[comp*tiles*outC:], outC, 1)
	}

	// 输出变换 + 写回（累加到 dst，dst 已含 bias）
	var m [4][4]float32
	var o2 [2][2]float32
	for t := 0; t < tiles; t++ {
		wy := ty0 + t/tilesW
		wx := t - (t/tilesW)*tilesW
		oy := wy * 2
		ox := wx * 2
		for o := 0; o < outC; o++ {
			for comp := 0; comp < winogradComponents; comp++ {
				m[comp/4][comp%4] = M[comp*tiles*outC+t*outC+o]
			}
			winogradOutputTransform(&m, &o2)
			for r := 0; r < 2; r++ {
				y := oy + r
				if y >= oh {
					continue
				}
				for cc := 0; cc < 2; cc++ {
					xx := ox + cc
					if xx >= ow {
						continue
					}
					dst.Data[o*oh*ow+y*ow+xx] += o2[r][cc]
				}
			}
		}
	}
}

// winogradInputTransform 计算 V = B^T d B（4×4 → 4×4）。
func winogradInputTransform(d *[4][4]float32, tmp *[4][4]float32, out *[]float32) {
	for j := 0; j < 4; j++ {
		tmp[0][j] = d[0][j] - d[2][j]
		tmp[1][j] = d[1][j] + d[2][j]
		tmp[2][j] = d[2][j] - d[1][j]
		tmp[3][j] = d[1][j] - d[3][j]
	}
	res := *out
	res[0] = tmp[0][0] - tmp[0][2]
	res[1] = tmp[0][1] + tmp[0][2]
	res[2] = tmp[0][2] - tmp[0][1]
	res[3] = tmp[0][1] - tmp[0][3]
	res[4] = tmp[1][0] - tmp[1][2]
	res[5] = tmp[1][1] + tmp[1][2]
	res[6] = tmp[1][2] - tmp[1][1]
	res[7] = tmp[1][1] - tmp[1][3]
	res[8] = tmp[2][0] - tmp[2][2]
	res[9] = tmp[2][1] + tmp[2][2]
	res[10] = tmp[2][2] - tmp[2][1]
	res[11] = tmp[2][1] - tmp[2][3]
	res[12] = tmp[3][0] - tmp[3][2]
	res[13] = tmp[3][1] + tmp[3][2]
	res[14] = tmp[3][2] - tmp[3][1]
	res[15] = tmp[3][1] - tmp[3][3]
}

// winogradOutputTransform 计算 out = A^T m A（4×4 → 2×2）。
func winogradOutputTransform(m *[4][4]float32, out *[2][2]float32) {
	var t [2][4]float32
	for j := 0; j < 4; j++ {
		t[0][j] = m[0][j] + m[1][j] + m[2][j]
		t[1][j] = m[1][j] - m[2][j] - m[3][j]
	}
	out[0][0] = t[0][0] + t[0][1] + t[0][2]
	out[0][1] = t[0][1] - t[0][2] - t[0][3]
	out[1][0] = t[1][0] + t[1][1] + t[1][2]
	out[1][1] = t[1][1] - t[1][2] - t[1][3]
}
