package ops

import (
	"fmt"
	"math"
	"sync"

	"github.com/nivrrex/trufor-go/internal/kernels"
	"github.com/nivrrex/trufor-go/internal/tensor"
)

// Conv2D 是 nn.Conv2d 的推理实现（NCHW、单样本、对称 padding、stride 均匀）。
//
// 计算方式为标准 im2col + GEMM：把输入展开成 K×M 的列矩阵（K = InC*KH*KW，
// M = 输出像素数），再与权重 (OutC×K) 相乘。GEMM 的输出行是"输出通道"，而每个
// 输出通道在 NCHW 中的行块是连续的，因此结果可直接写回，无需转置。
type Conv2D struct {
	InC, OutC  int
	KH, KW     int
	Stride     int
	PadH, PadW int
	w          []float32 // OutC×InC×KH×KW（PyTorch 布局）
	bias       []float32 // 可为 nil
	kernelVol  int       // (InC/Groups)*KH*KW
	tapOnce    sync.Once
	tap        *kernels.TapPlan // 3×3/stride=1 的按 tap 累加面板（懒构建）
	aOnce      sync.Once
	ap         []float32 // 预打包的 A 微面板（权重固定，可跨调用复用）
	apTiles    int
	apTail     int
	winoOnce   sync.Once
	wino       *winogradPlan
	// winoEnabled 控制是否使用 Winograd F(2×2,3×3) 路径。
	//
	// 实测（512²、64→64、4 线程）：Winograd 254 ms vs 按 tap 累加 74 ms——虽然乘加量降到
	// 1/2.25，但 NCHW 下输入/输出变换是标量循环、且需要物化 V/M 中间量，开销远超收益。
	// 在 oneDNN 里这类变换是与 GEMM 融合、在寄存器里完成并配合 NHWC 的，本代码结构下做不到。
	// 因此默认关闭；保留实现与 TestWinogradAccuracy 以便换硬件/布局后重新评估。
	winoEnabled bool
	// Groups 为分组数（1 表示普通卷积；等于 InC 且 InC==OutC 时为深度卷积）。
	// 分组卷积须调用 ForwardGrouped。
	Groups int
}

// NewConv2D 构造卷积层。w 为 OutC×InC×KH×KW（PyTorch 布局），bias 可为 nil。
func NewConv2D(w []float32, outC, inC, kh, kw, stride, padH, padW int, bias []float32) (*Conv2D, error) {
	if outC <= 0 || inC <= 0 || kh <= 0 || kw <= 0 || stride <= 0 || padH < 0 || padW < 0 {
		return nil, fmt.Errorf("conv2d 参数非法: outC=%d inC=%d k=%dx%d stride=%d pad=%dx%d", outC, inC, kh, kw, stride, padH, padW)
	}
	vol := inC * kh * kw
	if len(w) != outC*vol {
		return nil, fmt.Errorf("conv2d 权重长度 %d != %d (outC=%d, inC=%d, k=%dx%d)", len(w), outC*vol, outC, inC, kh, kw)
	}
	if bias != nil && len(bias) != outC {
		return nil, fmt.Errorf("conv2d bias 长度 %d != %d", len(bias), outC)
	}
	return &Conv2D{
		InC: inC, OutC: outC, KH: kh, KW: kw,
		Stride: stride, PadH: padH, PadW: padW,
		w: w, bias: bias, kernelVol: vol, winoEnabled: winogradDefault,
	}, nil
}

// OutputSize 返回给定输入尺寸下的输出尺寸。
func (c *Conv2D) OutputSize(h, w int) (int, int) {
	oh := (h+2*c.PadH-c.KH)/c.Stride + 1
	ow := (w+2*c.PadW-c.KW)/c.Stride + 1
	return oh, ow
}

// 卷积分块的两个预算：
//   - rowBlockBudget：单块像素上限，控制"输入行窗口"大小，使 9 次邻域复用尽量命中缓存；
//   - colBudget：im2col 列矩阵（kernelVol×行块像素×4B）的单块上限，超过则改用融合取值，
//     避免大图时列矩阵远超缓存、退化成 DRAM 往返。
const (
	// windowBudget 是"每个 worker 的输入行窗口"字节上限（含上下各一行 halo）。
	// 并行时 4 个 worker 的工作集合需一起装进 L3，因此按 ~512 KB/worker 估算。
	windowBudget = 512 << 10
	// colBudget 是 im2col 列矩阵的单块上限，超过则改用融合取值。
	colBudget = 8 << 20
)

// blockRowsFor 给出每次处理的行数（上限 8，下限 1），使单个 worker 的输入行窗口
// 大致落在 windowBudget 内——这是卷积访存的实际工作集合。
func (c *Conv2D) blockRowsFor(outW int) int {
	if outW <= 0 || c.InC <= 0 {
		return 1
	}
	rows := windowBudget / (outW * c.InC * 4)
	if rows > 8 {
		rows = 8
	}
	if rows < 1 {
		rows = 1
	}
	return rows
}

// useFused 判断是否跳过 im2col 物化、改为在 GEMM 打包 B 面板时直接从输入取值。
// 两种方式的取值顺序与数值结果完全相同，只是内存流量路径不同：
// 列矩阵能装进缓存时物化更快（连续 copy），超出缓存时融合取值明显更快。
func (c *Conv2D) useFused(outW int) bool {
	rows := c.blockRowsFor(outW)
	return c.kernelVol*rows*outW*4 > colBudget
}

// Forward 执行卷积。workers 为并行度（<=1 视为串行）。
//
// 计算方式为 im2col + GEMM（列矩阵能装进缓存时），或"融合 im2col 的隐式 GEMM"
// （大图时跳过列矩阵物化，见 useFused）。两种方式的取值顺序与累加顺序完全一致，
// 结果逐位相同。
func (c *Conv2D) Forward(x *tensor.Tensor, workers int) (*tensor.Tensor, error) {
	oh, ow := c.OutputSize(x.H, x.W)
	if oh <= 0 || ow <= 0 {
		return nil, fmt.Errorf("conv2d 输入 %dx%d 对 kernel %dx%d stride %d 太小", x.H, x.W, c.KH, c.KW, c.Stride)
	}
	out := tensor.New(c.OutC, oh, ow)
	if err := c.ForwardInto(x, out, workers); err != nil {
		return nil, err
	}
	return out, nil
}

// Affine 描述卷积输出的逐通道仿射后处理（y = x*Scale + Shift），可选接 ReLU。
// 依次执行 x += bias、x = x*Scale + Shift、x = max(0,x)，与"卷积 → BatchNorm → ReLU"
// 三次独立遍历的算术顺序完全一致，但只遍历一次输出。
type Affine struct {
	Scale []float32
	Shift []float32
	Relu  bool
}

// ForwardInto 与 Forward 相同，但把结果写入调用方提供的张量（形状必须完全一致），
// 便于在推理链中复用缓冲、避免每层都分配大张量。
func (c *Conv2D) ForwardInto(x, dst *tensor.Tensor, workers int) error {
	return c.forwardInto(x, dst, nil, workers)
}

// ForwardIntoAffine 在 ForwardInto 基础上融合逐通道仿射变换与 ReLU（见 Affine 说明）。
func (c *Conv2D) ForwardIntoAffine(x, dst *tensor.Tensor, epi *Affine, workers int) error {
	return c.forwardInto(x, dst, epi, workers)
}

func (c *Conv2D) forwardInto(x, dst *tensor.Tensor, epi *Affine, workers int) error {
	if dst == nil {
		return fmt.Errorf("conv2d.ForwardInto: dst 为 nil")
	}
	if x.C != c.InC {
		return fmt.Errorf("conv2d 输入通道 %d != %d", x.C, c.InC)
	}
	oh, ow := c.OutputSize(x.H, x.W)
	if dst.C != c.OutC || dst.H != oh || dst.W != ow {
		return fmt.Errorf("conv2d.ForwardInto: dst 形状 (%d,%d,%d) != (%d,%d,%d)", dst.C, dst.H, dst.W, c.OutC, oh, ow)
	}
	if workers < 1 {
		workers = 1
	}
	// Winograd F(2×2,3×3)：3×3/stride1/pad1 且尺寸够大时换用变换域计算
	if c.useWinograd(x.H, x.W) {
		c.convWinograd(x, dst, workers)
		c.finishEpi(dst, epi)
		return nil
	}
	blockRows := c.blockRowsFor(ow)
	useTaps := c.useTaps()
	// 输出初值直接填成 bias（无 bias 则清零）：所有卷积路径都在这块缓冲上累加，
	// 因此卷积结束后不必再单独遍历一遍加 bias。
	if c.bias != nil {
		plane := oh * ow
		for o := 0; o < c.OutC; o++ {
			seg := dst.Data[o*plane : (o+1)*plane]
			b := c.bias[o]
			for i := range seg {
				seg[i] = b
			}
		}
	} else {
		zeroF32(dst.Data)
	}

	// 1×1 / stride 1 / 无 padding：取值即恒等映射，直接把输入通道平面当作 GEMM 的 B。
	if c.KH == 1 && c.KW == 1 && c.Stride == 1 && c.PadH == 0 && c.PadW == 0 {
		planeIn := x.H * x.W
		ap, tiles, tail := c.packedA()
		kernels.SgemmPrepacked(ap, c.w, c.OutC, tiles, tail, planeIn, c.InC, x.Data, planeIn, dst.Data, planeIn, workers)
		c.finishEpi(dst, epi)
		return nil
	}

	fused := c.useFused(ow)
	run := c.convRowsIm2col
	if useTaps {
		run = c.convRowsTaps
	} else if fused {
		run = c.convRowsFused
	}
	if workers == 1 || oh <= blockRows {
		var buf []float32
		for lo := 0; lo < oh; lo += blockRows {
			hi := lo + blockRows
			if hi > oh {
				hi = oh
			}
			run(x, dst, lo, hi, workers, buf)
		}
		c.finishEpi(dst, epi)
		return nil
	}

	// 并行：按行块切分，块间输出区互不重叠；每块复用一份列矩阵缓冲
	var scratchLen int
	switch {
	case useTaps:
		scratchLen = c.InC * (blockRows + 2) * (ow + 2)
	case fused:
		scratchLen = 0
	default:
		scratchLen = c.kernelVol * blockRows * ow
	}
	pool := make(chan []float32, workers)
	if scratchLen > 0 {
		for i := 0; i < workers; i++ {
			pool <- make([]float32, scratchLen)
		}
	}
	sem := make(chan struct{}, workers)
	var wg sync.WaitGroup
	for lo := 0; lo < oh; lo += blockRows {
		hi := lo + blockRows
		if hi > oh {
			hi = oh
		}
		var buf []float32
		if scratchLen > 0 {
			buf = <-pool
		}
		wg.Add(1)
		sem <- struct{}{}
		go func(lo, hi int, buf []float32) {
			defer wg.Done()
			defer func() { <-sem }()
			run(x, dst, lo, hi, 1, buf)
			if scratchLen > 0 {
				pool <- buf
			}
		}(lo, hi, buf)
	}
	wg.Wait()
	c.finishEpi(dst, epi)
	return nil
}

// finishEpi 对卷积输出做逐通道仿射（+ReLU）。bias 已在累加前填入终点值，这里不再加。
func (c *Conv2D) finishEpi(out *tensor.Tensor, epi *Affine) {
	if epi == nil {
		return
	}
	plane := out.H * out.W
	for o := 0; o < c.OutC; o++ {
		seg := out.Data[o*plane : (o+1)*plane]
		s := epi.Scale[o]
		t := epi.Shift[o]
		if epi.Relu {
			for i, v := range seg {
				v = v*s + t
				if v < 0 {
					v = 0
				}
				seg[i] = v
			}
			continue
		}
		for i, v := range seg {
			seg[i] = v*s + t
		}
	}
}

// finish 在卷积输出上一次性完成 bias、仿射与 ReLU。
// packedA 返回（惰性构建并缓存的）A 微面板。
func (c *Conv2D) packedA() ([]float32, int, int) {
	c.aOnce.Do(func() {
		c.ap, c.apTiles, c.apTail = kernels.PackA(c.w, c.OutC, c.kernelVol, c.kernelVol)
	})
	return c.ap, c.apTiles, c.apTail
}

func (c *Conv2D) finish(out *tensor.Tensor, epi *Affine) {
	plane := out.H * out.W
	for o := 0; o < c.OutC; o++ {
		seg := out.Data[o*plane : (o+1)*plane]
		var b float32
		if c.bias != nil {
			b = c.bias[o]
		}
		if epi != nil {
			s := epi.Scale[o]
			t := epi.Shift[o]
			if epi.Relu {
				for i, v := range seg {
					v = (v+b)*s + t
					if v < 0 {
						v = 0
					}
					seg[i] = v
				}
				continue
			}
			for i, v := range seg {
				seg[i] = (v+b)*s + t
			}
			continue
		}
		if b != 0 {
			for i := range seg {
				seg[i] += b
			}
		}
	}
}

// convRowsIm2col 计算输出行区间 [rowLo, rowHi)：先把该行块展开成 K×N 列矩阵，再做 GEMM。
// buf 为可复用的列矩阵缓冲（nil 表示内部按需分配）。
func (c *Conv2D) convRowsIm2col(x, out *tensor.Tensor, rowLo, rowHi, workers int, buf []float32) {
	ow := out.W
	m := (rowHi - rowLo) * ow
	k := c.kernelVol
	if cap(buf) < k*m {
		buf = make([]float32, k*m)
	}
	col := buf[:k*m]

	hIn, wIn := x.H, x.W
	planeIn := hIn * wIn
	p := 0
	for ic := 0; ic < c.InC; ic++ {
		src := x.Data[ic*planeIn:]
		for ky := 0; ky < c.KH; ky++ {
			for kx := 0; kx < c.KW; kx++ {
				dst := col[p*m : (p+1)*m]
				p++
				idx := 0
				for oy := rowLo; oy < rowHi; oy++ {
					iy := oy*c.Stride - c.PadH + ky
					if uint(iy) >= uint(hIn) {
						zeroF32(dst[idx : idx+ow])
						idx += ow
						continue
					}
					srow := src[iy*wIn:]
					ix := -c.PadW + kx
					if c.Stride == 1 {
						// stride=1 时行内取值连续，用 copy 走 memmove 快路径
						end := ix + ow
						if ix >= wIn || end <= 0 {
							zeroF32(dst[idx : idx+ow])
						} else {
							left, right := 0, 0
							if ix < 0 {
								left = -ix
							}
							if end > wIn {
								right = end - wIn
							}
							if left > 0 {
								zeroF32(dst[idx : idx+left])
							}
							cs := idx + left
							ce := idx + ow - right
							base := ix + left
							copy(dst[cs:ce], srow[base:base+(ce-cs)])
							if right > 0 {
								zeroF32(dst[ce : idx+ow])
							}
						}
						idx += ow
						continue
					}
					for i := 0; i < ow; i++ {
						if uint(ix) < uint(wIn) {
							dst[idx] = srow[ix]
						} else {
							dst[idx] = 0
						}
						idx++
						ix += c.Stride
					}
				}
			}
		}
	}
	planeOut := out.H * out.W
	base := rowLo * ow
	ap, tiles, tail := c.packedA()
	kernels.SgemmPrepacked(ap, c.w, c.OutC, tiles, tail, m, k, col, m, out.Data[base:], planeOut, 1)
}

// convRowsFused 计算输出行区间 [rowLo, rowHi)，GEMM 的 B 面板直接由输入张量生成（跳过 im2col）。
func (c *Conv2D) convRowsFused(x, out *tensor.Tensor, rowLo, rowHi, workers int, _ []float32) {
	ow := out.W
	n := (rowHi - rowLo) * ow
	planeIn := x.H * x.W
	hIn, wIn := x.H, x.W
	khw := c.KH * c.KW
	stride, padH, padW := c.Stride, c.PadH, c.PadW
	kw := c.KW

	fill := func(k0, kk, jt, cols int, dst []float32) {
		for pl := 0; pl < kk; pl++ {
			p := k0 + pl
			ic := p / khw
			rem := p - ic*khw
			ky := rem / kw
			kxc := rem - ky*kw
			src := x.Data[ic*planeIn:]
			row := dst[pl*cols : pl*cols+cols]
			// 由 jt 定位起始像素，之后按行段递增（避免逐像素除法）
			oy := rowLo + jt/ow
			ox := jt - (jt/ow)*ow
			iy := oy*stride - padH + ky
			jj := 0
			for jj < cols {
				seg := ow - ox
				if seg > cols-jj {
					seg = cols - jj
				}
				if uint(iy) >= uint(hIn) {
					zeroF32(row[jj : jj+seg])
				} else if stride == 1 {
					// stride=1 时行内取值连续，用 copy 走 memmove 快路径
					ix0 := ox - padW + kxc
					end := ix0 + seg
					if ix0 >= wIn || end <= 0 {
						zeroF32(row[jj : jj+seg])
					} else {
						left, right := 0, 0
						if ix0 < 0 {
							left = -ix0
						}
						if end > wIn {
							right = end - wIn
						}
						if left > 0 {
							zeroF32(row[jj : jj+left])
						}
						cstart := jj + left
						cend := jj + seg - right
						base := iy*wIn + ix0 + left
						copy(row[cstart:cend], src[base:base+(cend-cstart)])
						if right > 0 {
							zeroF32(row[cend : jj+seg])
						}
					}
				} else {
					ix := ox*stride - padW + kxc
					for c := 0; c < seg; c++ {
						if uint(ix) < uint(wIn) {
							row[jj+c] = src[iy*wIn+ix]
						} else {
							row[jj+c] = 0
						}
						ix += stride
					}
				}
				jj += seg
				ox += seg
				if ox == ow {
					ox = 0
					iy += stride
				}
			}
		}
	}

	planeOut := out.H * out.W
	kernels.SgemmGather(c.OutC, n, c.kernelVol, c.w, c.kernelVol, fill, out.Data[rowLo*ow:], planeOut, workers)
}

// ForwardTokens 是深度卷积在 token-major（[n][C]，n = H*W）布局下的实现：
// 9 个 (ky,kx) 位移各做一次"按通道加权"的整段累加，无需 NCHW ↔ token 转置。
// 累加顺序为 (ky,kx) 递增，与 NCHW 版本逐位相同。
// 仅支持 groups==InC==OutC 的 3×3、stride=1、padding=1 情形，其余情形返回错误。
func (c *Conv2D) ForwardTokens(tok []float32, n, h, w int) ([]float32, error) {
	if c.Groups != c.InC || c.OutC != c.InC || c.KH != 3 || c.KW != 3 || c.Stride != 1 || c.PadH != 1 || c.PadW != 1 {
		return nil, fmt.Errorf("conv2d.ForwardTokens: 仅支持 3×3/stride1/pad1 的深度卷积")
	}
	if h*w != n {
		return nil, fmt.Errorf("conv2d.ForwardTokens: n=%d 与 %dx%d 不符", n, h, w)
	}
	if len(tok) < n*c.InC {
		return nil, fmt.Errorf("conv2d.ForwardTokens: 数据长度不足")
	}
	ch := c.InC
	dst := make([]float32, n*ch)
	if c.bias != nil {
		for i := 0; i < n; i++ {
			base := i * ch
			for cc := 0; cc < ch; cc++ {
				dst[base+cc] = c.bias[cc]
			}
		}
	}
	for ky := 0; ky < 3; ky++ {
		for kx := 0; kx < 3; kx++ {
			for oy := 0; oy < h; oy++ {
				iy := oy + ky - 1
				if uint(iy) >= uint(h) {
					continue
				}
				// 列方向的有效区间：ox + kx - 1 ∈ [0, w)
				lo := 0
				if kx == 0 {
					lo = 1
				}
				hi := w
				if kx == 2 {
					hi = w - 1
				}
				if hi <= lo {
					continue
				}
				srcOff := (iy*w + lo + kx - 1) * ch
				dstOff := (oy*w + lo) * ch
				kernels.AxpyChannelwise(dst[dstOff:dstOff+(hi-lo)*ch], tok[srcOff:srcOff+(hi-lo)*ch], c.tapWeights(ky, kx))
			}
		}
	}
	return dst, nil
}

// tapWeights 返回深度卷积在某 (ky,kx) 位移上的逐通道权重（长度为通道数）。
func (c *Conv2D) tapWeights(ky, kx int) []float32 {
	ch := c.InC
	out := make([]float32, ch)
	for cc := 0; cc < ch; cc++ {
		out[cc] = c.w[cc*9+ky*3+kx]
	}
	return out
}

// BatchNorm 是 nn.BatchNorm2d 的推理实现（使用 running 统计量）。
type BatchNorm struct {
	C     int
	scale []float32
	shift []float32
}

// NewBatchNorm 预计算 y = (x-mean)/sqrt(var+eps)*gamma + beta。
func NewBatchNorm(mean, variance, gamma, beta []float32, eps float64) (*BatchNorm, error) {
	c := len(mean)
	if c == 0 || len(variance) != c || len(gamma) != c || len(beta) != c {
		return nil, fmt.Errorf("batchnorm 参数长度不一致: mean=%d var=%d gamma=%d beta=%d", len(mean), len(variance), len(gamma), len(beta))
	}
	if eps < 0 {
		return nil, fmt.Errorf("batchnorm eps 非法: %v", eps)
	}
	b := &BatchNorm{C: c, scale: make([]float32, c), shift: make([]float32, c)}
	for i := 0; i < c; i++ {
		s := float32(float64(gamma[i]) / math.Sqrt(float64(variance[i])+eps))
		b.scale[i] = s
		b.shift[i] = beta[i] - mean[i]*s
	}
	return b, nil
}

// Apply 就地归一化，返回同一张量。
func (b *BatchNorm) Apply(x *tensor.Tensor) error {
	if x.C != b.C {
		return fmt.Errorf("batchnorm 输入通道 %d != %d", x.C, b.C)
	}
	plane := x.H * x.W
	for c := 0; c < b.C; c++ {
		s := b.scale[c]
		t := b.shift[c]
		d := x.Data[c*plane : (c+1)*plane]
		for i := range d {
			d[i] = d[i]*s + t
		}
	}
	return nil
}

// ReLUInPlace 就地执行 max(0, x)。
func ReLUInPlace(x *tensor.Tensor) {
	for i, v := range x.Data {
		if v < 0 {
			x.Data[i] = 0
		}
	}
}

// NewConv2DGrouped 构造分组卷积（groups>1）。w 为 OutC×(InC/groups)×KH×KW。
// 当 groups==InC==OutC 时走深度卷积专路（ForwardGrouped）。
func NewConv2DGrouped(w []float32, outC, inC, kh, kw, stride, padH, padW, groups int, bias []float32) (*Conv2D, error) {
	if groups <= 0 || inC%groups != 0 || outC%groups != 0 {
		return nil, fmt.Errorf("conv2d groups 非法: groups=%d inC=%d outC=%d", groups, inC, outC)
	}
	if groups == 1 {
		return NewConv2D(w, outC, inC, kh, kw, stride, padH, padW, bias)
	}
	if outC <= 0 || inC <= 0 || kh <= 0 || kw <= 0 || stride <= 0 || padH < 0 || padW < 0 {
		return nil, fmt.Errorf("conv2d 参数非法: outC=%d inC=%d k=%dx%d stride=%d pad=%dx%d", outC, inC, kh, kw, stride, padH, padW)
	}
	vol := (inC / groups) * kh * kw
	if len(w) != outC*vol {
		return nil, fmt.Errorf("分组卷积权重长度 %d != %d (outC=%d, inC/groups=%d, k=%dx%d)", len(w), outC*vol, outC, inC/groups, kh, kw)
	}
	if bias != nil && len(bias) != outC {
		return nil, fmt.Errorf("conv2d bias 长度 %d != %d", len(bias), outC)
	}
	return &Conv2D{
		InC: inC, OutC: outC, KH: kh, KW: kw,
		Stride: stride, PadH: padH, PadW: padW,
		w: w, bias: bias, kernelVol: vol, Groups: groups,
	}, nil
}

// ForwardGrouped 执行分组/深度卷积。
func (c *Conv2D) ForwardGrouped(x *tensor.Tensor, workers int) (*tensor.Tensor, error) {
	if c.Groups <= 1 {
		return c.Forward(x, workers)
	}
	if x.C != c.InC {
		return nil, fmt.Errorf("conv2d 输入通道 %d != %d", x.C, c.InC)
	}
	oh, ow := c.OutputSize(x.H, x.W)
	if oh <= 0 || ow <= 0 {
		return nil, fmt.Errorf("conv2d 输入 %dx%d 对 kernel %dx%d stride %d 太小", x.H, x.W, c.KH, c.KW, c.Stride)
	}
	out := tensor.New(c.OutC, oh, ow)
	inPer := c.InC / c.Groups
	outPer := c.OutC / c.Groups
	if inPer == 1 && outPer == 1 {
		c.depthwise(x, out, workers)
		return out, nil
	}
	// 通道路径：逐组 im2col + GEMM
	for g := 0; g < c.Groups; g++ {
		sub := &Conv2D{
			InC: inPer, OutC: outPer, KH: c.KH, KW: c.KW,
			Stride: c.Stride, PadH: c.PadH, PadW: c.PadW,
			w:         c.w[g*outPer*c.kernelVol : (g+1)*outPer*c.kernelVol],
			bias:      nil,
			kernelVol: c.kernelVol,
		}
		// 取出该组输入通道，保持 NCHW 连续
		plane := x.H * x.W
		xg := tensor.NewFromData(x.Data[g*inPer*plane:(g+1)*inPer*plane], inPer, x.H, x.W)
		og, err := sub.Forward(xg, workers)
		if err != nil {
			return nil, fmt.Errorf("分组 %d: %w", g, err)
		}
		planeOut := oh * ow
		copy(out.Data[g*outPer*planeOut:(g+1)*outPer*planeOut], og.Data)
	}
	if c.bias != nil {
		planeOut := oh * ow
		for o := 0; o < c.OutC; o++ {
			seg := out.Data[o*planeOut : (o+1)*planeOut]
			b := c.bias[o]
			for i := range seg {
				seg[i] += b
			}
		}
	}
	return out, nil
}

// depthwise 是 groups==InC==OutC 的深度卷积专路：
// 每个通道独立做一个 KH×KW 卷积，行内用 SIMD axpy 累加，边界列用标量补齐。
func (c *Conv2D) depthwise(x, out *tensor.Tensor, workers int) {
	oh, ow := out.H, out.W
	plane := oh * ow
	hIn, wIn := x.H, x.W
	planeIn := hIn * wIn
	kh, kw := c.KH, c.KW
	stride, padH, padW := c.Stride, c.PadH, c.PadW
	if workers < 1 {
		workers = 1
	}
	// 行内可连续累加的区间 [lo, hi)：所有 kx 都不越界
	lo, hi := 0, ow
	for kx := 0; kx < kw; kx++ {
		if -padW+kx < 0 {
			lo = 1
		}
		if -padW+kx+ow > wIn {
			hi = ow - 1
		}
	}
	work := func(chLo, chHi int) {
		for ch := chLo; ch < chHi; ch++ {
			src := x.Data[ch*planeIn : (ch+1)*planeIn]
			dst := out.Data[ch*plane : (ch+1)*plane]
			w := c.w[ch*kh*kw : (ch+1)*kh*kw]
			var bias float32
			if c.bias != nil {
				bias = c.bias[ch]
			}
			for oy := 0; oy < oh; oy++ {
				iy0 := oy*stride - padH
				drow := dst[oy*ow : (oy+1)*ow]
				for ox := range drow {
					drow[ox] = bias
				}
				if stride == 1 && hi > lo {
					for ky := 0; ky < kh; ky++ {
						iy := iy0 + ky
						if uint(iy) >= uint(hIn) {
							continue
						}
						srow := src[iy*wIn:]
						for kx := 0; kx < kw; kx++ {
							wv := w[ky*kw+kx]
							if wv == 0 {
								continue
							}
							kernels.Axpy(drow[lo:hi], srow[lo-padW+kx:hi-padW+kx], wv)
						}
					}
					// 边界列（最多两列）用标量补齐
					for ox := 0; ox < ow; ox++ {
						if ox >= lo && ox < hi {
							continue
						}
						var acc float32
						for ky := 0; ky < kh; ky++ {
							iy := iy0 + ky
							if uint(iy) >= uint(hIn) {
								continue
							}
							for kx := 0; kx < kw; kx++ {
								ix := ox*stride - padW + kx
								if uint(ix) < uint(wIn) {
									acc += src[iy*wIn+ix] * w[ky*kw+kx]
								}
							}
						}
						drow[ox] += acc
					}
					continue
				}
				for ky := 0; ky < kh; ky++ {
					iy := iy0 + ky
					if uint(iy) >= uint(hIn) {
						continue
					}
					srow := src[iy*wIn:]
					for kx := 0; kx < kw; kx++ {
						wv := w[ky*kw+kx]
						if wv == 0 {
							continue
						}
						ix := -padW + kx
						for ox := 0; ox < ow; ox++ {
							if uint(ix) < uint(wIn) {
								drow[ox] += srow[ix] * wv
							}
							ix += stride
						}
					}
				}
			}
		}
	}
	if workers == 1 || c.OutC < workers*2 {
		work(0, c.OutC)
		return
	}
	var wg sync.WaitGroup
	step := (c.OutC + workers - 1) / workers
	for lo2 := 0; lo2 < c.OutC; lo2 += step {
		hi2 := lo2 + step
		if hi2 > c.OutC {
			hi2 = c.OutC
		}
		wg.Add(1)
		go func(a, b int) {
			defer wg.Done()
			work(a, b)
		}(lo2, hi2)
	}
	wg.Wait()
}

// zeroF32 就地清零切片。
func zeroF32(v []float32) {
	for i := range v {
		v[i] = 0
	}
}

// Affine 返回 BatchNorm 的等价的逐通道仿射参数（y = x*scale + shift），可选叠加 ReLU。
func (b *BatchNorm) Affine(relu bool) *Affine {
	return &Affine{Scale: b.scale, Shift: b.shift, Relu: relu}
}

// winogradDefault 是 Winograd 路径的默认开关（实测更慢，见 winoEnabled 注释）。
const winogradDefault = false

// convMicroRows 与 kernels 包微内核的行数一致（6 行一个 M 微块）。
const convMicroRows = 6

// useTaps 判断是否走"按 tap 累加"的 3×3 卷积路径（stride=1、padding=1、非分组）。
func (c *Conv2D) useTaps() bool {
	return c.Groups <= 1 && c.KH == 3 && c.KW == 3 && c.Stride == 1 && c.PadH == 1 && c.PadW == 1 && c.InC >= 8
}

// tapPlan 惰性构建 9 个 tap 的权重面板（并发安全，构建后只读）。
func (c *Conv2D) tapPlan() *kernels.TapPlan {
	c.tapOnce.Do(func() {
		inC, outC := c.InC, c.OutC
		nTiles := outC / convMicroRows
		tailRows := outC - nTiles*convMicroRows
		stride := nTiles*convMicroRows*inC + tailRows*inC
		ap := make([]float32, 9*stride)
		for tap := 0; tap < 9; tap++ {
			ky := tap / 3
			kx := tap - ky*3
			base := tap * stride
			for t := 0; t < nTiles; t++ {
				for r := 0; r < convMicroRows; r++ {
					o := t*convMicroRows + r
					for ic := 0; ic < inC; ic++ {
						ap[base+t*convMicroRows*inC+ic*convMicroRows+r] = c.w[o*inC*9+ic*9+ky*3+kx]
					}
				}
			}
			if tailRows > 0 {
				for r := 0; r < tailRows; r++ {
					o := nTiles*convMicroRows + r
					for ic := 0; ic < inC; ic++ {
						ap[base+nTiles*convMicroRows*inC+ic*tailRows+r] = c.w[o*inC*9+ic*9+ky*3+kx]
					}
				}
			}
		}
		c.tap = &kernels.TapPlan{AP: ap, W: c.w, InC: inC, OutC: outC, Stride: stride, NTiles: nTiles, TailRows: tailRows}
	})
	return c.tap
}

// convRowsTaps 用"按 tap 累加"的方式计算输出行区间 [rowLo, rowHi)：
// 先构造 (InC, R+2, W+2) 的零填充行窗口，再按 9 个位移窗口累加。
func (c *Conv2D) convRowsTaps(x, out *tensor.Tensor, rowLo, rowHi, workers int, buf []float32) {
	plan := c.tapPlan()
	ow := out.W
	rowStride := ow + 2
	hRows := (rowHi - rowLo) + 2
	need := c.InC * hRows * rowStride
	if cap(buf) < need {
		buf = make([]float32, need)
	}
	win := buf[:need]
	zeroF32(win)

	planeIn := x.H * x.W
	for ic := 0; ic < c.InC; ic++ {
		src := x.Data[ic*planeIn : (ic+1)*planeIn]
		dstPlane := win[ic*hRows*rowStride : (ic+1)*hRows*rowStride]
		for r := 0; r < hRows; r++ {
			iy := rowLo - 1 + r
			if uint(iy) >= uint(x.H) {
				continue
			}
			copy(dstPlane[r*rowStride+1:r*rowStride+1+ow], src[iy*x.W:iy*x.W+ow])
		}
	}
	windowPlane := hRows * rowStride
	planeOut := out.H * out.W
	for r := 0; r < rowHi-rowLo; r++ {
		off := r * rowStride // 当前输出行对应的"上一行"在窗口中的起点
		plan.Conv3x3Accumulate(win[off:], windowPlane, rowStride, out.Data, (rowLo+r)*ow, planeOut, ow)
	}
}
