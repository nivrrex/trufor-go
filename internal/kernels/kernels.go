// Package kernels 提供 CPU 上的浮点 GEMM 内核。
//
// 语义（与 ops.Gemm 一致）：
//
//	C[i*ldc+j] += Σ_p A[i*lda+p] * B[p*ldb+j]     (0<=i<m, 0<=j<n, 0<=p<k)
//
// 累加语义意味着调用方需要先把 C 清零。每个输出元素的累加顺序固定为 p 递增
// （分块后仍按 p 升序逐块累加），因此结果与并行度无关，可复现。
//
// 实现分两路：
//   - amd64 且 CPU 支持 AVX2+FMA：6×16 寄存器分块汇编微内核 + A/B 打包；
//   - 其余情况：纯 Go 4×4 分块回退（正确但慢，约慢一个数量级）。
package kernels

import "sync"

// 分块参数（单位：元素）。MR×NR 与汇编微内核一致。
const (
	mr = 6
	nr = 16
	// kc 为 K 方向分块长度：Direct 路径每块只装卸一次 C；1024 在 Zen2 上
	// 减少输出访存，同时打包面板（1024×32 float32）仍可留在单核 L2。
	kc = 1024
	// nc 为 N 方向分块长度。
	nc = 16
	// mc 为 M 方向分块长度（6 的整数倍）。
	mc = 96
)

// BPanelFiller 负责把逻辑上的 B 矩阵块（行 [k0,k0+kk)、列 [jt,jt+cols)）写入
// 连续缓冲 dst，布局为 dst[p*cols + jj]（p 从 0 起、相对 k0）。
// 有了它，卷积可以跳过 im2col 物化，直接在这里做带 padding 的取值。
type BPanelFiller func(k0, kk, jt, cols int, dst []float32)

// Sgemm 计算 C += A·B。workers<=1 时串行。
//
// 该版本把 B 打包成 16 列微面板后再算：M 较小时（如卷积的 OutC=64，微面板只被 10 个
// M 微块复用）明显更快。M 较大时用 SgemmDirect。
func Sgemm(m, n, k int, a []float32, lda int, b []float32, ldb int, c []float32, ldc int, workers int) {
	fill := func(k0, kk, jt, cols int, dst []float32) {
		for p := 0; p < kk; p++ {
			src := b[(k0+p)*ldb+jt:]
			copy(dst[p*cols:p*cols+cols], src[:cols])
		}
	}
	SgemmGather(m, n, k, a, lda, fill, c, ldc, workers)
}

// PackA 把 A 打包成微面板并返回 (ap, nTiles, tailRows)，供权重固定的调用方
// 缓存复用。返回独立新建缓冲（非池化）——调用方可长期持有。
func PackA(a []float32, m, k, lda int) (ap []float32, nTiles, tailRows int) {
	nTiles = m / mr
	tailRows = m - nTiles*mr
	ap = make([]float32, nTiles*mr*k+tailRows*k)
	return packAInto(ap, a, m, k, lda)
}

// SgemmDirectPrepacked 与 SgemmDirect 相同，但 A 已由 PackA 预先打包（权重固定时省掉每次打包）。
func SgemmDirectPrepacked(ap, a []float32, m, nTiles, tailRows, n, k int, b []float32, ldb int, c []float32, ldc int, workers int) {
	fill := func(k0, kk, jt, cols int, dst []float32) {
		for p := 0; p < kk; p++ {
			copy(dst[p*cols:p*cols+cols], b[(k0+p)*ldb+jt:(k0+p)*ldb+jt+cols])
		}
	}
	_ = fill
	sgemmPrepacked(ap, a, m, nTiles, tailRows, n, k, nil, b, ldb, c, ldc, workers)
}

// SgemmPrepacked 与 Sgemm 相同，但 A 已预先打包（B 仍按 64 列面板打包）。
func SgemmPrepacked(ap, a []float32, m, nTiles, tailRows, n, k int, b []float32, ldb int, c []float32, ldc int, workers int) {
	sgemmPrepacked(ap, a, m, nTiles, tailRows, n, k, nil, b, ldb, c, ldc, workers)
}

// SgemmGatherPrepacked 与 SgemmGather 相同，但 A 已预先打包。
func SgemmGatherPrepacked(ap, a []float32, m, nTiles, tailRows, n, k int, fill BPanelFiller, c []float32, ldc int, workers int) {
	sgemmPrepacked(ap, a, m, nTiles, tailRows, n, k, fill, nil, 0, c, ldc, workers)
}

// sgemmPrepacked 是预打包 A 的公共核心：fill 为 nil 时按 ldb 从 b 直接打包 B 面板。
func sgemmPrepacked(ap, a []float32, m, nTiles, tailRows, n, k int, fill BPanelFiller, b []float32, ldb int, c []float32, ldc int, workers int) {
	if m <= 0 || n <= 0 || k <= 0 {
		return
	}
	if workers < 1 {
		workers = 1
	}
	if fill == nil {
		fill = func(k0, kk, jt, cols int, dst []float32) {
			for p := 0; p < kk; p++ {
				copy(dst[p*cols:p*cols+cols], b[(k0+p)*ldb+jt:(k0+p)*ldb+jt+cols])
			}
		}
	}
	if !simdAvailable() || (nTiles == 0 && tailRows < 2) {
		col := make([]float32, k*n)
		fill(0, k, 0, n, col)
		gemmRowsGo(a, m, k, k, col, n, n, c, ldc, workers)
		return
	}
	run := func(j0, j1 int) {
		workerRange(ap, a, fill, c, m, n, k, k, ldc, j0, j1, nTiles, tailRows)
	}
	jBlocks := (n + nc - 1) / nc
	if jBlocks == 1 || workers == 1 {
		run(0, n)
		return
	}
	if workers > jBlocks {
		workers = jBlocks
	}
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		lo := w * jBlocks / workers * nc
		hi := (w + 1) * jBlocks / workers * nc
		if hi > n {
			hi = n
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			run(lo, hi)
		}(lo, hi)
	}
	wg.Wait()
}

// SgemmDirect 与 Sgemm 数值完全等价，但跳过 B 面板打包：微内核按 bstride 直接读 B。
// 适用于 B 已是连续矩阵的场景（im2col 之后的列矩阵），可省掉一次全量拷贝。
func SgemmDirect(m, n, k int, a []float32, lda int, b []float32, ldb int, c []float32, ldc int, workers int) {
	if m <= 0 || n <= 0 || k <= 0 {
		return
	}
	if lda < k || ldb < n || ldc < n {
		panic("kernels.SgemmDirect: 跨度过小")
	}
	if workers < 1 {
		workers = 1
	}
	if !simdAvailable() {
		gemmRowsGo(a, m, k, lda, b, n, ldb, c, ldc, workers)
		return
	}
	nTiles := m / mr
	tailRows := m - nTiles*mr
	var tailAP []float32
	if tailRows > 0 {
		tailAP = getAp(tailRows * k)
		for r := 0; r < tailRows; r++ {
			row := a[(nTiles*mr+r)*lda:]
			for p := 0; p < k; p++ {
				tailAP[p*tailRows+r] = row[p]
			}
		}
		defer putAp(tailAP)
	}
	if nTiles == 0 && tailRows < 2 {
		gemmRowsGo(a, m, k, lda, b, n, ldb, c, ldc, workers)
		return
	}

	run := func(j0, j1 int) {
		for k0 := 0; k0 < k; k0 += kc {
			k1 := k0 + kc
			if k1 > k {
				k1 = k
			}
			kk := k1 - k0
			for jt := j0; jt < j1; jt += nr {
				jEnd := jt + nr
				if jEnd > j1 {
					jEnd = j1
				}
				if cols := jEnd - jt; cols < nr {
					// 列尾块：打包后走标量（列方向不满 16，占比极小）
					panel := make([]float32, kk*cols)
					for p := 0; p < kk; p++ {
						copy(panel[p*cols:(p+1)*cols], b[(k0+p)*ldb+jt:(k0+p)*ldb+jEnd])
					}
					for i := 0; i < m; i++ {
						scalarRowGen(a, panel, c, i, jt, cols, k0, kk, lda, ldc)
					}
					continue
				}
				bp := &b[k0*ldb+jt]
				for t := 0; t < nTiles; t++ {
					micro6x16sRows(&a[t*mr*lda+k0], bp, &c[t*mr*ldc+jt], kk, lda, ldc, ldb)
				}
				if tailRows > 0 {
					tailBase := k0 * tailRows
					tailC := &c[nTiles*mr*ldc+jt]
					switch tailRows {
					case 4:
						micro4x16s(&tailAP[tailBase], bp, tailC, kk, ldc, ldb)
					case 2:
						micro2x16s(&tailAP[tailBase], bp, tailC, kk, ldc, ldb)
					case 1:
						micro1x16s(&tailAP[tailBase], bp, tailC, kk, ldc, ldb)
					default:
						for i := nTiles * mr; i < m; i++ {
							panel := make([]float32, kk*nr)
							for p := 0; p < kk; p++ {
								copy(panel[p*nr:(p+1)*nr], b[(k0+p)*ldb+jt:(k0+p)*ldb+jt+nr])
							}
							for ii := i; ii < m; ii++ {
								for jj := 0; jj < nr; jj++ {
									var acc float32
									for p := 0; p < kk; p++ {
										acc += a[ii*lda+k0+p] * panel[p*nr+jj]
									}
									c[ii*ldc+jt+jj] += acc
								}
							}
							goto nextTile
						}
					}
				}
			nextTile:
			}
		}
	}

	jBlocks := (n + nc - 1) / nc
	if jBlocks == 1 || workers == 1 {
		run(0, n)
		return
	}
	if workers > jBlocks {
		workers = jBlocks
	}
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		lo := w * jBlocks / workers * nc
		hi := (w + 1) * jBlocks / workers * nc
		if hi > n {
			hi = n
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			run(lo, hi)
		}(lo, hi)
	}
	wg.Wait()
}

// apPool 复用 packA 的微面板缓冲（m*k float32/次，推理期最大单类分配 33%）。
// 池内缓冲按需扩容、只增不减；sync.Pool 由 GC 自动清理。
// 归还约束：Put 前调用方必须完成对该缓冲的全部读取（含 worker goroutine）。
var apPool = sync.Pool{New: func() any { return new([]float32) }}

func getAp(n int) []float32 {
	if v := apPool.Get(); v != nil {
		if buf := *(v.(*[]float32)); cap(buf) >= n {
			return buf[:n]
		}
	}
	return make([]float32, n)
}

func putAp(ap []float32) {
	if ap != nil {
		apPool.Put(&ap)
	}
}

// packAInto 把 A 打包进调用方提供的缓冲（长度须 ≥ m*k，结果恒为 m*k 元素）：
// 6 行一组按 k 交织，M 方向不足 6 行的尾块单独打包为 tailRows 行交织。
// 打包内容与新建缓冲逐位相同，仅生命周期改为池化。
func packAInto(dst []float32, a []float32, m, k, lda int) (ap []float32, nTiles, tailRows int) {
	nTiles = m / mr
	tailRows = m - nTiles*mr
	ap = dst[:nTiles*mr*k+tailRows*k]
	for t := 0; t < nTiles; t++ {
		base := t * mr * k
		for r := 0; r < mr; r++ {
			row := a[(t*mr+r)*lda:]
			for p := 0; p < k; p++ {
				ap[base+p*mr+r] = row[p]
			}
		}
	}
	if tailRows > 0 {
		base := nTiles * mr * k
		for r := 0; r < tailRows; r++ {
			row := a[(nTiles*mr+r)*lda:]
			for p := 0; p < k; p++ {
				ap[base+p*tailRows+r] = row[p]
			}
		}
	}
	return ap, nTiles, tailRows
}

// SgemmGather 与 Sgemm 相同，但 B 由 fill 现场生成（不要求 B 实际存在）。
// 累加顺序仍为 p 递增，因此结果与 Sgemm 完全等价。
func SgemmGather(m, n, k int, a []float32, lda int, fill BPanelFiller, c []float32, ldc int, workers int) {
	if m <= 0 || n <= 0 || k <= 0 {
		return
	}
	if lda < k || ldc < n {
		panic("kernels.Sgemm: 跨度过小")
	}
	if workers < 1 {
		workers = 1
	}
	if !simdAvailable() {
		// 回退路径：先把 B 物化出来再做标量 GEMM（数值等价，性能较低）
		col := make([]float32, k*n)
		fill(0, k, 0, n, col)
		gemmRowsGo(a, m, k, lda, col, n, n, c, ldc, workers)
		return
	}

	// 打包 A 微面板：按 6 行一组、k 交织，ap[t*6*k + p*6 + r] = A[(6t+r)*lda + p]。
	// 每组 6 行对应一次微内核调用；不足 6 行的尾部由标量路径处理。
	// 池化缓冲：全部读取（含 worker goroutine）都在 wg.Wait/同步调用结束后
	// 才返回，defer 归还安全。
	apBuf := getAp(m * k)
	ap, nTiles, tailRows := packAInto(apBuf, a, m, k, lda)
	defer putAp(ap)
	if nTiles == 0 && tailRows < 1 {
		col := make([]float32, k*n)
		fill(0, k, 0, n, col)
		gemmRowsGo(a, m, k, lda, col, n, n, c, ldc, workers)
		return
	}

	jBlocks := (n + nc - 1) / nc
	if jBlocks == 1 || workers == 1 {
		workerRange(ap, a, fill, c, m, n, k, lda, ldc, 0, n, nTiles, tailRows)
		return
	}
	if workers > jBlocks {
		workers = jBlocks
	}
	var wg sync.WaitGroup
	for w := 0; w < workers; w++ {
		lo := w * jBlocks / workers * nc
		hi := (w + 1) * jBlocks / workers * nc
		if hi > n {
			hi = n
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			workerRange(ap, a, fill, c, m, n, k, lda, ldc, lo, hi, nTiles, tailRows)
		}(lo, hi)
	}
	wg.Wait()
}

// panelWidth 是 B 打包的列宽：一次打包 32 列（两个 16 列微内核共用），
// 相比 16 列可把打包拷贝的调用次数减半，同时让微内核连续访问同一段面板。
const panelWidth = nr * 2

// workerRange 处理列区间 [j0, j1)。ap 为交织后的 A 微面板，a 为原始 A（标量尾块用）。
func workerRange(ap, a []float32, fill BPanelFiller, c []float32, m, n, k, lda, ldc, j0, j1, nTiles, tailRows int) {
	bp := make([]float32, kc*panelWidth)
	for k0 := 0; k0 < k; k0 += kc {
		k1 := k0 + kc
		if k1 > k {
			k1 = k
		}
		kk := k1 - k0
		for jt := j0; jt < j1; jt += panelWidth {
			jEnd := jt + panelWidth
			if jEnd > j1 {
				jEnd = j1
			}
			cols := jEnd - jt
			// 现场生成 B 面板（卷积在此完成带 padding 的取值）
			panel := bp[:kk*cols]
			fill(k0, kk, jt, cols, panel)
			fullCols := cols &^ (nr - 1)
			for sub := 0; sub < fullCols; sub += nr {
				for t := 0; t < nTiles; t++ {
					micro6x16s(&ap[t*mr*k+k0*mr], &panel[sub], &c[t*mr*ldc+jt+sub], kk, ldc, cols)
				}
				if tailRows > 0 {
					tailBase := nTiles*mr*k + k0*tailRows
					tailC := &c[nTiles*mr*ldc+jt+sub]
					switch tailRows {
					case 4:
						micro4x16s(&ap[tailBase], &panel[sub], tailC, kk, ldc, cols)
					case 2:
						micro2x16s(&ap[tailBase], &panel[sub], tailC, kk, ldc, cols)
					case 1:
						micro1x16s(&ap[tailBase], &panel[sub], tailC, kk, ldc, cols)
					}
				}
			}
			// 3/5 行没有对应微内核，只让这些尾行走标量；完整 16 列块不再整段退化。
			if tailRows == 3 || tailRows == 5 {
				for i := nTiles * mr; i < m; i++ {
					scalarRowGenStride(a, panel, c, i, jt, fullCols, cols, 0, k0, kk, lda, ldc)
				}
			}
			if tailCols := cols - fullCols; tailCols > 0 {
				for i := 0; i < m; i++ {
					scalarRowGenStride(a, panel, c, i, jt+fullCols, tailCols, cols, fullCols, k0, kk, lda, ldc)
				}
			}
		}
	}
}

// scalarRowGen 是标量尾块路径：C[i, jt:jt+cols] += Σ_p A[i,p]·panel[p][jj]。
func scalarRowGen(a, panel, c []float32, i, jt, cols, k0, kk, lda, ldc int) {
	scalarRowGenStride(a, panel, c, i, jt, cols, cols, 0, k0, kk, lda, ldc)
}

// scalarRowGenStride 与 scalarRowGen 相同，但 panel 的行跨度和起始列可以独立指定。
func scalarRowGenStride(a, panel, c []float32, i, jt, cols, panelStride, panelCol, k0, kk, lda, ldc int) {
	aRow := a[i*lda+k0 : i*lda+k0+kk]
	for jj := 0; jj < cols; jj++ {
		var acc float32
		for p := 0; p < kk; p++ {
			acc += aRow[p] * panel[p*panelStride+panelCol+jj]
		}
		c[i*ldc+jt+jj] += acc
	}
}

// gemmRowsGo 是纯 Go 回退实现（4×4 分块，无 SIMD）。
func gemmRowsGo(a []float32, m, k, lda int, b []float32, n, ldb int, c []float32, ldc int, workers int) {
	rowsPer := (m + workers - 1) / workers
	rowsPer = (rowsPer + 3) / 4 * 4
	if rowsPer > m {
		rowsPer = m
	}
	if workers == 1 || rowsPer >= m {
		goKernel(a, m, k, lda, b, n, ldb, c, ldc)
		return
	}
	var wg sync.WaitGroup
	for lo := 0; lo < m; lo += rowsPer {
		hi := lo + rowsPer
		if hi > m {
			hi = m
		}
		wg.Add(1)
		go func(lo, hi int) {
			defer wg.Done()
			goKernel(a[lo*lda:], hi-lo, k, lda, b, n, ldb, c[lo*ldc:], ldc)
		}(lo, hi)
	}
	wg.Wait()
}

func goKernel(a []float32, m, k, lda int, b []float32, n, ldb int, c []float32, ldc int) {
	i := 0
	for ; i+4 <= m; i += 4 {
		a0 := a[i*lda : i*lda+k]
		a1 := a[(i+1)*lda : (i+1)*lda+k]
		a2 := a[(i+2)*lda : (i+2)*lda+k]
		a3 := a[(i+3)*lda : (i+3)*lda+k]
		jmax := (n / 4) * 4
		for j := 0; j < jmax; j += 4 {
			var c00, c01, c02, c03, c10, c11, c12, c13 float32
			var c20, c21, c22, c23, c30, c31, c32, c33 float32
			for p := 0; p < k; p++ {
				bp := b[p*ldb+j : p*ldb+j+4]
				b0, b1, b2, b3 := bp[0], bp[1], bp[2], bp[3]
				x0 := a0[p]
				x1 := a1[p]
				x2 := a2[p]
				x3 := a3[p]
				c00 += x0 * b0
				c01 += x0 * b1
				c02 += x0 * b2
				c03 += x0 * b3
				c10 += x1 * b0
				c11 += x1 * b1
				c12 += x1 * b2
				c13 += x1 * b3
				c20 += x2 * b0
				c21 += x2 * b1
				c22 += x2 * b2
				c23 += x2 * b3
				c30 += x3 * b0
				c31 += x3 * b1
				c32 += x3 * b2
				c33 += x3 * b3
			}
			c[i*ldc+j] += c00
			c[i*ldc+j+1] += c01
			c[i*ldc+j+2] += c02
			c[i*ldc+j+3] += c03
			c[(i+1)*ldc+j] += c10
			c[(i+1)*ldc+j+1] += c11
			c[(i+1)*ldc+j+2] += c12
			c[(i+1)*ldc+j+3] += c13
			c[(i+2)*ldc+j] += c20
			c[(i+2)*ldc+j+1] += c21
			c[(i+2)*ldc+j+2] += c22
			c[(i+2)*ldc+j+3] += c23
			c[(i+3)*ldc+j] += c30
			c[(i+3)*ldc+j+1] += c31
			c[(i+3)*ldc+j+2] += c32
			c[(i+3)*ldc+j+3] += c33
		}
		for j := jmax; j < n; j++ {
			var s0, s1, s2, s3 float32
			for p := 0; p < k; p++ {
				bv := b[p*ldb+j]
				s0 += a0[p] * bv
				s1 += a1[p] * bv
				s2 += a2[p] * bv
				s3 += a3[p] * bv
			}
			c[i*ldc+j] += s0
			c[(i+1)*ldc+j] += s1
			c[(i+2)*ldc+j] += s2
			c[(i+3)*ldc+j] += s3
		}
	}
	for ; i < m; i++ {
		ai := a[i*lda : i*lda+k]
		for j := 0; j < n; j++ {
			var s float32
			for p := 0; p < k; p++ {
				s += ai[p] * b[p*ldb+j]
			}
			c[i*ldc+j] += s
		}
	}
}

// ---- 3×3 / stride=1 的"按 tap 累加"卷积核心 ----
//
// 把 3×3 卷积拆成 9 个位移窗口的 GEMM（K = InC）：B 的行是通道、列是同一输出行内
// 连续像素，直接读带 padding 的行窗口缓冲，不做 im2col、也不做 B 面板打包。
// 9 个 tap 都累加到同一段输出像素上，只要该像素段足够短（默认 64），
// 输出 tile 就留在 L1 内，不产生额外内存流量。
//
// 累加顺序是"tap 外层、通道内层"，与 im2col（通道外层）不同，差异在 fp32 舍入噪声内。

// TapPlan 保存 9 个 tap 的权重面板（供按 tap 累加的卷积核心使用）。
type TapPlan struct {
	// AP[tap*Stride + t*6*InC + ic*6 + r]：第 tap 个位移、第 t 个 6 行微块、
	// 第 ic 个输入通道、第 r 个输出通道的权重。
	AP   []float32
	W    []float32 // 原始 OIHW 权重（标量尾块用）
	InC  int
	OutC int
	// Stride 每个 tap 的面板长度；NTiles 为 6 行微块数；TailRows 为尾部行数（0..5）。
	Stride   int
	NTiles   int
	TailRows int
}

// weight 返回第 tap 个位移、输出通道 o、输入通道 ic 的权重。
func (p *TapPlan) weight(tap, o, ic int) float32 {
	ky := tap / 3
	kx := tap - ky*3
	return p.W[o*p.InC*9+ic*9+ky*3+kx]
}

// Conv3x3Accumulate 把行窗口 window（布局 [InC][3][rowStride]）按 9 个 tap
// 累加进输出 c 的像素段 [cOff, cOff+nPix)。c 的行跨度为 ldc（像素平面大小）。
func (p *TapPlan) Conv3x3Accumulate(window []float32, windowPlane, rowStride int, c []float32, cOff, ldc, nPix int) {
	const seg = 64
	for start := 0; start < nPix; start += seg {
		end := start + seg
		if end > nPix {
			end = nPix
		}
		for tap := 0; tap < 9; tap++ {
			ky := tap / 3
			kx := tap - ky*3
			base := ky*rowStride + kx
			apT := p.AP[tap*p.Stride:]
			for ox := start; ox < end; ox += nr {
				cols := end - ox
				if cols > nr {
					cols = nr
				}
				bOff := base + ox
				if cols == nr {
					for t := 0; t < p.NTiles; t++ {
						micro6x16s(&apT[t*mr*p.InC], &window[bOff], &c[cOff+t*mr*ldc+ox], p.InC, ldc, windowPlane)
					}
					tailAP := &apT[p.NTiles*mr*p.InC]
					tailC := &c[cOff+p.NTiles*mr*ldc+ox]
					switch p.TailRows {
					case 4:
						micro4x16s(tailAP, &window[bOff], tailC, p.InC, ldc, windowPlane)
					case 2:
						micro2x16s(tailAP, &window[bOff], tailC, p.InC, ldc, windowPlane)
					case 1:
						micro1x16s(tailAP, &window[bOff], tailC, p.InC, ldc, windowPlane)
					case 3, 5:
						for r := p.NTiles * mr; r < p.OutC; r++ {
							row := c[cOff+r*ldc+ox : cOff+r*ldc+ox+nr]
							for ic := 0; ic < p.InC; ic++ {
								w := p.weight(tap, r, ic)
								src := window[bOff+ic*windowPlane:]
								for j := 0; j < nr; j++ {
									row[j] += w * src[j]
								}
							}
						}
					}
					continue
				}
				// 列尾块（不足 16 列）：逐元素累加
				for r := 0; r < p.OutC; r++ {
					row := c[cOff+r*ldc+ox : cOff+r*ldc+ox+cols]
					for ic := 0; ic < p.InC; ic++ {
						w := p.weight(tap, r, ic)
						src := window[bOff+ic*windowPlane:]
						for j := 0; j < cols; j++ {
							row[j] += w * src[j]
						}
					}
				}
			}
		}
	}
}
