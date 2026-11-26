//go:build amd64

package kernels

// micro6x16 是 6×16 的 AVX2+FMA 微内核（实现在 micro_amd64.s）。
func micro6x16(ap, bp, c *float32, kc, ldc int)

// micro6x16s 与 micro6x16 相同，但 B 的行跨度由 bstride 给出（用于跳过 B 打包）。
func micro6x16s(ap, bp, c *float32, kc, ldc, bstride int)

// micro6x16sRows 直接读取 row-major A，避免交错打包。
func micro6x16sRows(a, bp, c *float32, kc, lda, ldc, bstride int)

// micro4x16 / micro2x16 用于 M 方向的尾块（OutC % 6 ∈ {4, 2} 时），
// 避免尾块落到标量路径上成为瓶颈。
func micro4x16(ap, bp, c *float32, kc, ldc int)

func micro2x16(ap, bp, c *float32, kc, ldc int)

// micro1x16s 用于 M%6==1 的尾块（如 DnCNN 最后一层 64→1）。
func micro1x16s(ap, bp, c *float32, kc, ldc, bstride int)

// micro4x16s / micro2x16s 是带 B 行跨度的尾块内核（配合 SgemmDirect 跳过 B 打包）。
func micro4x16s(ap, bp, c *float32, kc, ldc, bstride int)

func micro2x16s(ap, bp, c *float32, kc, ldc, bstride int)

// 以下为 AVX2 的 exp / 归约内核（实现在 exp_amd64.s）。
func exp8(x, y *float32)
func maxLanes(x *float32, n int, lanes *float32)
func sumLanes(x *float32, n int, lanes *float32)
func subExp8(x *float32, shift float32)

// axpy 计算 acc += w*src（AVX2，n 为 8 的倍数）。
func axpy(acc, src *float32, n int, w float32)

// gelu8 对 8 个元素计算 GELU（AVX2 + A&S 7.1.26 erf）。
func gelu8(x, y *float32)

// axpyVec 计算 dst[i] += w[i]*src[i]（AVX2，n 为 8 的倍数）。
func axpyVec(dst, src, w *float32, n int)
