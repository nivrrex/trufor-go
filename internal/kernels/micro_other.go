//go:build !amd64

package kernels

import (
	"math"
	"unsafe"
)

// 非 amd64 构建不包含汇编微内核；simdAvailable 恒为 false，以下函数仅用于满足
// 编译期符号解析，正常推理路径不会调用。若被误调用则明确 panic，不静默产出错误值。
func micro6x16(*float32, *float32, *float32, int, int)       { panic("SIMD micro6x16 unavailable") }
func micro6x16s(*float32, *float32, *float32, int, int, int) { panic("SIMD micro6x16s unavailable") }
func micro6x16sRows(*float32, *float32, *float32, int, int, int, int) {
	panic("SIMD micro6x16sRows unavailable")
}
func micro4x16(*float32, *float32, *float32, int, int)       { panic("SIMD micro4x16 unavailable") }
func micro2x16(*float32, *float32, *float32, int, int)       { panic("SIMD micro2x16 unavailable") }
func micro1x16s(*float32, *float32, *float32, int, int, int) { panic("SIMD micro1x16s unavailable") }
func micro4x16s(*float32, *float32, *float32, int, int, int) { panic("SIMD micro4x16s unavailable") }
func micro2x16s(*float32, *float32, *float32, int, int, int) { panic("SIMD micro2x16s unavailable") }

func exp8(x, y *float32) {
	xv, yv := unsafe.Slice(x, 8), unsafe.Slice(y, 8)
	for i := range xv {
		yv[i] = expScalar(xv[i])
	}
}
func maxLanes(x *float32, n int, lanes *float32) {
	xv, lv := unsafe.Slice(x, n), unsafe.Slice(lanes, 8)
	for i := range lv {
		lv[i] = float32(math.Inf(-1))
	}
	for _, v := range xv {
		if v > lv[0] {
			lv[0] = v
		}
	}
}
func sumLanes(x *float32, n int, lanes *float32) {
	xv, lv := unsafe.Slice(x, n), unsafe.Slice(lanes, 8)
	for i := range lv {
		lv[i] = 0
	}
	for i, v := range xv {
		lv[i%8] += v
	}
}
func subExp8(x *float32, shift float32) {
	xv := unsafe.Slice(x, 8)
	for i := range xv {
		xv[i] = expScalar(xv[i] - shift)
	}
}
func axpy(acc, src *float32, n int, w float32) {
	av, sv := unsafe.Slice(acc, n), unsafe.Slice(src, n)
	for i := range av {
		av[i] += w * sv[i]
	}
}
func gelu8(x, y *float32) {
	const invSqrt2 = 0.7071067811865476
	xv, yv := unsafe.Slice(x, 8), unsafe.Slice(y, 8)
	for i := range xv {
		f := float64(xv[i])
		yv[i] = float32(0.5 * f * (1 + math.Erf(f*invSqrt2)))
	}
}
func axpyVec(dst, src, w *float32, n int) {
	dv, sv, wv := unsafe.Slice(dst, n), unsafe.Slice(src, n), unsafe.Slice(w, n)
	for i := range dv {
		dv[i] += wv[i] * sv[i]
	}
}
