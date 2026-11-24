package kernels

import "math"

// SIMDAvailable 返回是否可用 AVX2+FMA 内核（含 SIMD exp/归约）。
func SIMDAvailable() bool { return simdAvailable() }

// MaxOf 返回切片最大值（SIMD + 标量尾部）。
func MaxOf(v []float32) float32 {
	if len(v) == 0 {
		return float32(math.Inf(-1))
	}
	if !simdAvailable() || len(v) < 16 {
		m := v[0]
		for _, x := range v[1:] {
			if x > m {
				m = x
			}
		}
		return m
	}
	var lanes [8]float32
	n := len(v) &^ 7
	maxLanes(&v[0], n, &lanes[0])
	m := lanes[0]
	for _, x := range lanes[1:] {
		if x > m {
			m = x
		}
	}
	for _, x := range v[n:] {
		if x > m {
			m = x
		}
	}
	return m
}

// SumOf 返回切片之和（SIMD 求部分和、Go 侧按 float64 归约）。
func SumOf(v []float32) float64 {
	if len(v) == 0 {
		return 0
	}
	if !simdAvailable() || len(v) < 16 {
		var s float64
		for _, x := range v {
			s += float64(x)
		}
		return s
	}
	var lanes [8]float32
	n := len(v) &^ 7
	sumLanes(&v[0], n, &lanes[0])
	var s float64
	for _, x := range lanes {
		s += float64(x)
	}
	for _, x := range v[n:] {
		s += float64(x)
	}
	return s
}

// ExpInPlace 就地计算 v[i] = exp(v[i])。
func ExpInPlace(v []float32) {
	ExpShiftInPlace(v, 0)
}

// ExpShiftInPlace 就地计算 v[i] = exp(v[i]-shift)（softmax 的数值稳定形式）。
func ExpShiftInPlace(v []float32, shift float32) {
	if len(v) == 0 {
		return
	}
	if !simdAvailable() {
		for i, x := range v {
			v[i] = expScalar(x - shift)
		}
		return
	}
	i := 0
	for ; i+8 <= len(v); i += 8 {
		subExp8(&v[i], shift)
	}
	for ; i < len(v); i++ {
		v[i] = expScalar(v[i] - shift)
	}
}

// expScalar 是 float32 exp 的标量实现（与 SIMD 版本同一多项式，相对误差 < 1e-7）。
func expScalar(x float32) float32 {
	if x < -87.3 {
		return 0
	}
	if x > 88.7 {
		return float32(math.Inf(1))
	}
	t := x * 1.4426950408889634
	k := int32(t)
	if t-float32(k) > 0.5 {
		k++
	} else if t-float32(k) < -0.5 {
		k--
	}
	z := x - float32(k)*6.93145752e-1
	z -= float32(k) * 1.42860677e-6
	y := float32(1.9875691500e-4)
	y = y*z + 1.3981999507e-3
	y = y*z + 8.3334519073e-3
	y = y*z + 4.1665795894e-2
	y = y*z + 1.6666665459e-1
	y = y*z + 5.0000001201e-1
	y = y*z*z + z + 1
	return math.Float32frombits(uint32(k+127)<<23) * y
}

// Axpy 计算 acc[i] += w*src[i]（用于深度卷积等逐通道累加）。
func Axpy(acc, src []float32, w float32) {
	n := len(acc)
	if len(src) < n {
		n = len(src)
	}
	if n == 0 {
		return
	}
	if !simdAvailable() {
		for i := 0; i < n; i++ {
			acc[i] += w * src[i]
		}
		return
	}
	m := n &^ 7
	if m > 0 {
		axpy(&acc[0], &src[0], m, w)
	}
	for i := m; i < n; i++ {
		acc[i] += w * src[i]
	}
}

// GELUInPlace 就地计算精确 GELU（0.5*x*(1+erf(x/sqrt2))）。
// erf 采用 A&S 7.1.26（|误差| < 1.5e-7），与 math.Erf 的差异在 fp32 舍入噪声内。
func GELUInPlace(v []float32) {
	if len(v) == 0 {
		return
	}
	if !simdAvailable() {
		const invSqrt2 = 0.7071067811865476
		for i, x := range v {
			f := float64(x)
			v[i] = float32(0.5 * f * (1 + math.Erf(f*invSqrt2)))
		}
		return
	}
	i := 0
	for ; i+8 <= len(v); i += 8 {
		gelu8(&v[i], &v[i])
	}
	const invSqrt2 = 0.7071067811865476
	for ; i < len(v); i++ {
		f := float64(v[i])
		v[i] = float32(0.5 * f * (1 + math.Erf(f*invSqrt2)))
	}
}

// AxpyChannelwise 计算 dst[t*C + c] += src[t*C + c] * w[c]，即 token-major 数据上
// 按通道加权的累加（深度卷积在 token-major 布局下的核心运算）。
// n 为 dst/src 的元素数，C = len(w)。
func AxpyChannelwise(dst, src, w []float32) {
	c := len(w)
	if c == 0 {
		return
	}
	n := len(dst)
	if len(src) < n {
		n = len(src)
	}
	n -= n % c
	if n == 0 {
		return
	}
	if !simdAvailable() || c%8 != 0 {
		for off := 0; off < n; off += c {
			d := dst[off : off+c]
			s := src[off : off+c]
			for i := range d {
				d[i] += s[i] * w[i]
			}
		}
		return
	}
	for off := 0; off < n; off += c {
		axpyVec(&dst[off], &src[off], &w[0], c)
	}
}
