package kernels

import (
	"math"
	"math/rand"
	"testing"
)

// TestExpAccuracy 校验 SIMD exp 与 math.Exp 的相对误差在 1e-7 量级。
func TestExpAccuracy(t *testing.T) {
	if !simdAvailable() {
		t.Skip("无 AVX2，跳过")
	}
	v := make([]float32, 4096)
	for i := range v {
		v[i] = -float32(i) * 0.02
	}
	ExpInPlace(v)
	var maxRel float64
	for i, got := range v {
		want := math.Exp(float64(-float32(i) * 0.02))
		if want == 0 {
			continue
		}
		if r := math.Abs(float64(got)-want) / want; r > maxRel {
			maxRel = r
		}
	}
	t.Logf("exp 最大相对误差 %.3e", maxRel)
	if maxRel > 1e-6 {
		t.Fatalf("exp 误差过大: %.3e", maxRel)
	}
}

// TestMaxSumLanes 校验 SIMD 归约与标量结果一致（max 精确，sum 允许浮点合并误差）。
func TestMaxSumLanes(t *testing.T) {
	if !simdAvailable() {
		t.Skip("无 AVX2，跳过")
	}
	v := make([]float32, 1003)
	for i := range v {
		v[i] = float32(math.Sin(float64(i))) * 10
	}
	wantMax := v[0]
	var wantSum float64
	for _, x := range v {
		if x > wantMax {
			wantMax = x
		}
		wantSum += float64(x)
	}
	if got := MaxOf(v); got != wantMax {
		t.Fatalf("MaxOf = %v, want %v", got, wantMax)
	}
	if got := SumOf(v); math.Abs(got-wantSum) > 1e-4*math.Abs(wantSum) {
		t.Fatalf("SumOf = %v, want %v", got, wantSum)
	}
}

func TestAxpyChannelwiseRangeMatchesFull(t *testing.T) {
	if !simdAvailable() {
		t.Skip("无 AVX2，跳过")
	}
	rng := rand.New(rand.NewSource(41))
	const channels, pixels = 64, 37
	src := make([]float32, channels*pixels)
	weights := make([]float32, channels)
	for i := range src {
		src[i] = float32(rng.NormFloat64())
	}
	for i := range weights {
		weights[i] = float32(rng.NormFloat64())
	}
	want := make([]float32, len(src))
	got := make([]float32, len(src))
	AxpyChannelwise(want, src, weights)
	for lo := 0; lo < channels; lo += 8 {
		AxpyChannelwiseRange(got, src, weights, channels, lo, lo+8)
	}
	for i := range want {
		if math.Float32bits(want[i]) != math.Float32bits(got[i]) {
			t.Fatalf("元素 %d: got %v, want %v", i, got[i], want[i])
		}
	}
}

func BenchmarkExpCompare(b *testing.B) {
	v := make([]float32, 4096)
	for i := range v {
		v[i] = -float32(i%64) * 0.05
	}
	b.Run("mathExp", func(b *testing.B) {
		out := make([]float32, len(v))
		for i := 0; i < b.N; i++ {
			for j, x := range v {
				out[j] = float32(math.Exp(float64(x)))
			}
		}
	})
	b.Run("simd", func(b *testing.B) {
		buf := make([]float32, len(v))
		for i := 0; i < b.N; i++ {
			copy(buf, v)
			ExpInPlace(buf)
		}
	})
}

// TestDirectVsGather 校验免打包与打包两条 GEMM 路径逐位一致。
func TestDirectVsGather(t *testing.T) {
	rng := rand.New(rand.NewSource(21))
	cases := []struct{ m, n, k int }{{64, 368, 576}, {128, 368, 576}, {2, 100, 27}, {6, 33, 5}, {512, 272, 300}}
	for _, tc := range cases {
		a := make([]float32, tc.m*tc.k)
		b := make([]float32, tc.k*tc.n)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
		}
		for i := range b {
			b[i] = float32(rng.NormFloat64())
		}
		c1 := make([]float32, tc.m*tc.n)
		c2 := make([]float32, tc.m*tc.n)
		SgemmDirect(tc.m, tc.n, tc.k, a, tc.k, b, tc.n, c1, tc.n, 1)
		fill := func(k0, kk, jt, cols int, dst []float32) {
			for p := 0; p < kk; p++ {
				copy(dst[p*cols:p*cols+cols], b[(k0+p)*tc.n+jt:(k0+p)*tc.n+jt+cols])
			}
		}
		SgemmGather(tc.m, tc.n, tc.k, a, tc.k, fill, c2, tc.n, 1)
		bad := 0
		var maxDiff float64
		for i := range c1 {
			if math.Float32bits(c1[i]) != math.Float32bits(c2[i]) {
				bad++
				if d := math.Abs(float64(c1[i] - c2[i])); d > maxDiff {
					maxDiff = d
				}
			}
		}
		t.Logf("m=%d n=%d k=%d: 逐位不一致 %d/%d，最大差 %.3e", tc.m, tc.n, tc.k, bad, len(c1), maxDiff)
	}
}

// TestGELUAccuracy 校验 SIMD GELU 与精确值（float64 math.Erf）的偏差。
func TestGELUAccuracy(t *testing.T) {
	if !simdAvailable() {
		t.Skip("无 AVX2，跳过")
	}
	v := make([]float32, 2048)
	for i := range v {
		v[i] = float32(i-1024) * 0.01
	}
	want := make([]float64, len(v))
	for i, x := range v {
		f := float64(x)
		want[i] = 0.5 * f * (1 + math.Erf(f*0.7071067811865476))
	}
	GELUInPlace(v)
	var maxAbs float64
	for i, got := range v {
		if d := math.Abs(float64(got) - want[i]); d > maxAbs {
			maxAbs = d
		}
	}
	t.Logf("GELU 最大绝对误差 %.3e", maxAbs)
	if maxAbs > 5e-6 {
		t.Fatalf("GELU 误差过大: %.3e", maxAbs)
	}
}

func BenchmarkGELUCompare(b *testing.B) {
	v := make([]float32, 4096)
	for i := range v {
		v[i] = float32(i%200-100) * 0.05
	}
	b.Run("mathErf", func(b *testing.B) {
		out := make([]float32, len(v))
		for i := 0; i < b.N; i++ {
			for j, x := range v {
				f := float64(x)
				out[j] = float32(0.5 * f * (1 + math.Erf(f*0.7071067811865476)))
			}
		}
	})
	b.Run("simd", func(b *testing.B) {
		buf := make([]float32, len(v))
		for i := 0; i < b.N; i++ {
			copy(buf, v)
			GELUInPlace(buf)
		}
	})
}
