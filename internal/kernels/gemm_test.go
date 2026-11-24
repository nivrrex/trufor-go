package kernels

import (
	"fmt"
	"math"
	"math/rand"
	"testing"
)

// naiveRef 是用 float64 累加的参考实现。
func naiveRef(a []float32, m, k, lda int, b []float32, n, ldb int) []float64 {
	out := make([]float64, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var s float64
			for p := 0; p < k; p++ {
				s += float64(a[i*lda+p]) * float64(b[p*ldb+j])
			}
			out[i*n+j] = s
		}
	}
	return out
}

func TestSgemmMatchesReference(t *testing.T) {
	rng := rand.New(rand.NewSource(5))
	cases := []struct{ m, n, k int }{
		{1, 1, 1},
		{6, 16, 8},
		{7, 17, 256},
		{64, 4096, 576},
		{512, 512, 2048},
		{96, 1000, 300},
		{5, 3, 1000},
	}
	for _, tc := range cases {
		a := make([]float32, tc.m*tc.k)
		b := make([]float32, tc.k*tc.n)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
		}
		for i := range b {
			b[i] = float32(rng.NormFloat64()) * 0.1
		}
		want := naiveRef(a, tc.m, tc.k, tc.k, b, tc.n, tc.n)
		for _, workers := range []int{1, 3, 4} {
			c := make([]float32, tc.m*tc.n)
			Sgemm(tc.m, tc.n, tc.k, a, tc.k, b, tc.n, c, tc.n, workers)
			var maxRel float64
			for i := range c {
				d := math.Abs(float64(c[i]) - want[i])
				scale := math.Abs(want[i])
				if scale < 1 {
					scale = 1
				}
				if r := d / scale; r > maxRel {
					maxRel = r
				}
			}
			// K=2048 时 fp32 累加噪声约 1e-5 量级；真正的实现错误会带来 O(1) 偏差
			if maxRel > 1e-4 {
				t.Fatalf("m=%d n=%d k=%d workers=%d: 最大相对误差 %g", tc.m, tc.n, tc.k, workers, maxRel)
			}
		}
	}
}

// TestSgemmWorkersDeterministic 校验并行度不影响结果（累加顺序固定）。
func TestSgemmWorkersDeterministic(t *testing.T) {
	rng := rand.New(rand.NewSource(6))
	const m, n, k = 96, 4096, 576
	a := make([]float32, m*k)
	b := make([]float32, k*n)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	for i := range b {
		b[i] = float32(rng.NormFloat64())
	}
	base := make([]float32, m*n)
	Sgemm(m, n, k, a, k, b, n, base, n, 1)
	for _, workers := range []int{2, 4, 8} {
		got := make([]float32, m*n)
		Sgemm(m, n, k, a, k, b, n, got, n, workers)
		for i := range base {
			if math.Float32bits(base[i]) != math.Float32bits(got[i]) {
				t.Fatalf("workers=%d 结果不可复现（第 %d 个元素 %v != %v）", workers, i, got[i], base[i])
			}
		}
	}
}

func BenchmarkSgemm(b *testing.B) {
	const m, n, k = 256, 4096, 576
	a := make([]float32, m*k)
	bb := make([]float32, k*n)
	for i := range a {
		a[i] = float32(i%13) * 0.01
	}
	for i := range bb {
		bb[i] = float32(i%7) * 0.01
	}
	c := make([]float32, m*n)
	impl := "go"
	if simdAvailable() {
		impl = "simd"
	}
	for _, workers := range []int{1, 4} {
		b.Run(fmt.Sprintf("%s_w%d", impl, workers), func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				Sgemm(m, n, k, a, k, bb, n, c, n, workers)
			}
			b.StopTimer()
			b.ReportMetric(2.0*m*n*k*float64(b.N)/b.Elapsed().Seconds()/1e9, "GFLOP/s")
		})
	}
}
