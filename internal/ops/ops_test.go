package ops

import (
	"math"
	"math/rand"
	"testing"

	"github.com/nivrrex/trufor-go/internal/tensor"
)

// naiveGemm 是独立的对照实现（不做分块，顺序累加）。
func naiveGemm(a []float32, m, k int, b []float32, n int) []float32 {
	c := make([]float32, m*n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			var s float64
			for p := 0; p < k; p++ {
				s += float64(a[i*k+p]) * float64(b[p*n+j])
			}
			c[i*n+j] = float32(s)
		}
	}
	return c
}

func TestGemmMatchesNaive(t *testing.T) {
	rng := rand.New(rand.NewSource(1))
	cases := []struct{ m, n, k int }{
		{1, 1, 1},
		{3, 5, 7},
		{16, 16, 576},
		{37, 19, 65},
		{64, 1, 9},
		{1, 64, 9},
	}
	for _, tc := range cases {
		a := make([]float32, tc.m*tc.k)
		b := make([]float32, tc.k*tc.n)
		for i := range a {
			a[i] = float32(rng.NormFloat64())
		}
		for i := range b {
			b[i] = float32(rng.NormFloat64())
		}
		got := make([]float32, tc.m*tc.n)
		Gemm(a, tc.m, tc.k, b, tc.n, got, 1)
		want := naiveGemm(a, tc.m, tc.k, b, tc.n)
		checkClose(t, "gemm", got, want)
	}
}

func TestGemmWorkersInvariant(t *testing.T) {
	rng := rand.New(rand.NewSource(2))
	const m, n, k = 96, 64, 129
	a := make([]float32, m*k)
	b := make([]float32, k*n)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	for i := range b {
		b[i] = float32(rng.NormFloat64())
	}
	base := make([]float32, m*n)
	Gemm(a, m, k, b, n, base, 1)
	for _, workers := range []int{2, 3, 4, 8} {
		got := make([]float32, m*n)
		Gemm(a, m, k, b, n, got, workers)
		for i := range base {
			if math.Float32bits(base[i]) != math.Float32bits(got[i]) {
				t.Fatalf("workers=%d 结果不可复现：第 %d 个元素 %v != %v", workers, i, got[i], base[i])
			}
		}
	}
}

func TestGemmStrided(t *testing.T) {
	rng := rand.New(rand.NewSource(3))
	const m, n, k = 12, 7, 5
	a := make([]float32, m*k)
	b := make([]float32, k*n)
	for i := range a {
		a[i] = float32(rng.NormFloat64())
	}
	for i := range b {
		b[i] = float32(rng.NormFloat64())
	}
	const ldc = 11
	c := make([]float32, m*ldc)
	for i := range c {
		if i%ldc < n {
			continue // 有效区域由 Gemm 累加（此处为 0）
		}
		c[i] = 100 // 列越界区域应保持不变
	}
	GemmStrided(a, m, k, b, n, c, ldc, 1)
	want := naiveGemm(a, m, k, b, n)
	for i := 0; i < m; i++ {
		for j := 0; j < n; j++ {
			got := c[i*ldc+j]
			exp := want[i*n+j]
			if math.Abs(float64(got-exp)) > 1e-4*(1+math.Abs(float64(exp))) {
				t.Fatalf("(%d,%d) got %v want %v", i, j, got, exp)
			}
		}
		for j := n; j < ldc; j++ {
			if c[i*ldc+j] != 100 {
				t.Fatalf("越界写入 (%d,%d)", i, j)
			}
		}
	}
}

// directConv 是与 im2col 无关的朴素卷积实现（float64 累加），用于交叉验证。
func directConv(x *tensor.Tensor, w []float32, outC, kh, kw, stride, padH, padW int, bias []float32) *tensor.Tensor {
	oh := (x.H+2*padH-kh)/stride + 1
	ow := (x.W+2*padW-kw)/stride + 1
	out := tensor.New(outC, oh, ow)
	for o := 0; o < outC; o++ {
		for oy := 0; oy < oh; oy++ {
			for ox := 0; ox < ow; ox++ {
				var s float64
				for ic := 0; ic < x.C; ic++ {
					for ky := 0; ky < kh; ky++ {
						iy := oy*stride - padH + ky
						if iy < 0 || iy >= x.H {
							continue
						}
						for kx := 0; kx < kw; kx++ {
							ix := ox*stride - padW + kx
							if ix < 0 || ix >= x.W {
								continue
							}
							wi := ((o*x.C+ic)*kh+ky)*kw + kx
							s += float64(x.At(ic, iy, ix)) * float64(w[wi])
						}
					}
				}
				if bias != nil {
					s += float64(bias[o])
				}
				out.Set(o, oy, ox, float32(s))
			}
		}
	}
	return out
}

func TestConvMatchesDirect(t *testing.T) {
	rng := rand.New(rand.NewSource(4))
	cases := []struct {
		name            string
		inC, outC, h, w int
		kh, kw, stride  int
		padH, padW      int
		withBias        bool
	}{
		{"1x1", 3, 4, 9, 7, 1, 1, 1, 0, 0, true},
		{"3x3-s1-p1", 3, 5, 11, 13, 3, 3, 1, 1, 1, true},
		{"3x3-s2-p1", 4, 6, 17, 15, 3, 3, 2, 1, 1, false},
		{"7x7-s4-p3", 3, 8, 32, 24, 7, 7, 4, 3, 3, true},
		{"3x3-nopad", 2, 3, 10, 9, 3, 3, 1, 0, 0, false},
		{"4x4-s4", 6, 2, 16, 16, 4, 4, 4, 0, 0, true},
	}
	for _, tc := range cases {
		x := tensor.New(tc.inC, tc.h, tc.w)
		for i := range x.Data {
			x.Data[i] = float32(rng.NormFloat64())
		}
		w := make([]float32, tc.outC*tc.inC*tc.kh*tc.kw)
		for i := range w {
			w[i] = float32(rng.NormFloat64())
		}
		var bias []float32
		if tc.withBias {
			bias = make([]float32, tc.outC)
			for i := range bias {
				bias[i] = float32(rng.NormFloat64())
			}
		}
		conv, err := NewConv2D(w, tc.outC, tc.inC, tc.kh, tc.kw, tc.stride, tc.padH, tc.padW, bias)
		if err != nil {
			t.Fatalf("%s: NewConv2D: %v", tc.name, err)
		}
		for _, workers := range []int{1, 4} {
			got, err := conv.Forward(x, workers)
			if err != nil {
				t.Fatalf("%s: Forward: %v", tc.name, err)
			}
			want := directConv(x, w, tc.outC, tc.kh, tc.kw, tc.stride, tc.padH, tc.padW, bias)
			if got.C != want.C || got.H != want.H || got.W != want.W {
				t.Fatalf("%s: 形状 (%d,%d,%d) != (%d,%d,%d)", tc.name, got.C, got.H, got.W, want.C, want.H, want.W)
			}
			checkClose(t, tc.name, got.Data, want.Data)
		}
	}
}

func TestConvRejectsShapeMismatch(t *testing.T) {
	if _, err := NewConv2D(make([]float32, 10), 2, 3, 3, 3, 1, 1, 1, nil); err == nil {
		t.Fatal("权重长度不匹配应报错")
	}
	conv, err := NewConv2D(make([]float32, 2*3*3*3), 2, 3, 3, 3, 1, 1, 1, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := conv.Forward(tensor.New(4, 8, 8), 1); err == nil {
		t.Fatal("输入通道不匹配应报错")
	}
	// 无 padding 的 3×3 卷积要求输入至少 3×3
	noPad, err := NewConv2D(make([]float32, 2*3*3*3), 2, 3, 3, 3, 1, 0, 0, nil)
	if err != nil {
		t.Fatal(err)
	}
	if _, err := noPad.Forward(tensor.New(3, 2, 2), 1); err == nil {
		t.Fatal("输入过小应报错")
	}
}

func TestBatchNorm(t *testing.T) {
	mean := []float32{0.5, -1, 2}
	variance := []float32{0.25, 4, 9}
	gamma := []float32{2, 0.5, -1}
	beta := []float32{0.1, 0.2, 0.3}
	const eps = 1e-5
	bn, err := NewBatchNorm(mean, variance, gamma, beta, eps)
	if err != nil {
		t.Fatal(err)
	}
	x := tensor.New(3, 2, 2)
	for i := range x.Data {
		x.Data[i] = float32(i) * 0.7
	}
	orig := append([]float32(nil), x.Data...)
	if err := bn.Apply(x); err != nil {
		t.Fatal(err)
	}
	plane := 4
	for c := 0; c < 3; c++ {
		for i := 0; i < plane; i++ {
			v := float64(orig[c*plane+i])
			want := (v-float64(mean[c]))/math.Sqrt(float64(variance[c])+eps)*float64(gamma[c]) + float64(beta[c])
			got := float64(x.Data[c*plane+i])
			if math.Abs(got-want) > 1e-5*(1+math.Abs(want)) {
				t.Fatalf("c=%d i=%d got %v want %v", c, i, got, want)
			}
		}
	}
	if err := bn.Apply(tensor.New(4, 1, 1)); err == nil {
		t.Fatal("通道不匹配应报错")
	}
}

func TestReLUInPlace(t *testing.T) {
	x := tensor.NewFromData([]float32{-1, 0, 2, -0.5, 3, -7}, 1, 2, 3)
	ReLUInPlace(x)
	want := []float32{0, 0, 2, 0, 3, 0}
	for i := range want {
		if x.Data[i] != want[i] {
			t.Fatalf("第 %d 个元素 %v != %v", i, x.Data[i], want[i])
		}
	}
}

// checkClose 用相对/绝对混合容差比较两组 fp32（累加顺序不同必然带来 1e-6 级差异）。
func checkClose(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s: 长度 %d != %d", name, len(got), len(want))
	}
	var maxAbs, maxRel float64
	var at int
	for i := range want {
		d := math.Abs(float64(got[i]) - float64(want[i]))
		scale := math.Abs(float64(want[i]))
		if scale < 1 {
			scale = 1
		}
		if d > maxAbs {
			maxAbs, at = d, i
		}
		if r := d / scale; r > maxRel {
			maxRel = r
		}
	}
	const tol = 3e-5
	if maxRel > tol {
		t.Fatalf("%s: 最大相对误差 %g（第 %d 个元素 got=%v want=%v，最大绝对差 %g）", name, maxRel, at, got[at], want[at], maxAbs)
	}
}

// TestConvPathsEquivalent 校验 im2col 物化路径与融合取值路径一致。
//
// 两者共享相同的取值顺序与 k 递增累加顺序，但列方向尾块的分块宽度不同：一个按 16 列、
// 一个按 32 列切分，导致"哪几个像素落在标量尾块（mul+add）而非 FMA 微内核"不同，
// 因此差异在 fp32 舍入噪声量级（实测 < 1e-6），而不是语义差异。
func TestConvPathsEquivalent(t *testing.T) {
	rng := rand.New(rand.NewSource(9))
	cases := []struct {
		name            string
		inC, outC, h, w int
		kh, kw, stride  int
		pad             int
	}{
		{"3x3s1p1", 64, 64, 32, 48, 3, 3, 1, 1},
		{"3x3s2p1", 64, 128, 31, 45, 3, 3, 2, 1},
		{"7x7s4p3", 3, 64, 33, 51, 7, 7, 4, 3},
		{"1x1s1p0", 32, 16, 20, 24, 1, 1, 1, 0},
	}
	for _, tc := range cases {
		x := tensor.New(tc.inC, tc.h, tc.w)
		for i := range x.Data {
			x.Data[i] = float32(rng.NormFloat64())
		}
		wgt := make([]float32, tc.outC*tc.inC*tc.kh*tc.kw)
		for i := range wgt {
			wgt[i] = float32(rng.NormFloat64()) * 0.1
		}
		bias := make([]float32, tc.outC)
		conv, err := NewConv2D(wgt, tc.outC, tc.inC, tc.kh, tc.kw, tc.stride, tc.pad, tc.pad, bias)
		if err != nil {
			t.Fatal(err)
		}
		oh, ow := conv.OutputSize(x.H, x.W)
		a := tensor.New(tc.outC, oh, ow)
		b := tensor.New(tc.outC, oh, ow)
		for lo := 0; lo < oh; lo += 4 {
			hi := lo + 4
			if hi > oh {
				hi = oh
			}
			conv.convRowsIm2col(x, a, lo, hi, 1, nil)
			conv.convRowsFused(x, b, lo, hi, 1, nil)
		}
		conv.finish(a, nil)
		conv.finish(b, nil)
		var maxDiff, maxScale float64
		for i := range a.Data {
			d := math.Abs(float64(a.Data[i]) - float64(b.Data[i]))
			if d > maxDiff {
				maxDiff = d
			}
			if s := math.Abs(float64(a.Data[i])); s > maxScale {
				maxScale = s
			}
		}
		scale := maxScale
		if scale < 1 {
			scale = 1
		}
		if maxDiff/scale > 1e-5 {
			t.Fatalf("%s: 两条卷积路径差异 %.3e（超过 1e-5 的相对阈值）", tc.name, maxDiff/scale)
		}
	}
}

// TestConvTapsPath 校验 3×3/stride1/pad1 的"按 tap 累加"路径与朴素卷积一致。
func TestConvTapsPath(t *testing.T) {
	rng := rand.New(rand.NewSource(31))
	cases := []struct {
		name            string
		inC, outC, h, w int
	}{
		{"inC16-outC16", 16, 16, 13, 17},
		{"inC64-outC64", 64, 64, 9, 11},
		{"inC8-outC4", 8, 4, 7, 5},
		{"inC32-outC70", 32, 70, 6, 9},
	}
	for _, tc := range cases {
		x := tensor.New(tc.inC, tc.h, tc.w)
		for i := range x.Data {
			x.Data[i] = float32(rng.NormFloat64())
		}
		wgt := make([]float32, tc.outC*tc.inC*9)
		for i := range wgt {
			wgt[i] = float32(rng.NormFloat64()) * 0.1
		}
		bias := make([]float32, tc.outC)
		for i := range bias {
			bias[i] = float32(rng.NormFloat64()) * 0.1
		}
		conv, err := NewConv2D(wgt, tc.outC, tc.inC, 3, 3, 1, 1, 1, bias)
		if err != nil {
			t.Fatal(err)
		}
		if !conv.useTaps() {
			t.Fatalf("%s: 未走 tap 路径", tc.name)
		}
		got, err := conv.Forward(x, 2)
		if err != nil {
			t.Fatal(err)
		}
		want := directConv(x, wgt, tc.outC, 3, 3, 1, 1, 1, bias)
		checkClose(t, tc.name, got.Data, want.Data)
	}
}

// TestDepthwiseTokens 校验 token-major 深度卷积与 NCHW 版本一致。
// 两者累加顺序相同，但边界列在 NCHW 版本走标量（mul+add）、在 token 版本走 SIMD（FMA），
// 因此差异在 fp32 舍入噪声量级（< 1e-5 相对）。
func TestDepthwiseTokensParallelDeterministic(t *testing.T) {
	rng := rand.New(rand.NewSource(42))
	const channels, h, w = 64, 64, 64
	tok := make([]float32, channels*h*w)
	weights := make([]float32, channels*9)
	bias := make([]float32, channels)
	for i := range tok {
		tok[i] = float32(rng.NormFloat64())
	}
	for i := range weights {
		weights[i] = float32(rng.NormFloat64()) * 0.2
	}
	for i := range bias {
		bias[i] = float32(rng.NormFloat64()) * 0.1
	}
	conv, err := NewConv2DGrouped(weights, channels, channels, 3, 3, 1, 1, 1, channels, bias)
	if err != nil {
		t.Fatal(err)
	}
	want, err := conv.ForwardTokens(tok, h*w, h, w)
	if err != nil {
		t.Fatal(err)
	}
	got, err := conv.ForwardTokensParallel(tok, h*w, h, w, 4)
	if err != nil {
		t.Fatal(err)
	}
	for i := range want {
		if math.Float32bits(want[i]) != math.Float32bits(got[i]) {
			t.Fatalf("元素 %d: got %v, want %v", i, got[i], want[i])
		}
	}
}

func TestDepthwiseTokens(t *testing.T) {
	rng := rand.New(rand.NewSource(32))
	const inC, h, w = 16, 9, 7
	x := tensor.New(inC, h, w)
	for i := range x.Data {
		x.Data[i] = float32(rng.NormFloat64())
	}
	wgt := make([]float32, inC*9)
	for i := range wgt {
		wgt[i] = float32(rng.NormFloat64()) * 0.2
	}
	bias := make([]float32, inC)
	for i := range bias {
		bias[i] = float32(rng.NormFloat64()) * 0.1
	}
	conv, err := NewConv2DGrouped(wgt, inC, inC, 3, 3, 1, 1, 1, inC, bias)
	if err != nil {
		t.Fatal(err)
	}
	got, err := conv.ForwardGrouped(x, 1)
	if err != nil {
		t.Fatal(err)
	}
	fromTokens, err := conv.ForwardTokens(x.ToTokens(), h*w, h, w)
	if err != nil {
		t.Fatal(err)
	}
	tok := got.ToTokens()
	var maxDiff, scale float64
	for i := range tok {
		d := math.Abs(float64(tok[i]) - float64(fromTokens[i]))
		if d > maxDiff {
			maxDiff = d
		}
		if a := math.Abs(float64(tok[i])); a > scale {
			scale = a
		}
	}
	if scale < 1 {
		scale = 1
	}
	if maxDiff/scale > 1e-5 {
		t.Fatalf("token 路径与 NCHW 路径差异 %.3e 超过 1e-5", maxDiff/scale)
	}
}
