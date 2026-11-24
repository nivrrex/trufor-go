package ops

import (
	"math"
	"math/rand"
	"testing"

	"github.com/nivrrex/trufor-go/internal/tensor"
)

// 大 K 的 1×1 卷积交叉验证（解码头 linear_fuse 的形状）
func TestConvBigK(t *testing.T) {
	if testing.Short() {
		t.Skip("-short 模式跳过（纯 Go 下约 10s）")
	}
	rng := rand.New(rand.NewSource(7))
	const inC, outC, h, w = 2048, 512, 64, 64
	x := tensor.New(inC, h, w)
	for i := range x.Data {
		x.Data[i] = float32(rng.NormFloat64())
	}
	wgt := make([]float32, outC*inC)
	for i := range wgt {
		wgt[i] = float32(rng.NormFloat64()) * 0.02
	}
	bias := make([]float32, outC)
	conv, err := NewConv2D(wgt, outC, inC, 1, 1, 1, 0, 0, bias)
	if err != nil {
		t.Fatal(err)
	}
	got, err := conv.Forward(x, 4)
	if err != nil {
		t.Fatal(err)
	}
	// 参考：逐像素点积（float64），抽样 200 个位置
	for k := 0; k < 200; k++ {
		oc := k % outC
		px := (k * 37) % (h * w)
		var s float64
		for ic := 0; ic < inC; ic++ {
			s += float64(x.Data[ic*h*w+px]) * float64(wgt[oc*inC+ic])
		}
		want := float32(s)
		if d := math.Abs(float64(got.Data[oc*h*w+px]) - float64(want)); d > 1e-2*(1+math.Abs(float64(want))) {
			t.Fatalf("oc=%d px=%d got=%v want=%v", oc, px, got.Data[oc*h*w+px], want)
		}
	}
	// 并行度一致性
	got1, err := conv.Forward(x, 1)
	if err != nil {
		t.Fatal(err)
	}
	for i := range got.Data {
		if got.Data[i] != got1.Data[i] {
			t.Fatalf("workers=1 与 4 不一致（第 %d 个元素 %v != %v）", i, got1.Data[i], got.Data[i])
		}
	}
}

// Linear 在解码头形状（n=4096 tokens）下的交叉验证
func TestLinearManyTokens(t *testing.T) {
	rng := rand.New(rand.NewSource(11))
	for _, dims := range [][2]int{{512, 512}, {512, 320}, {512, 64}} {
		out, in := dims[0], dims[1]
		const n = 4096
		src := make([]float32, n*in)
		for i := range src {
			src[i] = float32(rng.NormFloat64())
		}
		wgt := make([]float32, out*in)
		for i := range wgt {
			wgt[i] = float32(rng.NormFloat64()) * 0.02
		}
		lin, err := NewLinear(wgt, out, in, nil)
		if err != nil {
			t.Fatal(err)
		}
		dst := make([]float32, n*out)
		if err := lin.Forward(src, n, dst, 4); err != nil {
			t.Fatal(err)
		}
		for k := 0; k < 100; k++ {
			tok := (k * 41) % n
			oc := (k * 17) % out
			var s float64
			for p := 0; p < in; p++ {
				s += float64(src[tok*in+p]) * float64(wgt[oc*in+p])
			}
			if d := math.Abs(float64(dst[tok*out+oc]) - s); d > 1e-2*(1+math.Abs(s)) {
				t.Fatalf("%dx%d tok=%d oc=%d got=%v want=%v", out, in, tok, oc, dst[tok*out+oc], s)
			}
		}
	}
}
