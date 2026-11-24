package ops

import (
	"math"
	"math/rand"
	"testing"

	"github.com/nivrrex/trufor-go/internal/tensor"
)

// TestWinogradAccuracy 量化 Winograd F(2×2,3×3) 相对直接卷积的误差。
func TestWinogradAccuracy(t *testing.T) {
	rng := rand.New(rand.NewSource(41))
	for _, tc := range []struct {
		name            string
		inC, outC, h, w int
	}{
		{"64x64-33x45", 64, 64, 33, 45},
		{"64x64-32x32", 64, 64, 32, 32},
		{"16x32-17x19", 16, 32, 17, 19},
		{"64x64-9x11", 64, 64, 9, 11},
	} {
		x := tensor.New(tc.inC, tc.h, tc.w)
		for i := range x.Data {
			x.Data[i] = float32(rng.NormFloat64())
		}
		wgt := make([]float32, tc.outC*tc.inC*9)
		for i := range wgt {
			wgt[i] = float32(rng.NormFloat64()) * 0.1
		}
		bias := make([]float32, tc.outC)
		conv, err := NewConv2D(wgt, tc.outC, tc.inC, 3, 3, 1, 1, 1, bias)
		if err != nil {
			t.Fatal(err)
		}
		conv.winoEnabled = true
		if !conv.useWinograd(x.H, x.W) {
			t.Fatalf("%s: 未走 Winograd 路径", tc.name)
		}
		wino, err := conv.Forward(x, 1)
		if err != nil {
			t.Fatal(err)
		}
		conv.winoEnabled = false
		taps, err := conv.Forward(x, 1)
		if err != nil {
			t.Fatal(err)
		}
		var maxAbs, maxRel float64
		for i := range taps.Data {
			d := math.Abs(float64(wino.Data[i]) - float64(taps.Data[i]))
			if d > maxAbs {
				maxAbs = d
			}
			scale := math.Abs(float64(taps.Data[i]))
			if scale < 1 {
				scale = 1
			}
			if r := d / scale; r > maxRel {
				maxRel = r
			}
		}
		t.Logf("%-14s Winograd vs 直接卷积：maxAbs=%.3e maxRel=%.3e", tc.name, maxAbs, maxRel)
		if maxRel > 1e-5 {
			t.Fatalf("%s: Winograd 误差 %.3e 超过 1e-5", tc.name, maxRel)
		}
	}
}

// BenchmarkWinogradVsTaps 对比 512²、64→64 的单层耗时。
func BenchmarkWinogradVsTaps(b *testing.B) {
	const inC, outC, size = 64, 64, 512
	wgt := make([]float32, outC*inC*9)
	for i := range wgt {
		wgt[i] = float32(i%97) * 1e-3
	}
	x := tensor.New(inC, size, size)
	for i := range x.Data {
		x.Data[i] = float32(i%251) / 251
	}
	for _, mode := range []string{"winograd", "taps"} {
		conv, err := NewConv2D(wgt, outC, inC, 3, 3, 1, 1, 1, nil)
		if err != nil {
			b.Fatal(err)
		}
		conv.winoEnabled = mode == "winograd"
		b.Run(mode, func(b *testing.B) {
			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := conv.Forward(x, 4); err != nil {
					b.Fatal(err)
				}
			}
		})
	}
}
