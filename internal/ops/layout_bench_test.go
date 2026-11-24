package ops

import (
	"sync"
	"testing"

	"github.com/nivrrex/trufor-go/internal/tensor"
)

// 这组基准用于回答一个具体问题：把内部布局从 NCHW 换成 NHWC（通道连续）是否能更快地做 3×3 卷积。
// 两者都保证数值等价（同一累加顺序），因此只比较访存与指令开销。
//
// 原型实现：NCHW --分块转置--> NHWC，按 9 个位移分别做 (pixels×InC)·(InC×Cout) 的 GEMM 累加，
// 再 NHWC --分块转置--> NCHW。这正是 oneDNN 式 NHWC 卷积的最小骨架（不含 padding 特化）。

func benchConv(b *testing.B, inC, outC, h, w int, run func(x *tensor.Tensor) error) {
	b.Helper()
	x := tensor.New(inC, h, w)
	for i := range x.Data {
		x.Data[i] = float32(i%251) / 251
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if err := run(x); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkConv3x3NCHWTaps(b *testing.B) {
	const inC, outC, size = 64, 64, 512
	wgt := make([]float32, outC*inC*9)
	for i := range wgt {
		wgt[i] = float32(i%97) * 1e-3
	}
	bias := make([]float32, outC)
	conv, err := NewConv2D(wgt, outC, inC, 3, 3, 1, 1, 1, bias)
	if err != nil {
		b.Fatal(err)
	}
	benchConv(b, inC, outC, size, size, func(x *tensor.Tensor) error {
		_, err := conv.Forward(x, 4)
		return err
	})
}

func BenchmarkConv3x3NHWCProto(b *testing.B) {
	const inC, outC, size = 64, 64, 512
	wgt := make([]float32, outC*inC*9)
	for i := range wgt {
		wgt[i] = float32(i%97) * 1e-3
	}
	// 每个 tap 的权重转成 (InC × OutC) 供 GEMM 的 B 使用
	tapW := make([][]float32, 9)
	for tap := 0; tap < 9; tap++ {
		ky, kx := tap/3, tap%3
		m := make([]float32, inC*outC)
		for o := 0; o < outC; o++ {
			for ic := 0; ic < inC; ic++ {
				m[ic*outC+o] = wgt[o*inC*9+ic*9+ky*3+kx]
			}
		}
		tapW[tap] = m
	}
	x := tensor.New(inC, size, size)
	for i := range x.Data {
		x.Data[i] = float32(i%251) / 251
	}
	// NHWC 工作需要提前转置一次（与真实实现一致：每层一次转置 + 一次回转）
	nhwc := x.ToTokens()
	nhwcPadded := make([]float32, ((size+2)*(size+2))*inC)
	// 每行前后各留 1 个零像素的 padding
	for y := 0; y < size; y++ {
		copy(nhwcPadded[((y+1)*(size+2)+1)*inC:((y+1)*(size+2)+1+size)*inC], nhwc[y*size*inC:(y+1)*size*inC])
	}
	outNHWC := make([]float32, size*size*outC)

	b.ResetTimer()
	for n := 0; n < b.N; n++ {
		for i := range outNHWC {
			outNHWC[i] = 0
		}
		// 按输出行分块，块内 9 个 tap 累加（C tile 留在缓存里）；行块之间并行
		const rowBlock = 8
		const workers = 4
		var wg sync.WaitGroup
		sem := make(chan struct{}, workers)
		for y0 := 0; y0 < size; y0 += rowBlock {
			y1 := y0 + rowBlock
			if y1 > size {
				y1 = size
			}
			wg.Add(1)
			sem <- struct{}{}
			go func(y0, y1 int) {
				defer wg.Done()
				defer func() { <-sem }()
				for tap := 0; tap < 9; tap++ {
					ky, kx := tap/3, tap%3
					for y := y0; y < y1; y++ {
						// B = NHWC 窗口行：[pixels][InC]
						src := ((y+ky)*(size+2) + kx) * inC
						dst := (y * size) * outC
						GemmStrided(nhwcPadded[src:src+size*inC], size, inC, tapW[tap], outC, outNHWC[dst:dst+size*outC], outC, 1)
					}
				}
			}(y0, y1)
		}
		wg.Wait()
		// 回转为 NCHW（真实实现同样需要）
		_ = tensor.FromTokens(outNHWC, outC, size, size)
	}
}

// BenchmarkTransposeNCHWTokens 单独测一次布局转换的开销（NHWC 方案每层要付两次）。
func BenchmarkTransposeNCHWTokens(b *testing.B) {
	const c, size = 64, 512
	x := tensor.New(c, size, size)
	for i := range x.Data {
		x.Data[i] = float32(i%251) / 251
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		tok := x.ToTokens()
		_ = tensor.FromTokens(tok, c, size, size)
	}
}
