package ops

import "fmt"

// Linear 是 nn.Linear 的推理实现，输入输出均为 token-major（[n][C]）连续存储。
type Linear struct {
	In, Out int
	wt      []float32 // In×Out（由 Out×In 的权重转置打包而来；作为 GEMM 的 B）
	bias    []float32 // 可为 nil
}

// NewLinear 构造线性层。w 为 Out×In（PyTorch 布局）。
func NewLinear(w []float32, out, in int, bias []float32) (*Linear, error) {
	if out <= 0 || in <= 0 {
		return nil, fmt.Errorf("linear 参数非法: out=%d in=%d", out, in)
	}
	if len(w) != out*in {
		return nil, fmt.Errorf("linear 权重长度 %d != %d (out=%d, in=%d)", len(w), out*in, out, in)
	}
	if bias != nil && len(bias) != out {
		return nil, fmt.Errorf("linear bias 长度 %d != %d", len(bias), out)
	}
	wt := make([]float32, in*out)
	for o := 0; o < out; o++ {
		src := w[o*in : (o+1)*in]
		for p := 0; p < in; p++ {
			wt[p*out+o] = src[p]
		}
	}
	return &Linear{In: in, Out: out, wt: wt, bias: bias}, nil
}

// Forward 计算 dst[n×Out] = src[n×In] · wᵀ + b。dst 必须已分配（会被覆盖写入）。
func (l *Linear) Forward(src []float32, n int, dst []float32, workers int) error {
	if len(src) < n*l.In {
		return fmt.Errorf("linear 输入长度 %d < %d", len(src), n*l.In)
	}
	if len(dst) < n*l.Out {
		return fmt.Errorf("linear 输出长度 %d < %d", len(dst), n*l.Out)
	}
	out := dst[:n*l.Out]
	for i := range out {
		out[i] = 0
	}
	Gemm(src[:n*l.In], n, l.In, l.wt, l.Out, out, workers)
	if l.bias != nil {
		for i := 0; i < n; i++ {
			row := out[i*l.Out : (i+1)*l.Out]
			for o := 0; o < l.Out; o++ {
				row[o] += l.bias[o]
			}
		}
	}
	return nil
}
