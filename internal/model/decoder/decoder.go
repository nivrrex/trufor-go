// Package decoder 实现 TruFor 的 MLPDecoder 解码头（定位头与置信头结构相同，类别数不同）。
//
// 对应上游 models/cmx/decoders/MLPDecoder.py：
//
//	4 级特征 → 各自 1×1 线性投影(→embed_dim) → 双线性上采到 1/4 分辨率 → 通道拼接
//	→ 1×1 卷积(4*embed_dim→embed_dim) + BatchNorm + ReLU → 1×1 卷积(→num_classes)
package decoder

import (
	"fmt"

	"github.com/nivrrex/trufor-go/internal/ops"
	"github.com/nivrrex/trufor-go/internal/tensor"
	"github.com/nivrrex/trufor-go/internal/weights"
)

// Head 是一个 MLPDecoder 解码头。
type Head struct {
	embedDim int
	numClass int
	lin      [4]*ops.Linear
	fuse     *ops.Conv2D
	bn       *ops.BatchNorm
	pred     *ops.Conv2D
	workers  int
}

// New 从权重构造解码头。prefix 形如 "decode_head" 或 "decode_head_conf"。
// bnEps 必须按头分别给出：定位头为 1e-3（BN_EPS 仅写入该头），置信头为默认 1e-5。
func New(w *weights.File, prefix string, inChannels [4]int, numClass, embedDim int, bnEps float64, workers int) (*Head, error) {
	if workers < 1 {
		workers = 1
	}
	get := func(name string) (*weights.Tensor, error) { return w.Get(prefix + "." + name) }
	h := &Head{embedDim: embedDim, numClass: numClass, workers: workers}
	// feats[i] 为 1/4、1/8、1/16、1/32 分辨率，依次对应权重中的 linear_c1..linear_c4
	for i := 0; i < 4; i++ {
		c := i + 1
		wt, err := get(fmt.Sprintf("linear_c%d.proj.weight", c))
		if err != nil {
			return nil, err
		}
		bt, err := get(fmt.Sprintf("linear_c%d.proj.bias", c))
		if err != nil {
			return nil, err
		}
		if len(wt.Shape) != 2 || wt.Shape[0] != embedDim || wt.Shape[1] != inChannels[i] {
			return nil, fmt.Errorf("%s.linear_c%d.proj.weight 形状 %v != (%d,%d)", prefix, c, wt.Shape, embedDim, inChannels[i])
		}
		lin, err := ops.NewLinear(wt.Data, embedDim, inChannels[i], bt.Data)
		if err != nil {
			return nil, err
		}
		h.lin[i] = lin
	}
	fw, err := get("linear_fuse.0.weight")
	if err != nil {
		return nil, err
	}
	fb, err := get("linear_fuse.0.bias")
	if err != nil {
		return nil, err
	}
	if len(fw.Shape) != 4 || fw.Shape[0] != embedDim || fw.Shape[1] != embedDim*4 || fw.Shape[2] != 1 || fw.Shape[3] != 1 {
		return nil, fmt.Errorf("%s.linear_fuse.0.weight 形状 %v != (%d,%d,1,1)", prefix, fw.Shape, embedDim, embedDim*4)
	}
	if h.fuse, err = ops.NewConv2D(fw.Data, embedDim, embedDim*4, 1, 1, 1, 0, 0, fb.Data); err != nil {
		return nil, err
	}
	mean, err := get("linear_fuse.1.running_mean")
	if err != nil {
		return nil, err
	}
	variance, err := get("linear_fuse.1.running_var")
	if err != nil {
		return nil, err
	}
	gamma, err := get("linear_fuse.1.weight")
	if err != nil {
		return nil, err
	}
	beta, err := get("linear_fuse.1.bias")
	if err != nil {
		return nil, err
	}
	if h.bn, err = ops.NewBatchNorm(mean.Data, variance.Data, gamma.Data, beta.Data, bnEps); err != nil {
		return nil, err
	}
	pw, err := get("linear_pred.weight")
	if err != nil {
		return nil, err
	}
	pb, err := get("linear_pred.bias")
	if err != nil {
		return nil, err
	}
	if len(pw.Shape) != 4 || pw.Shape[0] != numClass || pw.Shape[1] != embedDim {
		return nil, fmt.Errorf("%s.linear_pred.weight 形状 %v != (%d,%d,1,1)", prefix, pw.Shape, numClass, embedDim)
	}
	if h.pred, err = ops.NewConv2D(pw.Data, numClass, embedDim, 1, 1, 1, 0, 0, pb.Data); err != nil {
		return nil, err
	}
	return h, nil
}

// Forward 计算 1/4 分辨率的 logits（numClass×h/4×w/4）。
// feats 为 4 级主干特征，feats[0] 为 1/4 分辨率。
func (h *Head) Forward(feats [4]*tensor.Tensor) (*tensor.Tensor, error) {
	var proj [4]*tensor.Tensor
	h0, w0 := feats[0].H, feats[0].W
	for i, f := range feats {
		if f == nil {
			return nil, fmt.Errorf("decoder: 第 %d 级特征为 nil", i)
		}
		if f.C != h.lin[i].In {
			return nil, fmt.Errorf("decoder: 第 %d 级通道 %d != %d", i, f.C, h.lin[i].In)
		}
		n := f.H * f.W
		buf := make([]float32, n*h.embedDim)
		if err := h.lin[i].Forward(f.ToTokens(), n, buf, h.workers); err != nil {
			return nil, err
		}
		proj[i] = tensor.FromTokens(buf, h.embedDim, f.H, f.W)
		if i > 0 {
			up, err := ops.ResizeBilinear(proj[i], h0, w0)
			if err != nil {
				return nil, err
			}
			proj[i] = up
		}
	}

	// 通道拼接顺序与上游 torch.cat([_c4, _c3, _c2, _c1]) 一致：高分辨率特征排在最后
	merged := tensor.New(h.embedDim*4, h0, w0)
	plane := h0 * w0
	for i := 0; i < 4; i++ {
		block := 3 - i
		copy(merged.Data[block*h.embedDim*plane:(block+1)*h.embedDim*plane], proj[i].Data)
	}

	fused, err := h.fuse.ForwardGrouped(merged, h.workers)
	if err != nil {
		return nil, err
	}
	if err := h.bn.Apply(fused); err != nil {
		return nil, err
	}
	ops.ReLUInPlace(fused)
	logits, err := h.pred.ForwardGrouped(fused, h.workers)
	if err != nil {
		return nil, err
	}
	return logits, nil
}
