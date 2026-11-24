package segformer

import (
	"fmt"

	"github.com/nivrrex/trufor-go/internal/ops"
	"github.com/nivrrex/trufor-go/internal/tensor"
	"github.com/nivrrex/trufor-go/internal/weights"
)

// loader 统一处理权重的取用、形状校验与打包。
type loader struct {
	w      *weights.File
	prefix string
}

func (l *loader) get(name string) (*weights.Tensor, error) {
	return l.w.Get(l.prefix + "." + name)
}

func (l *loader) lin(name string, out, in int) (*ops.Linear, error) {
	wt, err := l.get(name + ".weight")
	if err != nil {
		return nil, err
	}
	if !sameShape(wt.Shape, []int{out, in}) {
		return nil, fmt.Errorf("%s.%s.weight 形状 %v != (%d,%d)", l.prefix, name, wt.Shape, out, in)
	}
	var bias []float32
	bt, err := l.get(name + ".bias")
	if err == nil {
		if !sameShape(bt.Shape, []int{out}) {
			return nil, fmt.Errorf("%s.%s.bias 形状 %v != (%d)", l.prefix, name, bt.Shape, out)
		}
		bias = bt.Data
	}
	return ops.NewLinear(wt.Data, out, in, bias)
}

func (l *loader) linNoBias(name string, out, in int) (*ops.Linear, error) {
	wt, err := l.get(name + ".weight")
	if err != nil {
		return nil, err
	}
	if !sameShape(wt.Shape, []int{out, in}) {
		return nil, fmt.Errorf("%s.%s.weight 形状 %v != (%d,%d)", l.prefix, name, wt.Shape, out, in)
	}
	return ops.NewLinear(wt.Data, out, in, nil)
}

func (l *loader) conv(name string, outC, inC, kh, kw, stride, pad, groups int, withBias bool) (*ops.Conv2D, error) {
	wt, err := l.get(name + ".weight")
	if err != nil {
		return nil, err
	}
	if len(wt.Shape) != 4 || wt.Shape[0] != outC || wt.Shape[1] != inC/groups || wt.Shape[2] != kh || wt.Shape[3] != kw {
		return nil, fmt.Errorf("%s.%s.weight 形状 %v != (%d,%d,%d,%d)", l.prefix, name, wt.Shape, outC, inC/groups, kh, kw)
	}
	var bias []float32
	if withBias {
		bt, err := l.get(name + ".bias")
		if err != nil {
			return nil, err
		}
		if !sameShape(bt.Shape, []int{outC}) {
			return nil, fmt.Errorf("%s.%s.bias 形状 %v != (%d)", l.prefix, name, bt.Shape, outC)
		}
		bias = bt.Data
	}
	return ops.NewConv2DGrouped(wt.Data, outC, inC, kh, kw, stride, pad, pad, groups, bias)
}

func (l *loader) ln(name string, dim int, eps float64) (*ops.LayerNorm, error) {
	g, err := l.get(name + ".weight")
	if err != nil {
		return nil, err
	}
	be, err := l.get(name + ".bias")
	if err != nil {
		return nil, err
	}
	if !sameShape(g.Shape, []int{dim}) || !sameShape(be.Shape, []int{dim}) {
		return nil, fmt.Errorf("%s.%s 归一化参数形状 %v/%v != (%d)", l.prefix, name, g.Shape, be.Shape, dim)
	}
	return ops.NewLayerNorm(g.Data, be.Data, eps)
}

func (l *loader) bn(name string, dim int, eps float64) (*ops.BatchNorm, error) {
	mean, err := l.get(name + ".running_mean")
	if err != nil {
		return nil, err
	}
	variance, err := l.get(name + ".running_var")
	if err != nil {
		return nil, err
	}
	g, err := l.get(name + ".weight")
	if err != nil {
		return nil, err
	}
	be, err := l.get(name + ".bias")
	if err != nil {
		return nil, err
	}
	if !sameShape(mean.Shape, []int{dim}) || !sameShape(variance.Shape, []int{dim}) {
		return nil, fmt.Errorf("%s.%s batchnorm 参数形状异常 %v/%v != (%d)", l.prefix, name, mean.Shape, variance.Shape, dim)
	}
	return ops.NewBatchNorm(mean.Data, variance.Data, g.Data, be.Data, eps)
}

// ---- FRM（Feature Rectify Module） ----

type frm struct {
	cw1, cw2 *ops.Linear
	sw0, sw1 *ops.Conv2D
}

func newFRM(l *loader, stage int) (*frm, error) {
	dim := embedDims[stage]
	cw1, err := l.lin(fmt.Sprintf("FRMs.%d.channel_weights.mlp.0", stage), dim*4, dim*4)
	if err != nil {
		return nil, err
	}
	cw2, err := l.lin(fmt.Sprintf("FRMs.%d.channel_weights.mlp.2", stage), dim*2, dim*4)
	if err != nil {
		return nil, err
	}
	sw0, err := l.conv(fmt.Sprintf("FRMs.%d.spatial_weights.mlp.0", stage), dim, dim*2, 1, 1, 1, 0, 1, true)
	if err != nil {
		return nil, err
	}
	sw1, err := l.conv(fmt.Sprintf("FRMs.%d.spatial_weights.mlp.2", stage), 2, dim, 1, 1, 1, 0, 1, true)
	if err != nil {
		return nil, err
	}
	return &frm{cw1: cw1, cw2: cw2, sw0: sw0, sw1: sw1}, nil
}

// forward 返回修正后的 (x1, x2)。通道权重按全局平均/最大池化统计，空间权重按 1×1 卷积。
func (f *frm) forward(x1, x2 *tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, error) {
	dim := x1.C
	if x2.C != dim || x1.H != x2.H || x1.W != x2.W {
		return nil, nil, fmt.Errorf("FRM 输入形状不一致: %v vs %v", []int{x1.C, x1.H, x1.W}, []int{x2.C, x2.H, x2.W})
	}
	plane := x1.H * x1.W
	if plane == 0 || len(x1.Data) < dim*plane || len(x2.Data) < dim*plane {
		return nil, nil, fmt.Errorf("FRM 数据不足: plane=%d len1=%d len2=%d dim=%d shape1=(%d,%d,%d) shape2=(%d,%d,%d)", plane, len(x1.Data), len(x2.Data), dim, x1.C, x1.H, x1.W, x2.C, x2.H, x2.W)
	}

	// y = cat(avg, max)，各自跨 cat(x1,x2) 的 2*dim 通道
	stats := make([]float32, 4*dim)
	for c := 0; c < 2*dim; c++ {
		var src []float32
		if c < dim {
			src = x1.Data[c*plane : (c+1)*plane]
		} else {
			src = x2.Data[(c-dim)*plane : (c-dim+1)*plane]
		}
		var sum float32
		maxV := src[0]
		for _, v := range src {
			sum += v
			if v > maxV {
				maxV = v
			}
		}
		stats[c] = sum / float32(plane)
		stats[2*dim+c] = maxV
	}
	hidden := make([]float32, 4*dim)
	if err := f.cw1.Forward(stats, 1, hidden, 1); err != nil {
		return nil, nil, err
	}
	ops.ReLUSlice(hidden)
	y := make([]float32, 2*dim)
	if err := f.cw2.Forward(hidden, 1, y, 1); err != nil {
		return nil, nil, err
	}
	ops.SigmoidSlice(y)

	// 空间权重：conv1x1(2dim→dim) → ReLU → conv1x1(dim→2) → Sigmoid
	merged := tensor.New(2*dim, x1.H, x1.W)
	copy(merged.Data[:dim*plane], x1.Data)
	copy(merged.Data[dim*plane:], x2.Data)
	hid, err := f.sw0.ForwardGrouped(merged, 1)
	if err != nil {
		return nil, nil, err
	}
	ops.ReLUSlice(hid.Data)
	sw, err := f.sw1.ForwardGrouped(hid, 1)
	if err != nil {
		return nil, nil, err
	}
	ops.SigmoidSlice(sw.Data)

	out1 := tensor.New(dim, x1.H, x1.W)
	out2 := tensor.New(dim, x1.H, x1.W)
	for c := 0; c < dim; c++ {
		// channel_weights[1] 用于 x1，channel_weights[0] 用于 x2
		cw1c := frmLambdaC * y[dim+c]
		cw2c := frmLambdaC * y[c]
		// spatial_weights 只有 2 个通道，通道 1 用于 x1、通道 0 用于 x2
		sw1c := sw.Data[plane : 2*plane]
		sw2c := sw.Data[:plane]
		o1 := out1.Data[c*plane : (c+1)*plane]
		o2 := out2.Data[c*plane : (c+1)*plane]
		a1 := x1.Data[c*plane : (c+1)*plane]
		a2 := x2.Data[c*plane : (c+1)*plane]
		for i := 0; i < plane; i++ {
			o1[i] = a1[i] + cw1c*a2[i] + frmLambdaS*sw1c[i]*a2[i]
			o2[i] = a2[i] + cw2c*a1[i] + frmLambdaS*sw2c[i]*a1[i]
		}
	}
	return out1, out2, nil
}

// ---- FFM（Feature Fusion Module） ----

type ffm struct {
	proj1, proj2  *ops.Linear
	end1, end2    *ops.Linear
	norm1, norm2  *ops.LayerNorm
	kv1, kv2      *ops.Linear
	residual      *ops.Conv2D
	ce0, ce1, ce3 *ops.Conv2D
	ceBN          *ops.BatchNorm
	outNorm       *ops.BatchNorm
	heads         int
}

func newFFM(l *loader, stage int) (*ffm, error) {
	dim := embedDims[stage]
	name := fmt.Sprintf("FFMs.%d", stage)
	f := &ffm{heads: numHeads[stage]}
	var err error
	if f.proj1, err = l.lin(name+".cross.channel_proj1", dim*2, dim); err != nil {
		return nil, err
	}
	if f.proj2, err = l.lin(name+".cross.channel_proj2", dim*2, dim); err != nil {
		return nil, err
	}
	if f.end1, err = l.lin(name+".cross.end_proj1", dim, dim*2); err != nil {
		return nil, err
	}
	if f.end2, err = l.lin(name+".cross.end_proj2", dim, dim*2); err != nil {
		return nil, err
	}
	if f.norm1, err = l.ln(name+".cross.norm1", dim, blockLayerEps); err != nil {
		return nil, err
	}
	if f.norm2, err = l.ln(name+".cross.norm2", dim, blockLayerEps); err != nil {
		return nil, err
	}
	if f.kv1, err = l.linNoBias(name+".cross.cross_attn.kv1", dim*2, dim); err != nil {
		return nil, err
	}
	if f.kv2, err = l.linNoBias(name+".cross.cross_attn.kv2", dim*2, dim); err != nil {
		return nil, err
	}
	if f.residual, err = l.conv(name+".channel_emb.residual", dim, dim*2, 1, 1, 1, 0, 1, false); err != nil {
		return nil, err
	}
	if f.ce0, err = l.conv(name+".channel_emb.channel_embed.0", dim, dim*2, 1, 1, 1, 0, 1, true); err != nil {
		return nil, err
	}
	if f.ce1, err = l.conv(name+".channel_emb.channel_embed.1", dim, dim, 3, 3, 1, 1, dim, true); err != nil {
		return nil, err
	}
	if f.ce3, err = l.conv(name+".channel_emb.channel_embed.3", dim, dim, 1, 1, 1, 0, 1, true); err != nil {
		return nil, err
	}
	if f.ceBN, err = l.bn(name+".channel_emb.channel_embed.4", dim, mlpDefaultEps); err != nil {
		return nil, err
	}
	if f.outNorm, err = l.bn(name+".channel_emb.norm", dim, mlpDefaultEps); err != nil {
		return nil, err
	}
	return f, nil
}

// forward 执行 CrossPath + ChannelEmbed，返回融合后的 NCHW 特征。
func (f *ffm) forward(x1, x2 *tensor.Tensor, workers int) (*tensor.Tensor, error) {
	dim := x1.C
	if x2.C != dim || x1.H != x2.H || x1.W != x2.W {
		return nil, fmt.Errorf("FFM 输入形状不一致")
	}
	n := x1.H * x1.W
	tok1, tok2 := x1.ToTokens(), x2.ToTokens()

	p1 := make([]float32, n*2*dim)
	if err := f.proj1.Forward(tok1, n, p1, workers); err != nil {
		return nil, err
	}
	p2 := make([]float32, n*2*dim)
	if err := f.proj2.Forward(tok2, n, p2, workers); err != nil {
		return nil, err
	}
	y1 := make([]float32, n*dim)
	u1 := make([]float32, n*dim)
	y2 := make([]float32, n*dim)
	u2 := make([]float32, n*dim)
	for i := 0; i < n; i++ {
		copy(y1[i*dim:(i+1)*dim], p1[i*2*dim:i*2*dim+dim])
		copy(u1[i*dim:(i+1)*dim], p1[i*2*dim+dim:(i+1)*2*dim])
		copy(y2[i*dim:(i+1)*dim], p2[i*2*dim:i*2*dim+dim])
		copy(u2[i*dim:(i+1)*dim], p2[i*2*dim+dim:(i+1)*2*dim])
	}
	ops.ReLUSlice(y1)
	ops.ReLUSlice(u1)
	ops.ReLUSlice(y2)
	ops.ReLUSlice(u2)

	// k/v 投影：kv(x).reshape(B,-1,2,heads,d)，前一半为 k，后一半为 v
	k1, v1 := f.splitKV(f.kv1, u1, n, dim, workers)
	k2, v2 := f.splitKV(f.kv2, u2, n, dim, workers)

	// x1' = q1 · softmax(k2ᵀv2)；x2' = q2 · softmax(k1ᵀv1)
	x1new := make([]float32, n*dim)
	x2new := make([]float32, n*dim)
	ops.CrossAttentionKV(u1, k2, v2, x1new, n, n, dim, f.heads, workers)
	ops.CrossAttentionKV(u2, k1, v1, x2new, n, n, dim, f.heads, workers)

	// y = cat(y, v') → end_proj → 残差 → LayerNorm
	in1 := make([]float32, n*2*dim)
	in2 := make([]float32, n*2*dim)
	for i := 0; i < n; i++ {
		copy(in1[i*2*dim:i*2*dim+dim], y1[i*dim:(i+1)*dim])
		copy(in1[i*2*dim+dim:(i+1)*2*dim], x1new[i*dim:(i+1)*dim])
		copy(in2[i*2*dim:i*2*dim+dim], y2[i*dim:(i+1)*dim])
		copy(in2[i*2*dim+dim:(i+1)*2*dim], x2new[i*dim:(i+1)*dim])
	}
	o1 := make([]float32, n*dim)
	o2 := make([]float32, n*dim)
	if err := f.end1.Forward(in1, n, o1, workers); err != nil {
		return nil, err
	}
	if err := f.end2.Forward(in2, n, o2, workers); err != nil {
		return nil, err
	}
	cross1 := make([]float32, n*dim)
	cross2 := make([]float32, n*dim)
	for i := range cross1 {
		cross1[i] = tok1[i] + o1[i]
		cross2[i] = tok2[i] + o2[i]
	}
	if err := f.norm1.ApplyTokens(cross1, n); err != nil {
		return nil, err
	}
	if err := f.norm2.ApplyTokens(cross2, n); err != nil {
		return nil, err
	}

	// ChannelEmbed：merge = cat(x1', x2') → 1×1 残差 + 逐通道卷积链 → BN
	merged := tensor.New(2*dim, x1.H, x1.W)
	plane := n
	for c := 0; c < dim; c++ {
		dst := merged.Data[c*plane : (c+1)*plane]
		for i := 0; i < n; i++ {
			dst[i] = cross1[i*dim+c]
		}
		dst = merged.Data[(dim+c)*plane : (dim+c+1)*plane]
		for i := 0; i < n; i++ {
			dst[i] = cross2[i*dim+c]
		}
	}
	res, err := f.residual.ForwardGrouped(merged, workers)
	if err != nil {
		return nil, err
	}
	h, err := f.ce0.ForwardGrouped(merged, workers)
	if err != nil {
		return nil, err
	}
	h, err = f.ce1.ForwardGrouped(h, workers)
	if err != nil {
		return nil, err
	}
	ops.ReLUSlice(h.Data)
	h, err = f.ce3.ForwardGrouped(h, workers)
	if err != nil {
		return nil, err
	}
	if err := f.ceBN.Apply(h); err != nil {
		return nil, err
	}
	for i := range res.Data {
		res.Data[i] += h.Data[i]
	}
	if err := f.outNorm.Apply(res); err != nil {
		return nil, err
	}
	return res, nil
}

func (f *ffm) splitKV(layer *ops.Linear, x []float32, n, dim, workers int) ([]float32, []float32) {
	kv := make([]float32, n*2*dim)
	if err := layer.Forward(x, n, kv, workers); err != nil {
		panic(err) // 形状已在构造期校验，此处不会发生
	}
	k := make([]float32, n*dim)
	v := make([]float32, n*dim)
	for i := 0; i < n; i++ {
		copy(k[i*dim:(i+1)*dim], kv[i*2*dim:i*2*dim+dim])
		copy(v[i*dim:(i+1)*dim], kv[i*2*dim+dim:(i+1)*2*dim])
	}
	return k, v
}
