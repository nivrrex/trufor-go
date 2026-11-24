// Package segformer 实现 TruFor 主干：双支 SegFormer-B2（RGB 支路 + Noiseprint++ 支路）
// 以及 CMX 的跨模态融合模块 FRM / FFM。
//
// 结构与上游 test_docker/src/models/cmx/encoders/dual_segformer.py + net_utils.py 一一对应：
//
//	stage i: patch_embed → (Block×depth, RGB 与 extra 各自独立) → LayerNorm
//	         → FRM(rgb, extra) → FFM(rgb, extra) → 该级融合特征
//
// 注意 eps 差异：patch_embed 内的 LayerNorm 用默认 1e-5，Block/CrossPath 内用 1e-6，
// FFM 内 BatchNorm 用默认 1e-5（均已由参考实现逐模块核对）。
package segformer

import (
	"fmt"

	"github.com/nivrrex/trufor-go/internal/kernels"
	"github.com/nivrrex/trufor-go/internal/ops"
	"github.com/nivrrex/trufor-go/internal/tensor"
	"github.com/nivrrex/trufor-go/internal/weights"
)

// 后四个模态无关的超参常量（mit_b2）。
var (
	embedDims = [4]int{64, 128, 320, 512}
	numHeads  = [4]int{1, 2, 5, 8}
	depths    = [4]int{3, 4, 6, 3}
	srRatios  = [4]int{8, 4, 2, 1}
)

const (
	mlpRatio       = 4
	mlpDefaultEps  = 1e-5 // nn.LayerNorm / nn.BatchNorm2d 默认
	blockLayerEps  = 1e-6 // partial(nn.LayerNorm, eps=1e-6)
	patchKernel0   = 7
	patchStride0   = 4
	patchStrideN   = 2
	patchKernelN   = 3
	frmLambdaC     = 0.5
	frmLambdaS     = 0.5
	imagenetMeanR  = 0.485
	imagenetMeanG  = 0.456
	imagenetMeanB  = 0.406
	imagenetStdR   = 0.229
	imagenetStdG   = 0.224
	imagenetStdB   = 0.225
	embedDimDecode = 512
)

// TraceFunc 在指定检查点被调用，name 与 tools/dump_golden.py 的命名一致。
type TraceFunc func(name string, t *tensor.Tensor) error

// Backbone 是双支 SegFormer 主干，构造后只读，可并发调用 Forward。
type Backbone struct {
	workers int
	rgbPE   [4]*patchEmbed
	npPE    [4]*patchEmbed
	rgbBlk  [4][]*block
	npBlk   [4][]*block
	rgbNorm [4]*ops.LayerNorm
	npNorm  [4]*ops.LayerNorm
	frms    [4]*frm
	ffms    [4]*ffm
}

// New 从权重构造主干。prefix 为权重名前缀（官方 checkpoint 为 "backbone"）。
func New(w *weights.File, prefix string, workers int) (*Backbone, error) {
	if workers < 1 {
		workers = 1
	}
	b := &Backbone{workers: workers}
	ld := &loader{w: w, prefix: prefix}
	lin, conv, ln := ld.lin, ld.conv, ld.ln

	// patch_embed：stage0 为 7×7/s4，之后为 3×3/s2
	newPatchEmbed := func(prefixName string, stage int) (*patchEmbed, error) {
		inC, k, s := 3, patchKernel0, patchStride0
		if stage > 0 {
			inC, k, s = embedDims[stage-1], patchKernelN, patchStrideN
		}
		c, err := conv(prefixName+".proj", embedDims[stage], inC, k, k, s, k/2, 1, true)
		if err != nil {
			return nil, err
		}
		n, err := ln(prefixName+".norm", embedDims[stage], mlpDefaultEps)
		if err != nil {
			return nil, err
		}
		return &patchEmbed{conv: c, norm: n}, nil
	}

	newBlock := func(prefixName string, stage int) (*block, error) {
		dim := embedDims[stage]
		n1, err := ln(prefixName+".norm1", dim, blockLayerEps)
		if err != nil {
			return nil, err
		}
		n2, err := ln(prefixName+".norm2", dim, blockLayerEps)
		if err != nil {
			return nil, err
		}
		q, err := lin(prefixName+".attn.q", dim, dim)
		if err != nil {
			return nil, err
		}
		kv, err := lin(prefixName+".attn.kv", dim*2, dim)
		if err != nil {
			return nil, err
		}
		proj, err := lin(prefixName+".attn.proj", dim, dim)
		if err != nil {
			return nil, err
		}
		att := &attention{q: q, kv: kv, proj: proj, heads: numHeads[stage], srRatio: srRatios[stage]}
		if srRatios[stage] > 1 {
			sr, err := conv(prefixName+".attn.sr", dim, dim, srRatios[stage], srRatios[stage], srRatios[stage], 0, 1, true)
			if err != nil {
				return nil, err
			}
			srNorm, err := ln(prefixName+".attn.norm", dim, blockLayerEps)
			if err != nil {
				return nil, err
			}
			att.sr, att.srNorm = sr, srNorm
		}
		hidden := dim * mlpRatio
		fc1, err := lin(prefixName+".mlp.fc1", hidden, dim)
		if err != nil {
			return nil, err
		}
		fc2, err := lin(prefixName+".mlp.fc2", dim, hidden)
		if err != nil {
			return nil, err
		}
		dw, err := conv(prefixName+".mlp.dwconv.dwconv", hidden, hidden, 3, 3, 1, 1, hidden, true)
		if err != nil {
			return nil, err
		}
		return &block{
			norm1: n1, norm2: n2,
			attn: att,
			mlp:  &mixFFN{fc1: fc1, fc2: fc2, dw: dw, hidden: hidden},
		}, nil
	}

	for i := 0; i < 4; i++ {
		pe, err := newPatchEmbed(fmt.Sprintf("patch_embed%d", i+1), i)
		if err != nil {
			return nil, err
		}
		b.rgbPE[i] = pe
		xpe, err := newPatchEmbed(fmt.Sprintf("extra_patch_embed%d", i+1), i)
		if err != nil {
			return nil, err
		}
		b.npPE[i] = xpe
		for j := 0; j < depths[i]; j++ {
			blk, err := newBlock(fmt.Sprintf("block%d.%d", i+1, j), i)
			if err != nil {
				return nil, err
			}
			b.rgbBlk[i] = append(b.rgbBlk[i], blk)
			xblk, err := newBlock(fmt.Sprintf("extra_block%d.%d", i+1, j), i)
			if err != nil {
				return nil, err
			}
			b.npBlk[i] = append(b.npBlk[i], xblk)
		}
		n, err := ln(fmt.Sprintf("norm%d", i+1), embedDims[i], blockLayerEps)
		if err != nil {
			return nil, err
		}
		b.rgbNorm[i] = n
		xn, err := ln(fmt.Sprintf("extra_norm%d", i+1), embedDims[i], blockLayerEps)
		if err != nil {
			return nil, err
		}
		b.npNorm[i] = xn

		f, err := newFRM(ld, i)
		if err != nil {
			return nil, err
		}
		b.frms[i] = f
		ff, err := newFFM(ld, i)
		if err != nil {
			return nil, err
		}
		b.ffms[i] = ff
	}
	return b, nil
}

// Forward 返回 4 级融合特征（1/4、1/8、1/16、1/32）。
// rgb 必须是已做 ImageNet 归一化的张量；np 为 Noiseprint++ 三通道张量。
func (b *Backbone) Forward(rgb, np *tensor.Tensor) ([]*tensor.Tensor, error) {
	return b.forward(rgb, np, nil)
}

// ForwardTrace 与 Forward 相同，但逐检查点回调（对拍/调试用）。
func (b *Backbone) ForwardTrace(rgb, np *tensor.Tensor, trace TraceFunc) ([]*tensor.Tensor, error) {
	if trace == nil {
		return nil, fmt.Errorf("segformer.ForwardTrace: trace 不能为 nil")
	}
	return b.forward(rgb, np, trace)
}

func (b *Backbone) forward(rgb, np *tensor.Tensor, trace TraceFunc) ([]*tensor.Tensor, error) {
	if rgb == nil || np == nil {
		return nil, fmt.Errorf("segformer: 输入不能为 nil")
	}
	if rgb.C != 3 || np.C != 3 {
		return nil, fmt.Errorf("segformer: 两个输入都必须为 3 通道（rgb=%d, np=%d）", rgb.C, np.C)
	}
	xRGB, xNP := rgb, np
	outs := make([]*tensor.Tensor, 0, 4)

	for i := 0; i < 4; i++ {
		rgbTok, h, w, err := b.rgbPE[i].forward(xRGB, b.workers)
		if err != nil {
			return nil, fmt.Errorf("stage%d patch_embed: %w", i+1, err)
		}
		npTok, nh, nw, err := b.npPE[i].forward(xNP, b.workers)
		if err != nil {
			return nil, fmt.Errorf("stage%d extra_patch_embed: %w", i+1, err)
		}
		if nh != h || nw != w {
			return nil, fmt.Errorf("stage%d 两支特征尺寸不一致: rgb=%dx%d np=%dx%d", i+1, h, w, nh, nw)
		}
		if trace != nil {
			if err := b.traceTokens(trace, fmt.Sprintf("stage%d.rgb_patch_embed", i+1), rgbTok, embedDims[i], h, w); err != nil {
				return nil, err
			}
			if err := b.traceTokens(trace, fmt.Sprintf("stage%d.np_patch_embed", i+1), npTok, embedDims[i], h, w); err != nil {
				return nil, err
			}
		}
		for j, blk := range b.rgbBlk[i] {
			rgbTok, err = blk.forward(rgbTok, h, w, b.workers)
			if err != nil {
				return nil, fmt.Errorf("stage%d block%d: %w", i+1, j, err)
			}
			if trace != nil {
				if err := b.traceTokens(trace, fmt.Sprintf("stage%d.block%d", i+1, j), rgbTok, embedDims[i], h, w); err != nil {
					return nil, err
				}
			}
		}
		for j, blk := range b.npBlk[i] {
			npTok, err = blk.forward(npTok, h, w, b.workers)
			if err != nil {
				return nil, fmt.Errorf("stage%d extra_block%d: %w", i+1, j, err)
			}
			if trace != nil {
				if err := b.traceTokens(trace, fmt.Sprintf("stage%d.extra_block%d", i+1, j), npTok, embedDims[i], h, w); err != nil {
					return nil, err
				}
			}
		}
		if err := b.rgbNorm[i].ApplyTokens(rgbTok, h*w); err != nil {
			return nil, fmt.Errorf("stage%d norm: %w", i+1, err)
		}
		if err := b.npNorm[i].ApplyTokens(npTok, h*w); err != nil {
			return nil, fmt.Errorf("stage%d extra_norm: %w", i+1, err)
		}
		if trace != nil {
			if err := b.traceTokens(trace, fmt.Sprintf("stage%d.rgb_norm", i+1), rgbTok, embedDims[i], h, w); err != nil {
				return nil, err
			}
			if err := b.traceTokens(trace, fmt.Sprintf("stage%d.np_norm", i+1), npTok, embedDims[i], h, w); err != nil {
				return nil, err
			}
		}

		xRGB = tensor.FromTokens(rgbTok, embedDims[i], h, w)
		xNP = tensor.FromTokens(npTok, embedDims[i], h, w)

		rectRGB, rectNP, err := b.frms[i].forward(xRGB, xNP)
		if err != nil {
			return nil, fmt.Errorf("stage%d FRM: %w", i+1, err)
		}
		if trace != nil {
			if err := b.traceTensor(trace, fmt.Sprintf("stage%d.frm_rgb", i+1), rectRGB); err != nil {
				return nil, err
			}
			if err := b.traceTensor(trace, fmt.Sprintf("stage%d.frm_np", i+1), rectNP); err != nil {
				return nil, err
			}
		}
		fused, err := b.ffms[i].forward(rectRGB, rectNP, b.workers)
		if err != nil {
			return nil, fmt.Errorf("stage%d FFM: %w", i+1, err)
		}
		if trace != nil {
			if err := b.traceTensor(trace, fmt.Sprintf("stage%d.ffm", i+1), fused); err != nil {
				return nil, err
			}
		}
		outs = append(outs, fused)
		// 下一级的输入是 FRM 修正后的特征（不是 FFM 融合输出），与上游 forward_features 一致
		xRGB, xNP = rectRGB, rectNP
	}
	return outs, nil
}

func (b *Backbone) traceTensor(trace TraceFunc, name string, t *tensor.Tensor) error {
	return trace(name, t)
}

// traceTokens 先把 token-major 数据转成 NCHW 再回调（trace 契约是 NCHW）。
func (b *Backbone) traceTokens(trace TraceFunc, name string, tok []float32, c, h, w int) error {
	return trace(name, tensor.FromTokens(tok, c, h, w))
}

// ---- patch embed ----

type patchEmbed struct {
	conv *ops.Conv2D
	norm *ops.LayerNorm
}

func (p *patchEmbed) forward(x *tensor.Tensor, workers int) ([]float32, int, int, error) {
	out, err := p.conv.ForwardGrouped(x, workers)
	if err != nil {
		return nil, 0, 0, err
	}
	tok := out.ToTokens()
	if err := p.norm.ApplyTokens(tok, out.H*out.W); err != nil {
		return nil, 0, 0, err
	}
	return tok, out.H, out.W, nil
}

// ---- transformer block ----

type block struct {
	norm1 *ops.LayerNorm
	norm2 *ops.LayerNorm
	attn  *attention
	mlp   *mixFFN
}

func (b *block) forward(tok []float32, h, w, workers int) ([]float32, error) {
	n := h * w
	c := b.attn.dim()

	normed := append([]float32(nil), tok...)
	if err := b.norm1.ApplyTokens(normed, n); err != nil {
		return nil, err
	}
	att, err := b.attn.forward(normed, c, n, h, w, workers)
	if err != nil {
		return nil, err
	}
	res := make([]float32, len(tok))
	for i := range tok {
		res[i] = tok[i] + att[i]
	}

	normed2 := append([]float32(nil), res...)
	if err := b.norm2.ApplyTokens(normed2, n); err != nil {
		return nil, err
	}
	mlpOut, err := b.mlp.forward(normed2, c, n, h, w, workers)
	if err != nil {
		return nil, err
	}
	for i := range res {
		res[i] += mlpOut[i]
	}
	return res, nil
}

// ---- attention（含 SR 稀疏化） ----

type attention struct {
	q, kv, proj *ops.Linear
	sr          *ops.Conv2D
	srNorm      *ops.LayerNorm
	heads       int
	srRatio     int
}

func (a *attention) dim() int { return a.q.In }

func (a *attention) forward(tok []float32, dim, n, h, w, workers int) ([]float32, error) {
	q := make([]float32, n*dim)
	if err := a.q.Forward(tok, n, q, workers); err != nil {
		return nil, err
	}

	var src []float32
	var s int
	if a.sr != nil {
		img := tensor.FromTokens(tok, dim, h, w)
		reduced, err := a.sr.ForwardGrouped(img, workers)
		if err != nil {
			return nil, fmt.Errorf("attn.sr: %w", err)
		}
		src = reduced.ToTokens()
		s = reduced.H * reduced.W
		if err := a.srNorm.ApplyTokens(src, s); err != nil {
			return nil, err
		}
	} else {
		src = tok
		s = n
	}

	kvAll := make([]float32, s*2*dim)
	if err := a.kv.Forward(src, s, kvAll, workers); err != nil {
		return nil, err
	}
	k := make([]float32, s*dim)
	v := make([]float32, s*dim)
	for i := 0; i < s; i++ {
		copy(k[i*dim:(i+1)*dim], kvAll[i*2*dim:i*2*dim+dim])
		copy(v[i*dim:(i+1)*dim], kvAll[i*2*dim+dim:(i+1)*2*dim])
	}
	attnOut := make([]float32, n*dim)
	ops.MultiHeadSelfAttention(q, k, v, attnOut, n, s, dim, a.heads, workers)

	out := make([]float32, n*dim)
	if err := a.proj.Forward(attnOut, n, out, workers); err != nil {
		return nil, err
	}
	return out, nil
}

// ---- MixFFN ----

type mixFFN struct {
	fc1, fc2 *ops.Linear
	dw       *ops.Conv2D
	hidden   int
}

func (m *mixFFN) forward(tok []float32, dim, n, h, w, workers int) ([]float32, error) {
	hid := make([]float32, n*m.hidden)
	if err := m.fc1.Forward(tok, n, hid, workers); err != nil {
		return nil, err
	}
	// 深度卷积直接在 token-major 布局上做（9 次按通道加权的位移累加），
	// 省掉"token ↔ NCHW"两次整块转置；累加顺序仍为 (ky,kx) 递增，结果与 NCHW 版本一致。
	dw, err := m.dw.ForwardTokens(hid, n, h, w)
	if err != nil {
		return nil, fmt.Errorf("mlp.dwconv: %w", err)
	}
	kernels.GELUInPlace(dw)
	out := make([]float32, n*dim)
	if err := m.fc2.Forward(dw, n, out, workers); err != nil {
		return nil, err
	}
	return out, nil
}

func sameShape(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
