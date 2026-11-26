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
	"sync"

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
		runBranch := func(blocks []*block, tok []float32, branchWorkers int) ([]float32, error) {
			var err error
			for _, blk := range blocks {
				tok, err = blk.forward(tok, h, w, branchWorkers)
				if err != nil {
					return nil, err
				}
			}
			return tok, nil
		}
		if trace == nil && b.workers > 1 && len(b.rgbBlk[i]) > 0 && len(b.npBlk[i]) > 0 {
			branchWorkers := b.workers / 2
			if branchWorkers < 1 {
				branchWorkers = 1
			}
			var wg sync.WaitGroup
			var rgbErr, npErr error
			wg.Add(2)
			go func() {
				defer wg.Done()
				rgbTok, rgbErr = runBranch(b.rgbBlk[i], rgbTok, branchWorkers)
			}()
			go func() {
				defer wg.Done()
				npTok, npErr = runBranch(b.npBlk[i], npTok, branchWorkers)
			}()
			wg.Wait()
			if rgbErr != nil {
				return nil, fmt.Errorf("stage%d RGB block: %w", i+1, rgbErr)
			}
			if npErr != nil {
				return nil, fmt.Errorf("stage%d extra block: %w", i+1, npErr)
			}
		} else {
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
		}
		if err := b.rgbNorm[i].ApplyTokensParallel(rgbTok, h*w, b.workers); err != nil {
			return nil, fmt.Errorf("stage%d norm: %w", i+1, err)
		}
		if err := b.npNorm[i].ApplyTokensParallel(npTok, h*w, b.workers); err != nil {
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
	if err := p.norm.ApplyTokensParallel(tok, out.H*out.W, workers); err != nil {
		return nil, 0, 0, err
	}
	return tok, out.H, out.W, nil
}

// ---- transformer block ----

type blockScratch struct {
	n, dim, hidden   int
	normed1, normed2 []float32
	hid, dw, mlpOut  []float32
}

// blockScratchPool 复用 transformer block 的中间缓冲。每个调用独占一个实例，
// defer Put 前所有 worker 必须已结束；sync.Pool 由 GC 自动清理。
var blockScratchPool sync.Pool

func getBlockScratch(n, dim, hidden int) *blockScratch {
	var s *blockScratch
	if v := blockScratchPool.Get(); v != nil {
		s = v.(*blockScratch)
	} else {
		s = &blockScratch{}
	}
	if s.n != n || s.dim != dim || s.hidden != hidden {
		s.n, s.dim, s.hidden = n, dim, hidden
		s.normed1 = make([]float32, n*dim)
		s.normed2 = make([]float32, n*dim)
		s.hid = make([]float32, n*hidden)
		s.dw = make([]float32, n*hidden)
		s.mlpOut = make([]float32, n*dim)
	}
	return s
}

type block struct {
	norm1 *ops.LayerNorm
	norm2 *ops.LayerNorm
	attn  *attention
	mlp   *mixFFN
}

func (b *block) forward(tok []float32, h, w, workers int) ([]float32, error) {
	n := h * w
	c := b.attn.dim()
	scratch := getBlockScratch(n, c, b.mlp.hidden)
	defer blockScratchPool.Put(scratch)

	normed := scratch.normed1
	copy(normed, tok)
	if err := b.norm1.ApplyTokensParallel(normed, n, workers); err != nil {
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

	normed2 := scratch.normed2
	copy(normed2, res)
	if err := b.norm2.ApplyTokensParallel(normed2, n, workers); err != nil {
		return nil, err
	}
	if err := b.mlp.forwardInto(normed2, c, n, h, w, workers, scratch.hid, scratch.dw, scratch.mlpOut); err != nil {
		return nil, err
	}
	for i := range res {
		res[i] += scratch.mlpOut[i]
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
		if err := a.srNorm.ApplyTokensParallel(src, s, workers); err != nil {
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
	dw := make([]float32, n*m.hidden)
	out := make([]float32, n*dim)
	if err := m.forwardInto(tok, dim, n, h, w, workers, hid, dw, out); err != nil {
		return nil, err
	}
	return out, nil
}

func (m *mixFFN) forwardInto(tok []float32, dim, n, h, w, workers int, hid, dw, out []float32) error {
	if err := m.fc1.Forward(tok, n, hid, workers); err != nil {
		return err
	}
	// 深度卷积直接在 token-major 布局上做（9 次按通道加权的位移累加），
	// 省掉"token ↔ NCHW"两次整块转置；累加顺序仍为 (ky,kx) 递增，结果与 NCHW 版本一致。
	if err := m.dw.ForwardTokensParallelInto(hid, n, h, w, workers, dw); err != nil {
		return fmt.Errorf("mlp.dwconv: %w", err)
	}
	ops.GELUInPlaceParallel(dw, workers)
	return m.fc2.Forward(dw, n, out, workers)
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
