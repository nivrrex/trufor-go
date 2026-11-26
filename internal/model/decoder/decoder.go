// Package decoder 实现 TruFor 的 MLPDecoder 解码头（定位头与置信头结构相同，类别数不同）。
//
// 对应上游 models/cmx/decoders/MLPDecoder.py：
//
//	4 级特征 → 各自 1×1 线性投影(→embed_dim) → 双线性上采到 1/4 分辨率 → 通道拼接
//	→ 1×1 卷积(4*embed_dim→embed_dim) + BatchNorm + ReLU → 1×1 卷积(→num_classes)
package decoder

import (
	"fmt"
	"sync"

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
	fuseAP   [4][]float32
	fuseTile [4]int
	fuseTail [4]int
}

// decoderWorkspace 保存单个解码头可复用的中间缓冲。ForwardPair 串行共享一个实例；
// ForwardPairParallel 为两个头各创建一个实例，返回后由调用方释放。
type decoderWorkspace struct {
	tok   []float32
	proj  [4]*tensor.Tensor
	up    *tensor.Tensor
	fused *tensor.Tensor
}

func getDecoderWorkspace(h, w, embedDim int, feats [4]*tensor.Tensor) *decoderWorkspace {
	ws := &decoderWorkspace{}
	maxTokens := 0
	for i, f := range feats {
		if n := f.H * f.W; n > maxTokens {
			maxTokens = n
		}
		ws.proj[i] = tensor.New(embedDim, f.H, f.W)
	}
	ws.up = tensor.New(embedDim, h, w)
	ws.tok = make([]float32, maxTokens*embedDim)
	ws.fused = tensor.New(embedDim, h, w)
	return ws
}

// ForwardPair 串行执行定位头与置信头，并共享中间工作区。
func ForwardPair(loc, conf *Head, feats [4]*tensor.Tensor) (*tensor.Tensor, *tensor.Tensor, error) {
	if loc == nil || conf == nil || loc.embedDim != conf.embedDim {
		return nil, nil, fmt.Errorf("decoder: 定位头/置信头配置不匹配")
	}
	h, w := feats[0].H, feats[0].W
	ws := getDecoderWorkspace(h, w, loc.embedDim, feats)
	locOut, err := loc.forward(feats, ws)
	if err != nil {
		return nil, nil, err
	}
	confOut, err := conf.forward(feats, ws)
	if err != nil {
		return nil, nil, err
	}
	return locOut, confOut, nil
}

// ForwardPairParallel 并行执行定位头与置信头，各自使用独立工作区。
func ForwardPairParallel(loc, conf *Head, feats [4]*tensor.Tensor, workers int) (*tensor.Tensor, *tensor.Tensor, error) {
	if loc == nil || conf == nil || loc.embedDim != conf.embedDim {
		return nil, nil, fmt.Errorf("decoder: 定位头/置信头配置不匹配")
	}
	h, w := feats[0].H, feats[0].W
	locWS := getDecoderWorkspace(h, w, loc.embedDim, feats)
	confWS := getDecoderWorkspace(h, w, conf.embedDim, feats)
	branchWorkers := workers / 2
	if branchWorkers < 1 {
		branchWorkers = 1
	}
	var wg sync.WaitGroup
	var locOut, confOut *tensor.Tensor
	var locErr, confErr error
	wg.Add(2)
	go func() {
		defer wg.Done()
		locOut, locErr = loc.forwardWithWorkers(feats, locWS, branchWorkers)
	}()
	go func() {
		defer wg.Done()
		confOut, confErr = conf.forwardWithWorkers(feats, confWS, branchWorkers)
	}()
	wg.Wait()
	if locErr != nil {
		return nil, nil, locErr
	}
	if confErr != nil {
		return nil, nil, confErr
	}
	return locOut, confOut, nil
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
	for block := 0; block < 4; block++ {
		ap, tiles, tail, err := h.fuse.PackWeightRange(block*embedDim, embedDim)
		if err != nil {
			return nil, err
		}
		h.fuseAP[block], h.fuseTile[block], h.fuseTail[block] = ap, tiles, tail
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
	h0, w0 := feats[0].H, feats[0].W
	ws := getDecoderWorkspace(h0, w0, h.embedDim, feats)
	return h.forward(feats, ws)
}

func (h *Head) forward(feats [4]*tensor.Tensor, ws *decoderWorkspace) (*tensor.Tensor, error) {
	return h.forwardWithWorkers(feats, ws, h.workers)
}

func (h *Head) forwardWithWorkers(feats [4]*tensor.Tensor, ws *decoderWorkspace, workers int) (*tensor.Tensor, error) {
	h0, w0 := feats[0].H, feats[0].W
	for i, f := range feats {
		if f == nil {
			return nil, fmt.Errorf("decoder: 第 %d 级特征为 nil", i)
		}
		if f.C != h.lin[i].In {
			return nil, fmt.Errorf("decoder: 第 %d 级通道 %d != %d", i, f.C, h.lin[i].In)
		}
	}
	h.fuse.InitOutput(ws.fused, workers)
	// 上游拼接顺序是 [_c4,_c3,_c2,_c1]；逆序处理以保持融合卷积 K 方向的
	// 累加顺序，同时只复用一块上采缓冲并避免 4×embedDim 的 merged 张量。
	for i := 3; i >= 0; i-- {
		f := feats[i]
		n := f.H * f.W
		buf := ws.tok[:n*h.embedDim]
		if err := h.lin[i].Forward(f.ToTokens(), n, buf, workers); err != nil {
			return nil, err
		}
		tensor.FromTokensInto(buf, h.embedDim, f.H, f.W, ws.proj[i])
		var projected *tensor.Tensor
		if i == 0 {
			projected = ws.proj[i]
		} else {
			if err := ops.ResizeBilinearIntoParallel(ws.proj[i], h0, w0, ws.up, workers); err != nil {
				return nil, err
			}
			projected = ws.up
		}
		block := 3 - i
		if err := h.fuse.Accumulate1x1IntoPacked(ws.fused, projected, block*h.embedDim, h.fuseAP[block], h.fuseTile[block], h.fuseTail[block], workers); err != nil {
			return nil, err
		}
	}

	if err := h.bn.Apply(ws.fused); err != nil {
		return nil, err
	}
	ops.ReLUInPlace(ws.fused)
	return h.pred.Forward(ws.fused, workers)
}
