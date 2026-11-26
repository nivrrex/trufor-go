// Package trufor 组装完整推理链：
//
//	RGB[0,1] → DnCNN(Noiseprint++) → 双支 SegFormer + FRM/FFM（ImageNet 归一化的 RGB + NP++）
//	→ 定位头 logits → softmax 得 map；置信头 logits → sigmoid 得 conf；detection 头 → score
//
// 结构与上游 test_docker/src/trufor_test.py 的推理路径一致，单张图像、无 batch 维度。
package trufor

import (
	"fmt"
	"math"

	"github.com/nivrrex/trufor-go/internal/model/decoder"
	"github.com/nivrrex/trufor-go/internal/model/dncnn"
	"github.com/nivrrex/trufor-go/internal/model/segformer"
	"github.com/nivrrex/trufor-go/internal/ops"
	"github.com/nivrrex/trufor-go/internal/tensor"
	"github.com/nivrrex/trufor-go/internal/weights"
)

// 解码头相关常量（与 checkpoint 一致）。
var decoderInChannels = [4]int{64, 128, 320, 512}

const (
	embedDim      = 512
	numClasses    = 2
	locHeadBNEps  = 1e-3 // 定位头 BN 被 init_weight 改写为 config.BN_EPS
	confHeadBNEps = 1e-5 // 置信头 BN 保持 BatchNorm2d 默认值
	imagenetMeanR = 0.485
	imagenetMeanG = 0.456
	imagenetMeanB = 0.406
	imagenetStdR  = 0.229
	imagenetStdG  = 0.224
	imagenetStdB  = 0.225
)

// Result 是一次前向的输出。
type Result struct {
	// Logits 为定位头在原始分辨率上的 logits（2×H×W）。
	Logits *tensor.Tensor
	// Map 为 softmax(logits) 的伪造类概率（1×H×W）。
	Map *tensor.Tensor
	// Conf 为置信度的 sigmoid 值（1×H×W）。
	Conf *tensor.Tensor
	// Score 为整图完整性分数（sigmoid 后的 det），范围 [0,1]。
	Score float32
	// Noiseprint 为 Noiseprint++（1×H×W）。
	Noiseprint *tensor.Tensor
}

// Model 是完整推理模型，构造后只读，可并发调用 Forward（内部按行/通道切分的临时缓冲互不共享）。
type Model struct {
	dncnn    *dncnn.Model
	backbone *segformer.Backbone
	locHead  *decoder.Head
	confHead *decoder.Head
	det0     *ops.Linear // 8→128
	det3     *ops.Linear // 128→1
	workers  int
}

// New 从权重构造完整模型。workers 为算子并行度。
func New(w *weights.File, workers int) (*Model, error) {
	if workers < 1 {
		workers = 1
	}
	d, err := dncnn.New(w, workers)
	if err != nil {
		return nil, fmt.Errorf("构造 DnCNN: %w", err)
	}
	bb, err := segformer.New(w, "backbone", workers)
	if err != nil {
		return nil, fmt.Errorf("构造 backbone: %w", err)
	}
	loc, err := decoder.New(w, "decode_head", decoderInChannels, numClasses, embedDim, locHeadBNEps, workers)
	if err != nil {
		return nil, fmt.Errorf("构造定位头: %w", err)
	}
	conf, err := decoder.New(w, "decode_head_conf", decoderInChannels, 1, embedDim, confHeadBNEps, workers)
	if err != nil {
		return nil, fmt.Errorf("构造置信头: %w", err)
	}
	d0w, err := w.Get("detection.0.weight")
	if err != nil {
		return nil, err
	}
	d0b, err := w.Get("detection.0.bias")
	if err != nil {
		return nil, err
	}
	if len(d0w.Shape) != 2 || d0w.Shape[0] != 128 || d0w.Shape[1] != 8 {
		return nil, fmt.Errorf("detection.0.weight 形状 %v != (128,8)", d0w.Shape)
	}
	d0, err := ops.NewLinear(d0w.Data, 128, 8, d0b.Data)
	if err != nil {
		return nil, err
	}
	d3w, err := w.Get("detection.3.weight")
	if err != nil {
		return nil, err
	}
	d3b, err := w.Get("detection.3.bias")
	if err != nil {
		return nil, err
	}
	if len(d3w.Shape) != 2 || d3w.Shape[0] != 1 || d3w.Shape[1] != 128 {
		return nil, fmt.Errorf("detection.3.weight 形状 %v != (1,128)", d3w.Shape)
	}
	d3, err := ops.NewLinear(d3w.Data, 1, 128, d3b.Data)
	if err != nil {
		return nil, err
	}
	return &Model{dncnn: d, backbone: bb, locHead: loc, confHead: conf, det0: d0, det3: d3, workers: workers}, nil
}

// TraceFunc 在指定检查点被调用，name 为 "loc4"、"conf4"、"logits"、"map"、"conf"。
type TraceFunc func(name string, t *tensor.Tensor) error

// Forward 对单张 [0,1] 范围的 RGB 张量执行完整推理。
func (m *Model) Forward(rgb01 *tensor.Tensor) (*Result, error) {
	return m.forward(rgb01, nil)
}

// ForwardTrace 与 Forward 相同，但逐检查点回调（对拍/调试用）。
func (m *Model) ForwardTrace(rgb01 *tensor.Tensor, trace TraceFunc) (*Result, error) {
	if trace == nil {
		return nil, fmt.Errorf("trufor.ForwardTrace: trace 不能为 nil")
	}
	return m.forward(rgb01, trace)
}

func (m *Model) forward(rgb01 *tensor.Tensor, trace TraceFunc) (*Result, error) {
	if rgb01 == nil {
		return nil, fmt.Errorf("trufor: 输入张量为 nil")
	}
	if rgb01.C != 3 {
		return nil, fmt.Errorf("trufor: 输入必须为 3 通道 RGB（实际 %d）", rgb01.C)
	}

	// Noiseprint++（上游在 ImageNet 归一化之前计算）
	np, err := m.dncnn.Forward(rgb01)
	if err != nil {
		return nil, fmt.Errorf("Noiseprint++: %w", err)
	}
	np3 := tensor.New(3, np.H, np.W)
	plane := np.H * np.W
	for c := 0; c < 3; c++ {
		copy(np3.Data[c*plane:(c+1)*plane], np.Data)
	}

	// ImageNet 归一化后的 RGB 支路
	rgb := rgb01.Clone()
	means := [3]float32{imagenetMeanR, imagenetMeanG, imagenetMeanB}
	stds := [3]float32{imagenetStdR, imagenetStdG, imagenetStdB}
	for c := 0; c < 3; c++ {
		d := rgb.Data[c*plane : (c+1)*plane]
		for i := range d {
			d[i] = (d[i] - means[c]) / stds[c]
		}
	}

	feats, err := m.backbone.Forward(rgb, np3)
	if err != nil {
		return nil, fmt.Errorf("backbone: %w", err)
	}
	var stack [4]*tensor.Tensor
	copy(stack[:], feats)

	var locLogits4, confLogits4 *tensor.Tensor
	if m.workers > 1 {
		locLogits4, confLogits4, err = decoder.ForwardPairParallel(m.locHead, m.confHead, stack, m.workers)
	} else {
		locLogits4, confLogits4, err = decoder.ForwardPair(m.locHead, m.confHead, stack)
	}
	if err != nil {
		return nil, fmt.Errorf("decoder: %w", err)
	}

	if trace != nil {
		if err := trace("loc4", locLogits4); err != nil {
			return nil, err
		}
		if err := trace("conf4", confLogits4); err != nil {
			return nil, err
		}
	}

	// 上采到原始分辨率
	logits, err := ops.ResizeBilinearParallel(locLogits4, rgb01.H, rgb01.W, m.workers)
	if err != nil {
		return nil, err
	}
	confLogits, err := ops.ResizeBilinearParallel(confLogits4, rgb01.H, rgb01.W, m.workers)
	if err != nil {
		return nil, err
	}

	// map = softmax(logits)[1]；conf = sigmoid(confLogits)
	npix := rgb01.H * rgb01.W
	mapData := make([]float32, npix)
	for i := 0; i < npix; i++ {
		o0 := float64(logits.Data[i])
		o1 := float64(logits.Data[npix+i])
		mx := o0
		if o1 > mx {
			mx = o1
		}
		e0 := math.Exp(o0 - mx)
		e1 := math.Exp(o1 - mx)
		mapData[i] = float32(e1 / (e0 + e1))
	}
	mapT := tensor.NewFromData(mapData, 1, rgb01.H, rgb01.W)

	confData := make([]float32, npix)
	copy(confData, confLogits.Data)
	ops.SigmoidSlice(confData)
	confT := tensor.NewFromData(confData, 1, rgb01.H, rgb01.W)

	if trace != nil {
		if err := trace("logits", logits); err != nil {
			return nil, err
		}
		if err := trace("map", mapT); err != nil {
			return nil, err
		}
		if err := trace("conf", confT); err != nil {
			return nil, err
		}
	}

	score, err := m.detect(logits, confLogits, npix)
	if err != nil {
		return nil, fmt.Errorf("detection: %w", err)
	}

	return &Result{Logits: logits, Map: mapT, Conf: confT, Score: score, Noiseprint: np}, nil
}

// detect 复刻上游 confpool 检测头：
//
//	f1 = wsp(conf)；f2 = wsp(logit1-logit0, log_w=logsigmoid(conf))
//	det = Linear(8→128) → ReLU → Linear(128→1) → sigmoid
func (m *Model) detect(logits, confLogits *tensor.Tensor, npix int) (float32, error) {
	diff := make([]float32, npix)
	for i := 0; i < npix; i++ {
		diff[i] = logits.Data[npix+i] - logits.Data[i]
	}
	logW := make([]float32, npix)
	ops.LogSigmoid(confLogits.Data[:npix], logW)

	f1 := ops.WeightedStatisticsPooling(confLogits.Data[:npix], nil)
	f2 := ops.WeightedStatisticsPooling(diff, logW)
	feat := make([]float32, 8)
	copy(feat[:4], f1[:])
	copy(feat[4:], f2[:])

	hid := make([]float32, 128)
	if err := m.det0.Forward(feat, 1, hid, 1); err != nil {
		return 0, err
	}
	ops.ReLUSlice(hid)
	out := make([]float32, 1)
	if err := m.det3.Forward(hid, 1, out, 1); err != nil {
		return 0, err
	}
	return float32(1 / (1 + math.Exp(-float64(out[0])))), nil
}
