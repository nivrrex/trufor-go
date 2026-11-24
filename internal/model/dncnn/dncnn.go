// Package dncnn 实现 TruFor 的 Noiseprint++ 提取器（DnCNN，17 层 3×3 卷积）。
//
// 结构与上游 models/DnCNN.py 的 make_net 调用严格对应：
//
//	conv(3→64, bias, pad=1) → ReLU
//	15 × [ conv(64→64, no bias, pad=1) → BatchNorm2d(64, eps=1e-5) → ReLU ]
//	conv(64→1, bias, pad=1)  （无激活）
//
// 注意：该网络没有残差连接（上游直接用 make_net 构造 Sequential，未走 DnCNN 类）。
package dncnn

import (
	"fmt"
	"sync"

	"github.com/nivrrex/trufor-go/internal/ops"
	"github.com/nivrrex/trufor-go/internal/tensor"
	"github.com/nivrrex/trufor-go/internal/weights"
)

// bnEps 为上游 nn.BatchNorm2d 的默认 eps。DnCNN 的 BN 由 make_net 直接构造，
// 未经 init_weight 改写，因此保持默认 1e-5（已由参考实现逐模块核对）。
const bnEps = 1e-5

// 上游 Sequential 中各子模块的下标布局：conv0, relu1, 然后每 3 个一组 (conv, bn, relu)。
const (
	stemConvIdx  = 0
	hiddenBlocks = 15
	headConvIdx  = 47
)

// ones64 / zeros64 用于给无 BN 的层提供"恒等仿射 + ReLU"，保持与融合路径同一代码结构。
var (
	ones64 = []float32{1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
		1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1}
	zeros64 = make([]float32, 64)
)

type step struct {
	idx  int
	conv *ops.Conv2D
	// bn 非 nil 时表示该层后接 BatchNorm(+ReLU)；正常路径把 BN/ReLU 融合进卷积输出，
	// 只有 trace（对拍/调试）时才按原顺序单独执行，以便导出与参考实现同名的中间结果。
	bn  *ops.BatchNorm
	epi *ops.Affine
}

// 空间分块参数：
//
// 17 层 3×3 卷积的等效感受野半径为 17，因此分块时上下各取 halo=17 行真实数据，
// 块内结果与整图计算逐位相同（每个输出像素的取值与累加顺序都不变）。
// 大图分块的目的不是提速，而是把 64 通道的全分辨率缓冲压到可控范围
// （12 MP 时单块约 800 MB → 分块后约 300 MB）。
const (
	haloRows       = 17
	tilePixelLimit = 2 << 20 // 超过该像素数才分块（约 2 MP）
)

// Model 是 DnCNN 推理模型，构造后只读，可并发调用 Forward。
//
// 内部用 sync.Pool 复用两块隐藏层缓冲（ping-pong）与一块输出缓冲，
// 避免每层分配 64 通道的大张量（12 MP 时单块约 800 MB）。
type Model struct {
	steps      []step
	workers    int
	inChannels int
	pool       sync.Pool
}

// bufs 是一次前向用到的缓冲集合。
type bufs struct {
	a, b    *tensor.Tensor // 隐藏层 ping-pong（64 通道）
	out     *tensor.Tensor // 最终输出（1 通道）
	c, h, w int
}

// New 从已加载的权重构造 DnCNN。workers 为卷积并行度。
func New(w *weights.File, workers int) (*Model, error) {
	if workers < 1 {
		workers = 1
	}
	m := &Model{workers: workers, inChannels: 3}
	m.pool.New = func() any { return &bufs{} }

	conv := func(key string, outC, inC int, withBias bool) (*ops.Conv2D, error) {
		wht, err := w.Get("dncnn." + key + ".weight")
		if err != nil {
			return nil, err
		}
		if len(wht.Shape) != 4 || wht.Shape[0] != outC || wht.Shape[1] != inC || wht.Shape[2] != 3 || wht.Shape[3] != 3 {
			return nil, fmt.Errorf("dncnn.%s.weight 形状 %v 与预期 (%d,%d,3,3) 不符", key, wht.Shape, outC, inC)
		}
		var bias []float32
		if withBias {
			bt, err := w.Get("dncnn." + key + ".bias")
			if err != nil {
				return nil, err
			}
			if len(bt.Shape) != 1 || bt.Shape[0] != outC {
				return nil, fmt.Errorf("dncnn.%s.bias 形状 %v 与预期 (%d) 不符", key, bt.Shape, outC)
			}
			bias = bt.Data
		}
		return ops.NewConv2D(wht.Data, outC, inC, 3, 3, 1, 1, 1, bias)
	}

	bn := func(key string, ch int) (*ops.BatchNorm, error) {
		get := func(suffix string) ([]float32, error) {
			t, err := w.Get("dncnn." + key + "." + suffix)
			if err != nil {
				return nil, err
			}
			if len(t.Shape) != 1 || t.Shape[0] != ch {
				return nil, fmt.Errorf("dncnn.%s.%s 形状 %v 与预期 (%d) 不符", key, suffix, t.Shape, ch)
			}
			return t.Data, nil
		}
		mean, err := get("running_mean")
		if err != nil {
			return nil, err
		}
		variance, err := get("running_var")
		if err != nil {
			return nil, err
		}
		gamma, err := get("weight")
		if err != nil {
			return nil, err
		}
		beta, err := get("bias")
		if err != nil {
			return nil, err
		}
		return ops.NewBatchNorm(mean, variance, gamma, beta, bnEps)
	}

	stem, err := conv("0", 64, 3, true)
	if err != nil {
		return nil, err
	}
	m.steps = append(m.steps, step{idx: stemConvIdx, conv: stem, epi: &ops.Affine{
		Scale: ones64, Shift: zeros64, Relu: true,
	}})

	for i := 1; i <= hiddenBlocks; i++ {
		convIdx := 3*i - 1
		bnIdx := 3 * i
		c, err := conv(fmt.Sprintf("%d", convIdx), 64, 64, false)
		if err != nil {
			return nil, err
		}
		b, err := bn(fmt.Sprintf("%d", bnIdx), 64)
		if err != nil {
			return nil, err
		}
		// 把紧跟其后的 BN 与 ReLU 融合进卷积输出（算术顺序不变，少两次整块遍历）
		m.steps = append(m.steps, step{idx: convIdx, conv: c, bn: b, epi: b.Affine(true)})
	}

	head, err := conv(fmt.Sprintf("%d", headConvIdx), 1, 64, true)
	if err != nil {
		return nil, err
	}
	m.steps = append(m.steps, step{idx: headConvIdx, conv: head})
	return m, nil
}

// Workers 返回并行度。
func (m *Model) Workers() int { return m.workers }

// Forward 计算 Noiseprint++，输入为 [0,1] 范围的 RGB 张量（上游在这一步之前不做 ImageNet 归一化）。
func (m *Model) Forward(x *tensor.Tensor) (*tensor.Tensor, error) {
	return m.forward(x, nil)
}

// TraceFunc 在每一层之后被调用，name 采用 "layer_<idx>_<kind>" 形式，
// 与 tools/dump_golden.py 的命名一致，便于逐层对拍。返回错误会中止前向。
type TraceFunc func(name string, t *tensor.Tensor) error

// ForwardTrace 与 Forward 相同，但逐层回调（用于对拍与调试）。
func (m *Model) ForwardTrace(x *tensor.Tensor, trace TraceFunc) (*tensor.Tensor, error) {
	if trace == nil {
		return nil, fmt.Errorf("dncnn.ForwardTrace: trace 不能为 nil")
	}
	return m.forward(x, trace)
}

func (m *Model) forward(x *tensor.Tensor, trace TraceFunc) (*tensor.Tensor, error) {
	if x == nil {
		return nil, fmt.Errorf("dncnn: 输入张量为 nil")
	}
	if x.C != m.inChannels {
		return nil, fmt.Errorf("dncnn: 输入通道 %d != %d", x.C, m.inChannels)
	}
	if trace == nil && x.H*x.W > tilePixelLimit {
		return m.forwardTiled(x, 0)
	}
	return m.forwardFull(x, trace)
}

// forwardTiled 按行分块计算，仅保留有效行；结果与整图计算逐位相同。
// tileH<=0 时按约 300 MB 的缓冲预算自动选择。
func (m *Model) forwardTiled(x *tensor.Tensor, tileH int) (*tensor.Tensor, error) {
	if tileH <= 0 {
		tileH = 300 << 20 / (x.W * 64 * 4)
		if tileH < 32 {
			tileH = 32
		}
		if tileH > x.H {
			tileH = x.H
		}
	}
	out := tensor.New(1, x.H, x.W)
	for lo := 0; lo < x.H; lo += tileH {
		hi := lo + tileH
		if hi > x.H {
			hi = x.H
		}
		inLo := lo - haloRows
		if inLo < 0 {
			inLo = 0
		}
		inHi := hi + haloRows
		if inHi > x.H {
			inHi = x.H
		}
		sub := cropRows(x, inLo, inHi)
		res, err := m.forwardFull(sub, nil)
		if err != nil {
			return nil, err
		}
		copyRows(out, res, inLo, lo, hi)
	}
	return out, nil
}

// cropRows 取输入张量的行区间 [lo,hi)（三通道）。
func cropRows(x *tensor.Tensor, lo, hi int) *tensor.Tensor {
	h := hi - lo
	sub := tensor.New(x.C, h, x.W)
	planeIn := x.H * x.W
	planeOut := h * x.W
	for c := 0; c < x.C; c++ {
		copy(sub.Data[c*planeOut:(c+1)*planeOut], x.Data[c*planeIn+lo*x.W:c*planeIn+hi*x.W])
	}
	return sub
}

// copyRows 把 src 第 0 行对应整图行 srcOffset，据此把整图行 [dstLo,dstHi) 拷到 dst。
func copyRows(dst, src *tensor.Tensor, srcOffset, dstLo, dstHi int) {
	w := dst.W
	planeDst := dst.H * w
	planeSrc := src.H * w
	for c := 0; c < dst.C; c++ {
		for y := dstLo; y < dstHi; y++ {
			sy := y - srcOffset
			copy(dst.Data[c*planeDst+y*w:c*planeDst+(y+1)*w], src.Data[c*planeSrc+sy*w:c*planeSrc+(sy+1)*w])
		}
	}
}

func (m *Model) forwardFull(x *tensor.Tensor, trace TraceFunc) (*tensor.Tensor, error) {
	hiddenC := m.steps[0].conv.OutC
	bb := m.pool.Get().(*bufs)
	if bb.a == nil || bb.c != hiddenC || bb.h != x.H || bb.w != x.W {
		bb.a = tensor.New(hiddenC, x.H, x.W)
		bb.b = tensor.New(hiddenC, x.H, x.W)
		bb.out = tensor.New(1, x.H, x.W)
		bb.c, bb.h, bb.w = hiddenC, x.H, x.W
	}
	next := bb.a
	other := bb.b
	cur := x
	for _, st := range m.steps {
		dst := next
		isHead := st.idx == headConvIdx
		if isHead {
			dst = bb.out
		}
		var epi *ops.Affine
		if trace == nil {
			epi = st.epi
		}
		if err := st.conv.ForwardIntoAffine(cur, dst, epi, m.workers); err != nil {
			m.pool.Put(bb)
			return nil, fmt.Errorf("dncnn 层 %d: %w", st.idx, err)
		}
		cur = dst
		if !isHead {
			next, other = other, next
		}
		if trace == nil {
			continue
		}
		// 对拍路径：按"卷积 → BN → ReLU"逐个导出，命名与参考实现一致
		if err := trace(fmt.Sprintf("layer_%02d_conv2d", st.idx), cur); err != nil {
			m.pool.Put(bb)
			return nil, err
		}
		if st.bn != nil {
			if err := st.bn.Apply(cur); err != nil {
				m.pool.Put(bb)
				return nil, fmt.Errorf("dncnn 层 %d BN: %w", st.idx, err)
			}
			if err := trace(fmt.Sprintf("layer_%02d_batchnorm2d", st.idx+1), cur); err != nil {
				m.pool.Put(bb)
				return nil, err
			}
		}
		if st.epi != nil && st.epi.Relu {
			ops.ReLUInPlace(cur)
			reluIdx := st.idx + 1
			if st.bn != nil {
				reluIdx++
			}
			if err := trace(fmt.Sprintf("layer_%02d_relu", reluIdx), cur); err != nil {
				m.pool.Put(bb)
				return nil, err
			}
		}
	}
	// 输出缓冲也被复用：调用方若需保留结果，应由自己拷贝（Forward 会拷贝一次）。
	if cur == bb.out {
		res := cur.Clone()
		m.pool.Put(bb)
		return res, nil
	}
	m.pool.Put(bb)
	return cur.Clone(), nil
}
