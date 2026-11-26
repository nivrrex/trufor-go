// Package tensor 提供模型推理所需的最小张量类型。
//
// 约定：所有张量都是单样本的 NCHW 内存布局（batch 固定为 1，不显式保存 batch 维），
// Data 长度为 C*H*W，索引方式为 ((c*H)+h)*W+w。
//
// 本包不含任何共享可变状态，所有方法对单个张量自身是独占写入的；并发安全由调用方
// 保证（不同租户不得同时写同一个 Tensor 的同一区域）。
package tensor

import "fmt"

// Tensor 是单样本 fp32 NCHW 张量。
type Tensor struct {
	C, H, W int
	Data    []float32
}

// New 分配一个已清零的张量。
func New(c, h, w int) *Tensor {
	if c <= 0 || h <= 0 || w <= 0 {
		panic(fmt.Sprintf("tensor.New: 非法维度 c=%d h=%d w=%d", c, h, w))
	}
	return &Tensor{C: c, H: h, W: w, Data: make([]float32, c*h*w)}
}

// NewFromData 用已有数据构造张量，不复制；len(data) 必须等于 c*h*w。
func NewFromData(data []float32, c, h, w int) *Tensor {
	if c <= 0 || h <= 0 || w <= 0 {
		panic(fmt.Sprintf("tensor.NewFromData: 非法维度 c=%d h=%d w=%d", c, h, w))
	}
	if len(data) != c*h*w {
		panic(fmt.Sprintf("tensor.NewFromData: 数据长度 %d != %d", len(data), c*h*w))
	}
	return &Tensor{C: c, H: h, W: w, Data: data}
}

// Len 返回元素个数。
func (t *Tensor) Len() int { return t.C * t.H * t.W }

// Shape 返回 (C, H, W)。
func (t *Tensor) Shape() (int, int, int) { return t.C, t.H, t.W }

// Clone 返回一份深拷贝。
func (t *Tensor) Clone() *Tensor {
	d := make([]float32, len(t.Data))
	copy(d, t.Data)
	return &Tensor{C: t.C, H: t.H, W: t.W, Data: d}
}

// At 读取 (c, h, w) 位置的元素，仅用于调试与非热点路径。
func (t *Tensor) At(c, h, w int) float32 {
	if c < 0 || c >= t.C || h < 0 || h >= t.H || w < 0 || w >= t.W {
		panic(fmt.Sprintf("tensor.At: 下标越界 (%d,%d,%d) 形状 (%d,%d,%d)", c, h, w, t.C, t.H, t.W))
	}
	return t.Data[(c*t.H+h)*t.W+w]
}

// Set 写入 (c, h, w) 位置的元素，仅用于调试与非热点路径。
func (t *Tensor) Set(c, h, w int, v float32) {
	if c < 0 || c >= t.C || h < 0 || h >= t.H || w < 0 || w >= t.W {
		panic(fmt.Sprintf("tensor.Set: 下标越界 (%d,%d,%d) 形状 (%d,%d,%d)", c, h, w, t.C, t.H, t.W))
	}
	t.Data[(c*t.H+h)*t.W+w] = v
}

// Plane 返回第 c 个通道的连续 H*W 视图（不复制）。
func (t *Tensor) Plane(c int) []float32 {
	if c < 0 || c >= t.C {
		panic(fmt.Sprintf("tensor.Plane: 通道越界 %d >= %d", c, t.C))
	}
	n := t.H * t.W
	return t.Data[c*n : (c+1)*n]
}

// transBlock 是 NCHW ↔ token-major 转置的分块尺寸。
// 按块转置可让两个方向的访存都保持局部性（朴素实现的写入步长是 C，命中率很差）。
const transBlock = 16

// ToTokens 把 NCHW 张量展平为 token-major（[H*W][C]）布局，
// 等价于 PyTorch 的 x.flatten(2).transpose(1, 2)。
func (t *Tensor) ToTokens() []float32 {
	n := t.H * t.W
	c := t.C
	out := make([]float32, n*c)
	for i0 := 0; i0 < n; i0 += transBlock {
		iEnd := i0 + transBlock
		if iEnd > n {
			iEnd = n
		}
		for c0 := 0; c0 < c; c0 += transBlock {
			cEnd := c0 + transBlock
			if cEnd > c {
				cEnd = c
			}
			for ch := c0; ch < cEnd; ch++ {
				src := t.Data[ch*n+i0 : ch*n+iEnd]
				for k, v := range src {
					out[(i0+k)*c+ch] = v
				}
			}
		}
	}
	return out
}

// FromTokens 是 ToTokens 的逆操作：token-major → NCHW。
func FromTokens(tok []float32, c, h, w int) *Tensor {
	n := h * w
	if len(tok) != n*c {
		panic(fmt.Sprintf("tensor.FromTokens: 长度 %d != %d", len(tok), n*c))
	}
	out := New(c, h, w)
	FromTokensInto(tok, c, h, w, out)
	return out
}

// FromTokensInto 将 token-major 数据转置写入已有 NCHW 张量，避免重复分配。
func FromTokensInto(tok []float32, c, h, w int, out *Tensor) {
	n := h * w
	if len(tok) != n*c {
		panic(fmt.Sprintf("tensor.FromTokens: 长度 %d != %d", len(tok), n*c))
	}
	if out.C != c || out.H != h || out.W != w || len(out.Data) != n*c {
		panic(fmt.Sprintf("tensor.FromTokensInto: 目标形状或长度不匹配"))
	}
	for i0 := 0; i0 < n; i0 += transBlock {
		iEnd := i0 + transBlock
		if iEnd > n {
			iEnd = n
		}
		for c0 := 0; c0 < c; c0 += transBlock {
			cEnd := c0 + transBlock
			if cEnd > c {
				cEnd = c
			}
			for ch := c0; ch < cEnd; ch++ {
				dst := out.Data[ch*n+i0 : ch*n+iEnd]
				for k := range dst {
					dst[k] = tok[(i0+k)*c+ch]
				}
			}
		}
	}
}
