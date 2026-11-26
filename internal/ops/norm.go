package ops

import (
	"fmt"
	"math"
	"sync"

	"github.com/nivrrex/trufor-go/internal/kernels"
	"github.com/nivrrex/trufor-go/internal/tensor"
)

// LayerNorm 是 nn.LayerNorm 的推理实现，沿通道维归一化（等价于对每个 token 归一化）。
// 上游存在两种 eps：OverlapPatchEmbed 内为默认 1e-5，Block/CrossPath 内为 1e-6，
// 因此 eps 必须由调用方显式给出。
type LayerNorm struct {
	C     int
	eps   float64
	gamma []float32
	beta  []float32
}

// NewLayerNorm 构造 LayerNorm；gamma/beta 长度必须等于通道数。
func NewLayerNorm(gamma, beta []float32, eps float64) (*LayerNorm, error) {
	c := len(gamma)
	if c == 0 || len(beta) != c {
		return nil, fmt.Errorf("layernorm 参数长度不一致: gamma=%d beta=%d", len(gamma), len(beta))
	}
	if eps <= 0 {
		return nil, fmt.Errorf("layernorm eps 非法: %v", eps)
	}
	return &LayerNorm{C: c, eps: eps, gamma: gamma, beta: beta}, nil
}

// Apply 就地归一化（NCHW，逐像素跨通道）。
func (l *LayerNorm) Apply(x *tensor.Tensor) error {
	if x.C != l.C {
		return fmt.Errorf("layernorm 输入通道 %d != %d", x.C, l.C)
	}
	plane := x.H * x.W
	inv := 1.0 / float64(l.C)
	for p := 0; p < plane; p++ {
		var sum float64
		for c := 0; c < l.C; c++ {
			sum += float64(x.Data[c*plane+p])
		}
		mean := sum * inv
		var varSum float64
		for c := 0; c < l.C; c++ {
			d := float64(x.Data[c*plane+p]) - mean
			varSum += d * d
		}
		varScale := 1.0 / math.Sqrt(varSum*inv+l.eps)
		for c := 0; c < l.C; c++ {
			v := (float64(x.Data[c*plane+p]) - mean) * varScale
			x.Data[c*plane+p] = float32(v*float64(l.gamma[c]) + float64(l.beta[c]))
		}
	}
	return nil
}

// GELUInPlace 就地执行精确 GELU（PyTorch nn.GELU 默认 approximate='none'）：
// 0.5*x*(1+erf(x/sqrt(2)))。有 AVX2 时走 SIMD 版 erf（A&S 7.1.26，误差 < 1.5e-7）。
func GELUInPlace(x *tensor.Tensor) {
	kernels.GELUInPlace(x.Data)
}

// GELUInPlaceParallel 与 GELUInPlace 相同，但大切片按连续区间并行。
func GELUInPlaceParallel(data []float32, workers int) {
	parallelFor(len(data), workers, 1<<20, func(lo, hi int) {
		kernels.GELUInPlace(data[lo:hi])
	})
}

// SigmoidInPlace 就地执行 1/(1+exp(-x))。
func SigmoidInPlace(x *tensor.Tensor) {
	for i, v := range x.Data {
		x.Data[i] = float32(1 / (1 + math.Exp(-float64(v))))
	}
}

// SoftmaxRows 对 rows×cols 的矩阵按行做 softmax（就地）。
func SoftmaxRows(data []float32, rows, cols int) {
	for r := 0; r < rows; r++ {
		row := data[r*cols : (r+1)*cols]
		softmaxInPlace(row)
	}
}

func softmaxInPlace(v []float32) {
	if len(v) == 0 {
		return
	}
	maxV := kernels.MaxOf(v)
	kernels.ExpShiftInPlace(v, maxV)
	sum := kernels.SumOf(v)
	if sum == 0 {
		return
	}
	inv := float32(1 / sum)
	for i := range v {
		v[i] *= inv
	}
}

// inPlaceTranspose 转置 n×d 矩阵（用于把 NCHW 平面上的 token×channel 数据按头切分）。
func splitHeads(tok []float32, n, heads, d int) []float32 {
	// tok 布局为 [n][heads*d]，输出为 [heads][n][d]
	out := make([]float32, n*heads*d)
	for i := 0; i < n; i++ {
		src := tok[i*heads*d : (i+1)*heads*d]
		for h := 0; h < heads; h++ {
			dst := out[(h*n+i)*d : (h*n+i+1)*d]
			copy(dst, src[h*d:(h+1)*d])
		}
	}
	return out
}

// mergeHeads 与 splitHeads 相反：把 [heads][n][d] 合回 [n][heads*d]。
func mergeHeads(headsData []float32, n, heads, d int) []float32 {
	out := make([]float32, n*heads*d)
	for h := 0; h < heads; h++ {
		for i := 0; i < n; i++ {
			src := headsData[(h*n+i)*d : (h*n+i+1)*d]
			copy(out[i*heads*d+h*d:], src)
		}
	}
	return out
}

// attnChunk 是注意力分数的分块行数，限制单块 scores 缓冲在数 MB 级别。
const attnChunk = 4096

// attentionScorePool 复用每个 query 块 worker 的 scores 缓冲。缓冲所有权只在该
// worker 内，函数返回前不得再读取；sync.Pool 由 GC 自动清理。
var attentionScorePool sync.Pool

func getAttentionScores(n int) []float32 {
	if v := attentionScorePool.Get(); v != nil {
		if buf := *(v.(*[]float32)); cap(buf) >= n {
			return buf[:n]
		}
	}
	return make([]float32, n)
}

func putAttentionScores(buf []float32) {
	// 不保留过大的临时缓冲，避免模型常驻时把多阶段大图内存锁住。
	if cap(buf) <= 128<<20 {
		attentionScorePool.Put(&buf)
	}
}

// MultiHeadSelfAttention 对 n 个 token、dim 通道的多头自注意力：
// out[n×dim] = concat_h( softmax(q_h k_hᵀ * scale) v_h )，scale = headDim^-0.5。
// 两次矩阵乘法都走 GEMM 内核（与 PyTorch 的 scaled_dot_product_attention 等价）。
func MultiHeadSelfAttention(q, k, v, out []float32, n, s, dim, heads, workers int) {
	if dim%heads != 0 {
		panic("MultiHeadSelfAttention: dim 不能被 heads 整除")
	}
	if n <= 0 || s <= 0 {
		return
	}
	if workers < 1 {
		workers = 1
	}
	d := dim / heads
	scale := float32(1 / math.Sqrt(float64(d)))
	qh := q
	kh := k
	vh := v
	oh := out
	if heads > 1 {
		// heads==1 时 split/merge 只是恒等映射，省掉三次整块拷贝
		qh = splitHeads(q, n, heads, d)
		kh = splitHeads(k, s, heads, d)
		vh = splitHeads(v, s, heads, d)
		oh = make([]float32, n*dim)
	}
	kt := make([]float32, heads*d*s)
	for h := 0; h < heads; h++ {
		ktHead := kt[h*d*s:]
		for p := 0; p < s; p++ {
			base := (h*s + p) * d
			for c := 0; c < d; c++ {
				ktHead[c*s+p] = kh[base+c]
			}
		}
	}
	chunks := (n + attnChunk - 1) / attnChunk
	if workers == 1 || chunks*heads == 1 {
		// 单任务时保留原路径：两次 GEMM 各自利用 workers，避免把可并行的 N 方向退化掉。
		scores := getAttentionScores(attnChunk * s)
		defer putAttentionScores(scores)
		for h := 0; h < heads; h++ {
			for lo := 0; lo < n; lo += attnChunk {
				hi := lo + attnChunk
				if hi > n {
					hi = n
				}
				m := hi - lo
				sc := scores[:m*s]
				for i := range sc {
					sc[i] = 0
				}
				GemmStrided(qh[(h*n+lo)*d:], m, d, kt[h*d*s:], s, sc, s, workers)
				for i := 0; i < m; i++ {
					row := sc[i*s : (i+1)*s]
					for j := range row {
						row[j] *= scale
					}
					softmaxInPlace(row)
				}
				GemmStrided(sc, m, s, vh[h*s*d:], d, oh[(h*n+lo)*d:], d, workers)
			}
		}
	} else {
		// query 块之间没有依赖。每个 worker 独占一块 scores，连续完成 QK→softmax→AV，
		// 只同步一次；这同时避免了 softmax 串行和 AV-GEMM 在 n=d≤64 时退化为串行。
		// 每块仍按 k 递增累加，浮点运算顺序与串行路径一致。
		tasks := chunks * heads
		parallel := min(workers, tasks)
		const scoreBudget = 256 << 20 // float32 元素；限制新增 scores 缓冲总量约 1 GiB。
		rowsPerBuffer := min(attnChunk, n)
		if byMemory := scoreBudget / (rowsPerBuffer * s); byMemory > 0 {
			parallel = min(parallel, byMemory)
		}
		if parallel < 1 {
			parallel = 1
		}
		var wg sync.WaitGroup
		for w := 0; w < parallel; w++ {
			wg.Add(1)
			go func(w int) {
				defer wg.Done()
				scores := getAttentionScores(rowsPerBuffer * s)
				defer putAttentionScores(scores)
				for task := w; task < tasks; task += parallel {
					h := task / chunks
					chunk := task - h*chunks
					lo := chunk * attnChunk
					hi := lo + attnChunk
					if hi > n {
						hi = n
					}
					m := hi - lo
					sc := scores[:m*s]
					for i := range sc {
						sc[i] = 0
					}
					GemmStrided(qh[(h*n+lo)*d:], m, d, kt[h*d*s:], s, sc, s, 1)
					for i := 0; i < m; i++ {
						row := sc[i*s : (i+1)*s]
						for j := range row {
							row[j] *= scale
						}
						softmaxInPlace(row)
					}
					GemmStrided(sc, m, s, vh[h*s*d:], d, oh[(h*n+lo)*d:], d, 1)
				}
			}(w)
		}
		wg.Wait()
	}
	if heads > 1 {
		copy(out, mergeHeads(oh, n, heads, d))
	}
}

// CrossAttentionKV 实现 CMX 的 CrossAttention：ctx = softmax_{dim=-2}( (kᵀ·v) * scale )，
// out[n×dim] = q·ctx。softmax 沿第一个下标（即 kᵀv 的行）归一化。
func CrossAttentionKV(q, k, v, out []float32, n, s, dim, heads, workers int) {
	if dim%heads != 0 {
		panic("CrossAttentionKV: dim 不能被 heads 整除")
	}
	if n <= 0 || s <= 0 {
		return
	}
	if workers < 1 {
		workers = 1
	}
	d := dim / heads
	scale := float32(1 / math.Sqrt(float64(d)))
	qh := q
	kh := k
	vh := v
	oh := out
	if heads > 1 {
		qh = splitHeads(q, n, heads, d)
		kh = splitHeads(k, s, heads, d)
		vh = splitHeads(v, s, heads, d)
		oh = make([]float32, n*dim)
	}
	kt := make([]float32, d*s)
	ctx := make([]float32, d*d)
	col := make([]float32, d)
	for h := 0; h < heads; h++ {
		for p := 0; p < s; p++ {
			base := (h*s + p) * d
			for c := 0; c < d; c++ {
				kt[c*s+p] = kh[base+c]
			}
		}
		for i := range ctx {
			ctx[i] = 0
		}
		GemmStrided(kt, d, s, vh[h*s*d:], d, ctx, d, workers)
		for i := range ctx {
			ctx[i] *= scale
		}
		// softmax over dim=-2：对每一列 j 在 i 方向归一化
		for j := 0; j < d; j++ {
			for i := 0; i < d; i++ {
				col[i] = ctx[i*d+j]
			}
			softmaxInPlace(col)
			for i := 0; i < d; i++ {
				ctx[i*d+j] = col[i]
			}
		}
		GemmStrided(qh[h*n*d:], n, d, ctx, d, oh[h*n*d:], d, workers)
	}
	if heads > 1 {
		copy(out, mergeHeads(oh, n, heads, d))
	}
}

// ReLUSlice 就地执行 max(0, x)（切片级，供非张量数据使用）。
func ReLUSlice(v []float32) {
	for i, x := range v {
		if x < 0 {
			v[i] = 0
		}
	}
}

// SigmoidSlice 就地执行 1/(1+exp(-x))（切片级）。
func SigmoidSlice(v []float32) {
	for i, x := range v {
		v[i] = float32(1 / (1 + math.Exp(-float64(x))))
	}
}

// ApplyTokens 对 token-major（[n][C]）数据沿通道维归一化，等价于对 NCHW 的逐像素归一化。
func (l *LayerNorm) ApplyTokens(tok []float32, n int) error {
	return l.ApplyTokensParallel(tok, n, 1)
}

// ApplyTokensParallel 与 ApplyTokens 相同，但按 token 并行；每个 token 独立归一化。
func (l *LayerNorm) ApplyTokensParallel(tok []float32, n, workers int) error {
	if n <= 0 {
		return fmt.Errorf("layernorm: token 数非法 %d", n)
	}
	if len(tok) < n*l.C {
		return fmt.Errorf("layernorm: 数据长度 %d < %d", len(tok), n*l.C)
	}
	c := l.C
	inv := 1.0 / float64(c)
	parallelFor(n, workers, 1<<15, func(lo, hi int) {
		for i := lo; i < hi; i++ {
			row := tok[i*c : (i+1)*c]
			var sum float64
			for _, v := range row {
				sum += float64(v)
			}
			mean := sum * inv
			var varSum float64
			for _, v := range row {
				d := float64(v) - mean
				varSum += d * d
			}
			scale := 1.0 / math.Sqrt(varSum*inv+l.eps)
			for j := 0; j < c; j++ {
				v := (float64(row[j]) - mean) * scale
				row[j] = float32(v*float64(l.gamma[j]) + float64(l.beta[j]))
			}
		}
	})
	return nil
}
