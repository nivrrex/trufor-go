// Package ops 提供 TruFor 推理所需的 CPU 算子。
//
// 线程安全：本包的函数/方法自身不持有可变全局状态；并行方法内部按行/通道切分，
// 输出分区互不重叠，因此同一输入并发调用是安全的。调用方不得在并发执行期间
// 修改传入的权重或输入张量。
package ops

import "github.com/nivrrex/trufor-go/internal/kernels"

// Gemm 计算 C[M×N] += A[M×K] · B[K×N]（累加语义，C 需由调用方初始化），
// A/B/C 均为 row-major 连续存储。workers<=1 时串行。
// 实现委托给 internal/kernels：支持 AVX2+FMA 时走汇编微内核，否则走纯 Go 回退。
// 该入口（Gemm/GemmStrided）用于 M 较大的场景（Linear、注意力），走免打包 B 的快路；
// 卷积走 gemmConv/gemmRaw（B 打包版）。
// 每个输出元素的累加顺序固定为 k 递增，因此结果与 workers 无关（可复现）。
func Gemm(a []float32, m, k int, b []float32, n int, c []float32, workers int) {
	GemmStrided(a, m, k, b, n, c, n, workers)
}

// gemmRaw 是最底层的 GEMM 入口，显式给出 A/B/C 三者的跨度。
// 用于 1×1 卷积这类"无需 im2col"的场景：B 直接引用输入通道平面（ldb=H*W）。
// 走 B 打包版本（卷积的 M 通常较小，打包更划算）。
func gemmRaw(a []float32, m, k int, lda int, b []float32, n int, ldb int, c []float32, ldc int, workers int) {
	if m <= 0 || n <= 0 || k <= 0 {
		return
	}
	kernels.Sgemm(m, n, k, a, lda, b, ldb, c, ldc, workers)
}

// gemmConv 供卷积使用。M 较小时（如 DnCNN 的 OutC=64）B 打包能显著提升微面板复用率；
// M 较大时（主干/解码头的 128~512 通道）免打包直接读更快，避免一次性拷贝。
func gemmConv(a []float32, m, k int, b []float32, n int, c []float32, ldc int, workers int) {
	if m <= 0 || n <= 0 || k <= 0 {
		return
	}
	if m >= packBDirectThreshold {
		kernels.SgemmDirect(m, n, k, a, k, b, n, c, ldc, workers)
		return
	}
	kernels.Sgemm(m, n, k, a, k, b, n, c, ldc, workers)
}

// packBDirectThreshold 是"跳过 B 打包"的 M 阈值（实测 M=64 打包更快，M≥128 免打包更快）。
const packBDirectThreshold = 96

// GemmStrided 与 Gemm 相同，但 C 的行跨度为 ldc（要求 ldc >= n）。
func GemmStrided(a []float32, m, k int, b []float32, n int, c []float32, ldc int, workers int) {
	if m <= 0 || n <= 0 || k <= 0 {
		return
	}
	if ldc < n {
		panic("ops.GemmStrided: ldc < n")
	}
	kernels.SgemmDirect(m, n, k, a, k, b, n, c, ldc, workers)
}
