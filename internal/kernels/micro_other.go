//go:build !amd64

package kernels

// simdAvailable 在非 amd64 平台恒为 false（SIMD 内核尚未提供）。
func simdAvailable() bool { return false }

// 非 amd64 平台不提供 SIMD exp/归约内核（由 MaxOf/SumOf/ExpShiftInPlace 的标量路径兜底）。
