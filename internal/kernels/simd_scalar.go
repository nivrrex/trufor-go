//go:build !amd64 || trufor_scalar

package kernels

// simdAvailable 在非 amd64 或 trufor_scalar 构建下强制使用纯 Go 回退。
// 该构建标签用于在 amd64 上回归测试非 AVX2 路径，不改变默认构建行为。
func simdAvailable() bool { return false }
