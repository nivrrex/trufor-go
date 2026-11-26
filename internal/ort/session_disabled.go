//go:build !trufor_ort || !cgo || !linux || !amd64

package ort

import "fmt"

// Output 是 ORT 返回的张量。
type Output struct {
	Data  []float32
	Shape []int64
}

// Session 是未启用 ORT 构建时的占位类型。
type Session struct{}

// Open 在未启用 trufor_ort 构建标签时返回明确错误。
func Open(string, int) (*Session, error) {
	return nil, fmt.Errorf("ORT 后端未启用：请使用 CGO_ENABLED=1 且 -tags trufor_ort 构建（当前仅 linux/amd64）")
}

// Run 在未启用 ORT 构建时返回错误。
func (*Session) Run([]float32, int, int) ([3]Output, error) {
	return [3]Output{}, fmt.Errorf("ORT 后端未启用")
}

// Close 在未启用 ORT 构建时无操作。
func (*Session) Close() error { return nil }
