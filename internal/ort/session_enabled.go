//go:build trufor_ort && cgo && linux && amd64

// Package ort 通过 ONNX Runtime C API 执行 TruFor ONNX 模型。
package ort

/*
#cgo CFLAGS: -I${SRCDIR}/include
#cgo LDFLAGS: -ldl
#include <stdlib.h>
#include "ort_shim.h"
*/
import "C"

import (
	"fmt"
	"sync"
	"unsafe"
)

// Output 是 ORT 返回的一个 float32 张量；Data 为连续 row-major 数据。
type Output struct {
	Data  []float32
	Shape []int64
}

// Session 是已加载的 ONNX 模型。Run 可并发调用；Close 会等待进行中的 Run 结束。
type Session struct {
	mu  sync.RWMutex
	ptr unsafe.Pointer
}

// Open 加载模型。线程数对应 ORT intra-op 线程数。
func Open(modelPath string, threads int) (*Session, error) {
	if threads < 1 {
		return nil, fmt.Errorf("ort: 线程数必须为正")
	}
	path := C.CString(modelPath)
	defer C.free(unsafe.Pointer(path))
	var errBuf [1024]C.char
	ptr := C.tr_ort_open(path, C.int(threads), &errBuf[0], C.size_t(len(errBuf)))
	if ptr == nil {
		return nil, fmt.Errorf("ort: 加载模型: %s", C.GoString(&errBuf[0]))
	}
	return &Session{ptr: ptr}, nil
}

// Run 执行 CHW float32 输入，返回模型的 3 个输出。
func (s *Session) Run(input []float32, height, width int) ([3]Output, error) {
	var outputs [3]Output
	if s == nil {
		return outputs, fmt.Errorf("ort: session 未加载")
	}
	if height <= 0 || width <= 0 || len(input) != 3*height*width {
		return outputs, fmt.Errorf("ort: 输入形状不匹配")
	}
	s.mu.RLock()
	defer s.mu.RUnlock()
	if s.ptr == nil {
		return outputs, fmt.Errorf("ort: session 已关闭")
	}

	caps := [3]int{2 * height * width, 2 * height * width, 1}
	buffers := (*C.tr_ort_buffer)(C.calloc(3, C.size_t(unsafe.Sizeof(C.tr_ort_buffer{}))))
	if buffers == nil {
		return outputs, fmt.Errorf("ort: 分配输出描述符失败")
	}
	defer C.free(unsafe.Pointer(buffers))
	bufferView := unsafe.Slice(buffers, 3)
	data := make([][]float32, 3)
	for i := 0; i < 3; i++ {
		data[i] = make([]float32, caps[i])
		bufferView[i].data = (*C.float)(unsafe.Pointer(&data[i][0]))
		bufferView[i].capacity = C.size_t(caps[i])
	}
	shape := [4]C.int64_t{1, 3, C.int64_t(height), C.int64_t(width)}
	var errBuf [1024]C.char
	rc := C.tr_ort_run(s.ptr, (*C.float)(unsafe.Pointer(&input[0])), &shape[0],
		buffers, &errBuf[0], C.size_t(len(errBuf)))
	if rc != 0 {
		return outputs, fmt.Errorf("ort: 推理: %s", C.GoString(&errBuf[0]))
	}
	for i := 0; i < 3; i++ {
		n := 1
		shape := make([]int64, int(bufferView[i].rank))
		for j := range shape {
			shape[j] = int64(bufferView[i].shape[j])
			n *= int(shape[j])
		}
		if n < 0 || n > len(data[i]) {
			return outputs, fmt.Errorf("ort: 输出 %d 形状非法", i)
		}
		outputs[i] = Output{Data: data[i][:n], Shape: shape}
	}
	return outputs, nil
}

// Close 释放 ORT session 与动态库句柄。
func (s *Session) Close() error {
	if s == nil {
		return nil
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.ptr != nil {
		C.tr_ort_close(s.ptr)
		s.ptr = nil
	}
	return nil
}
