// Package npz 写出与 numpy.savez 兼容的 .npz（zip 容器 + .npy 成员）。
//
// 支持的 dtype：float32（<f4）、float64（<f8）、int64（<i8），足以复刻上游
// trufor_test.py 的 np.savez 产物（map/conf/np++ 为 float32，score 为 float64 标量，
// imgsize 为 int64 数组）。
package npz

import (
	"archive/zip"
	"encoding/binary"
	"fmt"
	"math"
	"os"
	"sort"
	"strings"
)

// Dtype 是成员的 numpy dtype。
type Dtype string

const (
	Float32 Dtype = "<f4"
	Float64 Dtype = "<f8"
	Int64   Dtype = "<i8"
)

// Array 是一个待写入的数组（只填与 Dtype 对应的那个字段）。
type Array struct {
	Shape []int
	Dtype Dtype
	F32   []float32
	F64   []float64
	I64   []int64
}

// F32 构造 float32 数组。
func F32(shape []int, data []float32) Array { return Array{Shape: shape, Dtype: Float32, F32: data} }

// ScalarF64 构造 float64 标量（与 Python float 一致，np.savez 存为 0 维 <f8）。
func ScalarF64(v float64) Array { return Array{Shape: []int{}, Dtype: Float64, F64: []float64{v}} }

// I64 构造 int64 数组。
func I64(shape []int, data []int64) Array { return Array{Shape: shape, Dtype: Int64, I64: data} }

// Write 把 arrays 写成 path。名字即 .npz 内的成员名（不含 .npy 后缀）。
func Write(path string, arrays map[string]Array) error {
	if len(arrays) == 0 {
		return fmt.Errorf("npz: 没有可写数组")
	}
	names := make([]string, 0, len(arrays))
	for k := range arrays {
		names = append(names, k)
	}
	sort.Strings(names)

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	zw := zip.NewWriter(f)
	for _, name := range names {
		if err := writeNpy(zw, name+".npy", arrays[name]); err != nil {
			f.Close()
			return err
		}
	}
	if err := zw.Close(); err != nil {
		f.Close()
		return err
	}
	return f.Close()
}

// writeNpy 写一个 C 顺序、小端的 .npy 成员（numpy 格式 1.0）。
func writeNpy(zw *zip.Writer, name string, a Array) error {
	n := 1
	for _, d := range a.Shape {
		if d < 0 {
			return fmt.Errorf("npz: %s 形状非法 %v", name, a.Shape)
		}
		n *= d
	}
	var elem int
	switch a.Dtype {
	case Float32:
		elem = len(a.F32)
	case Float64:
		elem = len(a.F64)
	case Int64:
		elem = len(a.I64)
	default:
		return fmt.Errorf("npz: %s 的 dtype %q 不支持", name, a.Dtype)
	}
	if elem != n {
		return fmt.Errorf("npz: %s 数据长度 %d 与形状 %v 不符", name, elem, a.Shape)
	}

	shape := "("
	for i, d := range a.Shape {
		if i > 0 {
			shape += ", "
		}
		shape += fmt.Sprint(d)
	}
	if len(a.Shape) == 1 {
		shape += ","
	}
	shape += ")"
	header := fmt.Sprintf("{'descr': '%s', 'fortran_order': False, 'shape': %s, }", a.Dtype, shape)
	// 头部总长（magic 8 + 长度 2 + header + '\n'）需为 64 的倍数
	const preamble = 10
	pad := (64 - (preamble+len(header)+1)%64) % 64
	header += strings.Repeat(" ", pad) + "\n"

	w, err := zw.Create(name)
	if err != nil {
		return err
	}
	if _, err := w.Write([]byte("\x93NUMPY\x01\x00")); err != nil {
		return err
	}
	var l [2]byte
	binary.LittleEndian.PutUint16(l[:], uint16(len(header)))
	if _, err := w.Write(l[:]); err != nil {
		return err
	}
	if _, err := w.Write([]byte(header)); err != nil {
		return err
	}

	buf := make([]byte, bytesPerElem(a.Dtype)*n)
	for i := 0; i < n; i++ {
		switch a.Dtype {
		case Float32:
			binary.LittleEndian.PutUint32(buf[i*4:], math.Float32bits(a.F32[i]))
		case Float64:
			binary.LittleEndian.PutUint64(buf[i*8:], math.Float64bits(a.F64[i]))
		case Int64:
			binary.LittleEndian.PutUint64(buf[i*8:], uint64(a.I64[i]))
		}
	}
	_, err = w.Write(buf)
	return err
}

func bytesPerElem(d Dtype) int {
	if d == Float32 {
		return 4
	}
	return 8
}
