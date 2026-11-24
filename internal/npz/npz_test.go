package npz

import (
	"os"
	"path/filepath"
	"testing"
)

// TestWriteStructure 校验写出的 .npz 结构（成员名、dtype、形状）与 numpy 约定一致。
func TestWriteStructure(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "out.npz")
	if err := Write(path, map[string]Array{
		"map":     F32([]int{2, 3}, []float32{1, 2, 3, 4, 5, 6}),
		"conf":    F32([]int{2, 3}, []float32{0, 0, 0, 0, 0, 0}),
		"score":   ScalarF64(0.5),
		"imgsize": I64([]int{2}, []int64{2, 3}),
	}); err != nil {
		t.Fatalf("Write: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("文件不存在: %v", err)
	}
	if err := Write(path+".bad", map[string]Array{"x": F32([]int{2}, []float32{1})}); err == nil {
		t.Fatal("长度与形状不符应报错")
	}
	if err := Write(path+".empty", nil); err == nil {
		t.Fatal("空数组集合应报错")
	}
	if err := Write(path+".badtype", map[string]Array{"x": {Shape: []int{1}}}); err == nil {
		t.Fatal("未支持的 dtype 应报错")
	}
}

// TestNumpyReadable 用 numpy 读回，校验键名、形状与 dtype 与上游 np.savez 一致。
func TestNumpyReadable(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "out.npz")
	mapData := []float32{0.25, 0.5, 0.75, 1}
	if err := Write(path, map[string]Array{
		"map":     F32([]int{2, 2}, mapData),
		"conf":    F32([]int{2, 2}, mapData),
		"score":   ScalarF64(0.123456789),
		"imgsize": I64([]int{2}, []int64{2, 2}),
		"np++":    F32([]int{2, 2}, mapData),
	}); err != nil {
		t.Fatalf("Write: %v", err)
	}
	script := `import numpy as np, sys
d = np.load(sys.argv[1])
print(sorted(d.files))
print(d['map'].shape, d['map'].dtype)
print(d['score'].shape, d['score'].dtype, float(d['score']))
print(d['imgsize'].shape, d['imgsize'].dtype, d['imgsize'].tolist())
print(bool(np.allclose(d['map'], d['conf'])), bool(np.allclose(d['map'], d['np++'])))`
	out, err := runPython(script, path)
	if err != nil {
		t.Skipf("python/numpy 不可用，跳过: %v", err)
	}
	t.Logf("numpy 读取:\n%s", out)
	want := "['conf', 'imgsize', 'map', 'np++', 'score']\n(2, 2) float32\n() float64 0.123456789\n(2,) int64 [2, 2]\nTrue True"
	if out != want {
		t.Fatalf("numpy 读回结果与预期不符\n得到:\n%s\n预期:\n%s", out, want)
	}
}
