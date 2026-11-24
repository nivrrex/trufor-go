// Package testgolden 读取 tools/dump_golden.py 产出的黄金数据，供各包测试对拍。
//
// 该包只被 _test.go 使用；它不参与生产路径。
package testgolden

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"runtime"
)

// LayerRef 描述一个层快照。
type LayerRef struct {
	Name  string    `json:"name"`
	Shape []int     `json:"shape"`
	File  string    `json:"file"`
	Stats LayerStat `json:"stats"`
}

// LayerStat 是参考实现的统计量，用于快速判断"整体量级是否对得上"。
type LayerStat struct {
	Min    float64 `json:"min"`
	Max    float64 `json:"max"`
	Mean   float64 `json:"mean"`
	AbsSum float64 `json:"abs_sum"`
}

// TensorRef 描述一个权重校验张量。
type TensorRef struct {
	Shape []int  `json:"shape"`
	File  string `json:"file"`
}

// E2ECase 是一个端到端用例的元数据。
type E2ECase struct {
	Name           string    `json:"name"`
	Format         string    `json:"format"`
	ImageRel       string    `json:"image_rel"`
	InputShape     []int     `json:"input_shape"`
	Score          float64   `json:"score"`
	MapStats       LayerStat `json:"map_stats"`
	ConfStats      LayerStat `json:"conf_stats"`
	NPStats        LayerStat `json:"np_stats"`
	PixelsFile     string    `json:"pixels_file"`
	LogitsFile     string    `json:"logits_file"`
	ConfLogitsFile string    `json:"conf_logits_file"`
	MapFile        string    `json:"map_file"`
	ConfFile       string    `json:"conf_file"`
	Loc4File       string    `json:"loc4_file"`
	Conf4File      string    `json:"conf4_file"`
	NPFile         string    `json:"np_file"`
	BackboneFiles  map[string]struct {
		File  string `json:"file"`
		Shape []int  `json:"shape"`
	} `json:"backbone_files"`
}

// Manifest 对应 dump_golden.py 写出的 manifest.json。
type Manifest struct {
	Env            map[string]any       `json:"env"`
	Config         map[string]any       `json:"config"`
	NormEps        map[string]float64   `json:"norm_eps"`
	Weights        map[string]TensorRef `json:"weights"`
	DncnnLayers    []LayerRef           `json:"dncnn_layers"`
	BackboneLayers []LayerRef           `json:"backbone_layers"`
	EndToEnd       []E2ECase            `json:"e2e"`

	dir string
}

// Root 返回仓库根目录（以 go.mod 为标志），找不到时以当前目录兜底。
func Root() string {
	_, file, _, ok := runtime.Caller(0)
	if !ok {
		return "."
	}
	dir := filepath.Dir(file)
	for {
		if _, err := os.Stat(filepath.Join(dir, "go.mod")); err == nil {
			return dir
		}
		parent := filepath.Dir(dir)
		if parent == dir {
			return "."
		}
		dir = parent
	}
}

// DefaultDir 为默认黄金数据目录。
func DefaultDir() string { return filepath.Join(Root(), "testdata", "golden") }

// Load 读取 manifest.json。
func Load(dir string) (*Manifest, error) {
	blob, err := os.ReadFile(filepath.Join(dir, "manifest.json"))
	if err != nil {
		return nil, fmt.Errorf("读取 manifest: %w", err)
	}
	var m Manifest
	if err := json.Unmarshal(blob, &m); err != nil {
		return nil, fmt.Errorf("解析 manifest: %w", err)
	}
	m.dir = dir
	return &m, nil
}

// SkipIfMissing 在黄金数据不存在时返回 true（调用方据此 t.Skip）。
func SkipIfMissing(dir string) bool {
	_, err := os.Stat(filepath.Join(dir, "manifest.json"))
	return err != nil
}

// ReadWeights 读取 <dir>/weights/<file> 的 float32 数据。
func (m *Manifest) ReadWeights(file string, shape []int) ([]float32, error) {
	return readF32(filepath.Join(m.dir, "weights", file), shape)
}

// ReadEndToEnd 读取 <dir>/<file> 的 float32 数据。
func (m *Manifest) ReadEndToEnd(file string, shape []int) ([]float32, error) {
	return readF32(filepath.Join(m.dir, file), shape)
}

// EndToEndImagePath 返回端到端用例对应的原图路径。
// 优先用随仓库分发的 testdata/images（来自上游仓库的示例图，授权一致）；
// 若缺失则回退到本地浅克隆的上游源码目录，两者都没有时调用方应跳过。
func (m *Manifest) EndToEndImagePath(name string) string {
	local := filepath.Join(Root(), "testdata", "images", name)
	if _, err := os.Stat(local); err == nil {
		return local
	}
	return filepath.Join(Root(), "upstream", "test_docker", "images", name)
}

// ReadBackbone 读取 <dir>/<file> 的 float32 数据（file 为相对 golden 目录的路径）。
func (m *Manifest) ReadBackbone(file string, shape []int) ([]float32, error) {
	return readF32(filepath.Join(m.dir, file), shape)
}

// ReadDncnn 读取 <dir>/dncnn/<file> 的 float32 数据。
func (m *Manifest) ReadDncnn(file string, shape []int) ([]float32, error) {
	return readF32(filepath.Join(m.dir, "dncnn", file), shape)
}

func readF32(path string, shape []int) ([]float32, error) {
	blob, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("读取 %s: %w", path, err)
	}
	n := 1
	for _, d := range shape {
		n *= d
	}
	if len(blob) != n*4 {
		return nil, fmt.Errorf("%s 长度 %d 与 shape %v 不符（期望 %d 字节）", path, len(blob), shape, n*4)
	}
	out := make([]float32, n)
	for i := 0; i < n; i++ {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(blob[i*4:]))
	}
	return out, nil
}
