// Package trufor 是 TruFor（图像篡改检测与定位）的 Go 推理实现对外接口。
//
// 与上游 PyTorch 参考实现（github.com/grip-unina/TruFor）的输出保持一致：
// 给定一张 RGB 图像，返回定位图 map、置信图 conf、整图分数 score 与 Noiseprint++。
//
// 用法：
//
//	m, err := trufor.Load(trufor.Config{Weights: "trufor.pth.tar"})
//	if err != nil { ... }
//	defer m.Close()
//	res, err := m.DetectFile("image.png")
//	if err != nil { ... }
//	fmt.Println(res.Score, len(res.Map))
package trufor

import (
	"fmt"
	"image"
	"os"
	"runtime"

	"github.com/nivrrex/trufor-go/internal/imageio"
	"github.com/nivrrex/trufor-go/internal/model/trufor"
	"github.com/nivrrex/trufor-go/internal/tensor"
	"github.com/nivrrex/trufor-go/internal/weights"
)

// Config 是加载模型时的配置。
type Config struct {
	// Weights 是官方权重文件（trufor.pth.tar）路径，必填。
	Weights string
	// Threads 是算子并行度，<=0 时取 runtime.NumCPU()。
	Threads int
}

// Result 是一次推理的输出。所有切片均为 row-major、长度 Width*Height。
type Result struct {
	// Map 是定位图（伪造类概率，0~1）。
	Map []float32
	// Conf 是置信图（sigmoid 后的 TCP 置信度，0~1）。
	Conf []float32
	// Noiseprint 是 Noiseprint++（未归一化的噪声残差指纹）。
	Noiseprint []float32
	// Score 是整图完整性分数（0~1，越大越可能被篡改）。
	Score float32
	// Width、Height 是输入图像尺寸。
	Width, Height int
}

// Model 是已加载的模型。可并发调用 Detect* 方法。
type Model struct {
	inner   *trufor.Model
	workers int
}

// Load 从 Config 指定的权重文件加载模型。
func Load(cfg Config) (*Model, error) {
	if cfg.Weights == "" {
		return nil, fmt.Errorf("trufor: 未指定权重文件")
	}
	if _, err := os.Stat(cfg.Weights); err != nil {
		return nil, fmt.Errorf("trufor: 权重文件不可用: %w", err)
	}
	workers := cfg.Threads
	if workers <= 0 {
		workers = runtime.NumCPU()
	}
	w, err := weights.Open(cfg.Weights)
	if err != nil {
		return nil, fmt.Errorf("trufor: 读取权重: %w", err)
	}
	inner, err := trufor.New(w, workers)
	if err != nil {
		return nil, fmt.Errorf("trufor: 构造模型: %w", err)
	}
	return &Model{inner: inner, workers: workers}, nil
}

// Threads 返回加载时确定的并行度。
func (m *Model) Threads() int { return m.workers }

// DetectImage 对已解码的图像执行推理。
func (m *Model) DetectImage(img image.Image) (*Result, error) {
	x, err := imageio.FromImage(img, "")
	if err != nil {
		return nil, err
	}
	return m.detect(x)
}

// DetectFile 解码文件后执行推理。支持 PNG/JPEG/GIF。
//
// 注意：PNG 的解码与上游 PIL 逐像素一致；JPEG 受解码器差异影响，score 可能有
// 0.1 量级偏差（详见项目 README 的“已知差异”一节）。
func (m *Model) DetectFile(path string) (*Result, error) {
	x, err := imageio.LoadRGB(path)
	if err != nil {
		return nil, err
	}
	return m.detect(x)
}

func (m *Model) detect(x *tensor.Tensor) (*Result, error) {
	if m == nil || m.inner == nil {
		return nil, fmt.Errorf("trufor: 模型未加载")
	}
	r, err := m.inner.Forward(x)
	if err != nil {
		return nil, err
	}
	return &Result{
		Map:        r.Map.Data,
		Conf:       r.Conf.Data,
		Noiseprint: r.Noiseprint.Data,
		Score:      r.Score,
		Width:      x.W,
		Height:     x.H,
	}, nil
}

// Close 释放模型持有的资源（当前实现无需显式释放，保留以兼容未来版本）。
func (m *Model) Close() error { return nil }
