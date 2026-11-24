package trufor_test

import (
	"os"
	"path/filepath"
	"testing"

	"github.com/nivrrex/trufor-go"
	"github.com/nivrrex/trufor-go/internal/testgolden"
)

// weightsPath 返回官方权重路径；不存在时返回空串（测试跳过）。
func weightsPath() string {
	if v := os.Getenv("TRUFOR_WEIGHTS"); v != "" {
		return v
	}
	p := filepath.Join(testgolden.Root(), "ref_weights", "weights", "trufor.pth.tar")
	if _, err := os.Stat(p); err == nil {
		return p
	}
	return ""
}

// TestPublicAPI 用公开接口跑一张端到端用例，并对照黄金数据检查输出。
func TestPublicAPI(t *testing.T) {
	if testing.Short() {
		t.Skip("-short 模式跳过端到端推理")
	}
	wp := weightsPath()
	goldenDir := testgolden.DefaultDir()
	if wp == "" || testgolden.SkipIfMissing(goldenDir) {
		t.Skip("缺少官方权重或黄金数据，跳过")
	}
	g, err := testgolden.Load(goldenDir)
	if err != nil {
		t.Fatalf("加载黄金数据: %v", err)
	}
	var e *testgolden.E2ECase
	for i := range g.EndToEnd {
		if g.EndToEnd[i].Format == "png" && g.EndToEnd[i].MapFile != "" {
			e = &g.EndToEnd[i]
			break
		}
	}
	if e == nil {
		t.Skip("黄金数据中没有 PNG 端到端用例")
	}
	imgPath := g.EndToEndImagePath(e.Name)
	if _, err := os.Stat(imgPath); err != nil {
		t.Skipf("缺少测试图像 %s", imgPath)
	}

	m, err := trufor.Load(trufor.Config{Weights: wp, Threads: 4})
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	defer m.Close()
	if m.Threads() != 4 {
		t.Fatalf("Threads() = %d", m.Threads())
	}

	res, err := m.DetectFile(imgPath)
	if err != nil {
		t.Fatalf("DetectFile: %v", err)
	}
	if res.Width != e.InputShape[2] || res.Height != e.InputShape[1] {
		t.Fatalf("尺寸 (%d,%d) != %v", res.Width, res.Height, e.InputShape)
	}
	for _, v := range [][]float32{res.Map, res.Conf, res.Noiseprint} {
		if len(v) != res.Width*res.Height {
			t.Fatalf("输出长度 %d != %d", len(v), res.Width*res.Height)
		}
	}
	if d := res.Score - float32(e.Score); d > 1e-3 || d < -1e-3 {
		t.Fatalf("score %.6f 与参考 %.6f 偏差过大", res.Score, e.Score)
	}
	want, err := g.ReadEndToEnd(e.MapFile, []int{e.InputShape[1], e.InputShape[2]})
	if err != nil {
		t.Fatalf("读取 map: %v", err)
	}
	var maxDiff float64
	for i := range want {
		d := float64(res.Map[i] - want[i])
		if d < 0 {
			d = -d
		}
		if d > maxDiff {
			maxDiff = d
		}
	}
	t.Logf("public API: score=%.6f ref=%.6f map maxDiff=%.3e", res.Score, e.Score, maxDiff)
	if maxDiff > 1e-3 {
		t.Fatalf("map 最大差 %.3e 超过 1e-3", maxDiff)
	}
}

// TestLoadErrors 校验参数错误路径。
func TestLoadErrors(t *testing.T) {
	if _, err := trufor.Load(trufor.Config{}); err == nil {
		t.Fatal("未指定权重应报错")
	}
	if _, err := trufor.Load(trufor.Config{Weights: "/nonexistent/trufor.pth.tar"}); err == nil {
		t.Fatal("权重不存在应报错")
	}
	var m *trufor.Model
	if _, err := m.DetectFile("x.png"); err == nil {
		t.Fatal("nil 模型应报错")
	}
}
