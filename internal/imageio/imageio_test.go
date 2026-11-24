package imageio

import (
	"os"
	"testing"

	"github.com/nivrrex/trufor-go/internal/testgolden"
)

// TestLoadPNGMatchesPIL 校验 PNG 解码与 PIL 逐像素一致（黄金数据由 dump_golden.py 写出）。
func TestLoadPNGMatchesPIL(t *testing.T) {
	golden := testgolden.DefaultDir()
	if testgolden.SkipIfMissing(golden) {
		t.Skip("缺少 testdata/golden，跳过")
	}
	g, err := testgolden.Load(golden)
	if err != nil {
		t.Fatalf("加载黄金数据: %v", err)
	}
	if len(g.EndToEnd) == 0 {
		t.Skip("黄金数据中没有端到端用例")
	}
	checked := 0
	for _, e := range g.EndToEnd {
		if e.PixelsFile == "" {
			continue
		}
		want, err := g.ReadEndToEnd(e.PixelsFile, e.InputShape)
		if err != nil {
			t.Fatalf("%s: %v", e.Name, err)
		}
		imgPath := g.EndToEndImagePath(e.Name)
		if _, err := os.Stat(imgPath); err != nil {
			t.Skipf("缺少测试图像 %s（来自上游仓库的 test_docker/images，未随包分发）", imgPath)
		}
		got, err := LoadRGB(imgPath)
		if err != nil {
			t.Fatalf("%s: %v", e.Name, err)
		}
		if got.C != e.InputShape[0] || got.H != e.InputShape[1] || got.W != e.InputShape[2] {
			t.Fatalf("%s 形状 (%d,%d,%d) != %v", e.Name, got.C, got.H, got.W, e.InputShape)
		}
		var maxDiff float64
		for i := range want {
			d := float64(got.Data[i]) - float64(want[i])
			if d < 0 {
				d = -d
			}
			if d > maxDiff {
				maxDiff = d
			}
		}
		t.Logf("%-24s maxDiff=%.3e", e.Name, maxDiff)
		if e.Format == "png" && maxDiff != 0 {
			t.Fatalf("%s: PNG 解码与 PIL 不一致，最大差 %g", e.Name, maxDiff)
		}
		checked++
	}
	if checked == 0 {
		t.Skip("没有可用的像素用例")
	}
}
