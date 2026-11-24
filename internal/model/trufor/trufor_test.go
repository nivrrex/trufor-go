package trufor

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"sort"
	"testing"

	"github.com/nivrrex/trufor-go/internal/imageio"
	"github.com/nivrrex/trufor-go/internal/model/decoder"
	"github.com/nivrrex/trufor-go/internal/ops"
	"github.com/nivrrex/trufor-go/internal/tensor"
	"github.com/nivrrex/trufor-go/internal/testgolden"
	"github.com/nivrrex/trufor-go/internal/weights"
)

func checkpointPath() string {
	root := testgolden.Root()
	candidates := []string{filepath.Join(root, "ref_weights", "weights", "trufor.pth.tar")}
	if v := os.Getenv("TRUFOR_WEIGHTS"); v != "" {
		candidates = append([]string{v}, candidates...)
	}
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	return ""
}

// TestEndToEnd 是端到端对拍：以 PyTorch 参考实现的 score / map / conf 为基准。
// 纯 Go（未用汇编）在 256² 上需数十秒，故跳过 -short 模式。
func TestEndToEnd(t *testing.T) {
	if testing.Short() {
		t.Skip("-short 模式跳过端到端推理")
	}
	ckpt := checkpointPath()
	goldenDir := testgolden.DefaultDir()
	if ckpt == "" || testgolden.SkipIfMissing(goldenDir) {
		t.Skip("缺少官方权重或黄金数据，跳过")
	}
	g, err := testgolden.Load(goldenDir)
	if err != nil {
		t.Fatalf("加载黄金数据: %v", err)
	}
	if len(g.EndToEnd) == 0 {
		t.Skip("黄金数据中没有端到端用例")
	}
	w, err := weights.Open(ckpt)
	if err != nil {
		t.Fatalf("加载权重: %v", err)
	}
	m, err := New(w, 4)
	if err != nil {
		t.Fatalf("构造模型: %v", err)
	}

	for _, e := range g.EndToEnd {
		e := e
		t.Run(e.Name, func(t *testing.T) {
			imgPath := g.EndToEndImagePath(e.Name)
			if _, err := os.Stat(imgPath); err != nil {
				t.Skipf("缺少测试图像 %s", imgPath)
			}
			rgb, err := imageio.LoadRGB(imgPath)
			if err != nil {
				t.Fatalf("读图: %v", err)
			}
			checkpoints := map[string][]float32{}
			checkpointsShape := map[string][]int{}
			res, err := m.ForwardTrace(rgb, func(name string, got *tensor.Tensor) error {
				checkpoints[name] = append([]float32(nil), got.Data...)
				checkpointsShape[name] = []int{got.C, got.H, got.W}
				return nil
			})
			if err != nil {
				t.Fatalf("推理: %v", err)
			}

			// 输入像素与 Noiseprint++ 的一致性检查（把误差定位到具体环节）
			if e.PixelsFile != "" {
				refPix, err := g.ReadEndToEnd(e.PixelsFile, e.InputShape)
				if err != nil {
					t.Fatalf("读取像素基准: %v", err)
				}
				pixDiff, pixAt := maxAbsDiff(rgb.Data, refPix)
				t.Logf("输入像素 maxDiff=%.3e", pixDiff)
				if e.Format == "png" && pixDiff != 0 {
					t.Fatalf("PNG 解码与 PIL 不一致，最大差 %g（第 %d 个元素）", pixDiff, pixAt)
				}
			}
			if e.NPFile != "" {
				refNP, err := g.ReadEndToEnd(e.NPFile, []int{1, e.InputShape[1], e.InputShape[2]})
				if err != nil {
					t.Fatalf("读取 NP++ 基准: %v", err)
				}
				npDiff, npAt := maxAbsDiff(res.Noiseprint.Data, refNP)
				t.Logf("Noiseprint++ maxDiff=%.3e", npDiff)
				if e.Format == "png" && npDiff > 1e-3 {
					t.Fatalf("Noiseprint++ 最大差 %.3e（第 %d 个元素 go=%v ref=%v）", npDiff, npAt, res.Noiseprint.Data[npAt], refNP[npAt])
				}
			}

			if e.Loc4File != "" {
				ref, err := g.ReadEndToEnd(e.Loc4File, checkpointsShape["loc4"])
				if err != nil {
					t.Fatalf("读取 loc4: %v", err)
				}
				d, at := maxAbsDiff(checkpoints["loc4"], ref)
				t.Logf("loc4 maxDiff=%.3e", d)
				if e.Format == "png" && d > 1e-2 {
					t.Fatalf("loc4 最大差 %.3e（第 %d 个元素 go=%v ref=%v）", d, at, checkpoints["loc4"][at], ref[at])
				}
			}

			// 若黄金数据带有主干检查点，逐个比较以定位第一个偏离的 stage
			if len(e.BackboneFiles) > 0 && e.Format == "png" {
				if err := compareBackbone(t, g, e, m, rgb); err != nil {
					t.Fatal(err)
				}
			}

			// score：PNG 路径要求 1e-3；JPEG 路径受解码差异限制，仅记录
			scoreDiff := math.Abs(float64(res.Score) - e.Score)
			t.Logf("score: go=%.6f ref=%.6f diff=%.3e", res.Score, e.Score, scoreDiff)
			switch e.Format {
			case "png":
				if scoreDiff > 1e-3 {
					t.Fatalf("score 偏差 %.3e 超过 1e-3", scoreDiff)
				}
			default:
				// JPEG 走标准库解码，与 PIL/libturbojpeg 的 IDCT/上采样不同，
				// 实测 score 偏差 ~0.07、map 最大差 ~0.19（见 docs/porting-notes.md）。
				if scoreDiff > 0.15 {
					t.Fatalf("JPEG 路径 score 偏差 %.3e 超出预期范围（0.15）", scoreDiff)
				}
			}

			if e.MapFile == "" {
				return
			}
			refMap, err := g.ReadEndToEnd(e.MapFile, []int{1, e.InputShape[1], e.InputShape[2]})
			if err != nil {
				t.Fatalf("读取 map: %v", err)
			}
			refConf, err := g.ReadEndToEnd(e.ConfFile, []int{1, e.InputShape[1], e.InputShape[2]})
			if err != nil {
				t.Fatalf("读取 conf: %v", err)
			}
			refLogits, err := g.ReadEndToEnd(e.LogitsFile, []int{2, e.InputShape[1], e.InputShape[2]})
			if err != nil {
				t.Fatalf("读取 logits: %v", err)
			}

			mapDiff, mapAt := maxAbsDiff(res.Map.Data, refMap)
			confDiff, confAt := maxAbsDiff(res.Conf.Data, refConf)
			logitDiff, _ := maxAbsDiff(res.Logits.Data, refLogits)
			t.Logf("map maxDiff=%.3e conf maxDiff=%.3e logits maxDiff=%.3e", mapDiff, confDiff, logitDiff)

			tol := 1e-3
			if e.Format != "png" {
				tol = 0.3 // JPEG 解码差异（实测 map 最大差 ~0.19）
			}
			if mapDiff > tol {
				t.Fatalf("map 最大差 %.3e 超过 %.1e（第 %d 个元素 go=%v ref=%v）", mapDiff, tol, mapAt, res.Map.Data[mapAt], refMap[mapAt])
			}
			if confDiff > tol {
				t.Fatalf("conf 最大差 %.3e 超过 %.1e（第 %d 个元素）", confDiff, tol, confAt)
			}
			if e.Format == "png" && logitDiff > 1e-2 {
				t.Fatalf("logits 最大差 %.3e 超过 1e-2", logitDiff)
			}
		})
	}
}

func maxAbsDiff(got, want []float32) (float64, int) {
	var maxDiff float64
	at := 0
	for i := range want {
		d := math.Abs(float64(got[i]) - float64(want[i]))
		if d > maxDiff {
			maxDiff, at = d, i
		}
	}
	return maxDiff, at
}

// TestDecoderParity 单独对拍解码头的 1/4 分辨率输出，便于把端到端误差定位到
// "主干已对齐 / 解码与后处理有偏差" 的分界。
func TestDecoderParity(t *testing.T) {
	if testing.Short() {
		t.Skip("-short 模式跳过")
	}
	ckpt := checkpointPath()
	goldenDir := testgolden.DefaultDir()
	if ckpt == "" || testgolden.SkipIfMissing(goldenDir) {
		t.Skip("缺少官方权重或黄金数据，跳过")
	}
	g, err := testgolden.Load(goldenDir)
	if err != nil {
		t.Fatalf("加载黄金数据: %v", err)
	}
	for _, e := range g.EndToEnd {
		if e.Loc4File == "" {
			continue
		}
		imgPath := g.EndToEndImagePath(e.Name)
		if _, err := os.Stat(imgPath); err != nil {
			t.Skipf("缺少测试图像 %s", imgPath)
		}
		w, err := weights.Open(ckpt)
		if err != nil {
			t.Fatalf("加载权重: %v", err)
		}
		m, err := New(w, 4)
		if err != nil {
			t.Fatalf("构造模型: %v", err)
		}
		rgb, err := imageio.LoadRGB(imgPath)
		if err != nil {
			t.Fatalf("读图: %v", err)
		}
		res, err := m.Forward(rgb)
		if err != nil {
			t.Fatalf("推理: %v", err)
		}
		h4 := e.InputShape[1] / 4
		w4 := e.InputShape[2] / 4
		refLoc4, err := g.ReadEndToEnd(e.Loc4File, []int{2, h4, w4})
		if err != nil {
			t.Fatalf("读取 loc4: %v", err)
		}
		refConf4, err := g.ReadEndToEnd(e.Conf4File, []int{1, h4, w4})
		if err != nil {
			t.Fatalf("读取 conf4: %v", err)
		}
		// 从结果反推 1/4 分辨率 logits 需要重新跑解码头，这里直接比较上采后的 logits
		// 与"参考 1/4 logits 上采"的差异，同样能区分解码器错误与后处理错误。
		up, err := opsResize(refLoc4, 2, h4, w4, e.InputShape[1], e.InputShape[2])
		if err != nil {
			t.Fatalf("上采参考 logits: %v", err)
		}
		diff, at := maxAbsDiff(res.Logits.Data, up)
		t.Logf("%s: loc4 上采后与 Go logits 最大差 %.3e（loc4 参考均值 %.4f，conf4 均值 %.4f）",
			e.Name, diff, meanOf(refLoc4), meanOf(refConf4))
		_ = at
	}
}

// opsResize 是把 1/4 logits 上采到原分辨率的薄封装（保持测试独立于内部实现细节）。
func opsResize(data []float32, c, h, w, outH, outW int) ([]float32, error) {
	t := tensor.NewFromData(data, c, h, w)
	out, err := ops.ResizeBilinear(t, outH, outW)
	if err != nil {
		return nil, err
	}
	return out.Data, nil
}

func meanOf(v []float32) float64 {
	if len(v) == 0 {
		return 0
	}
	var s float64
	for _, x := range v {
		s += float64(x)
	}
	return s / float64(len(v))
}

// compareBackbone 在同一张端到端图像上重放主干并对拍指定检查点。
func compareBackbone(t *testing.T, g *testgolden.Manifest, e testgolden.E2ECase, m *Model, rgb *tensor.Tensor) error {
	np, err := m.dncnn.Forward(rgb)
	if err != nil {
		return err
	}
	np3 := tensor.New(3, np.H, np.W)
	plane := np.H * np.W
	for c := 0; c < 3; c++ {
		copy(np3.Data[c*plane:(c+1)*plane], np.Data)
	}
	norm := rgb.Clone()
	means := [3]float32{0.485, 0.456, 0.406}
	stds := [3]float32{0.229, 0.224, 0.225}
	for c := 0; c < 3; c++ {
		d := norm.Data[c*plane : (c+1)*plane]
		for i := range d {
			d[i] = (d[i] - means[c]) / stds[c]
		}
	}
	names := make([]string, 0, len(e.BackboneFiles))
	for name := range e.BackboneFiles {
		names = append(names, name)
	}
	sort.Strings(names)
	_, err = m.backbone.ForwardTrace(norm, np3, func(name string, got *tensor.Tensor) error {
		ref, ok := e.BackboneFiles[name]
		if !ok {
			return nil
		}
		want, err := g.ReadEndToEnd(ref.File, ref.Shape)
		if err != nil {
			return err
		}
		d, at := maxAbsDiff(got.Data, want)
		t.Logf("backbone %-24s maxDiff=%.3e", name, d)
		if d > 1e-3 {
			return fmt.Errorf("主干检查点 %s 最大差 %.3e（第 %d 个元素 go=%v ref=%v）", name, d, at, got.Data[at], want[at])
		}
		return nil
	})
	return err
}

// TestDecoderFromGoldenFeatures256 用 256² 端到端图像的主干黄金特征驱动 Go 解码头，
// 与黄金 loc4 对比，判定误差在解码头还是主干。
func TestDecoderFromGoldenFeatures256(t *testing.T) {
	if testing.Short() {
		t.Skip("-short 模式跳过")
	}
	ckpt := checkpointPath()
	goldenDir := testgolden.DefaultDir()
	if ckpt == "" || testgolden.SkipIfMissing(goldenDir) {
		t.Skip("缺少官方权重或黄金数据，跳过")
	}
	g, err := testgolden.Load(goldenDir)
	if err != nil {
		t.Fatalf("加载黄金数据: %v", err)
	}
	w, err := weights.Open(ckpt)
	if err != nil {
		t.Fatalf("加载权重: %v", err)
	}
	for _, e := range g.EndToEnd {
		if len(e.BackboneFiles) == 0 || e.Loc4File == "" {
			continue
		}
		var feats [4]*tensor.Tensor
		for i := 1; i <= 4; i++ {
			name := fmt.Sprintf("stage%d.ffm", i)
			ref, ok := e.BackboneFiles[name]
			if !ok {
				t.Fatalf("缺少 %s", name)
			}
			data, err := g.ReadEndToEnd(ref.File, ref.Shape)
			if err != nil {
				t.Fatalf("读取 %s: %v", name, err)
			}
			feats[i-1] = tensor.NewFromData(data, ref.Shape[0], ref.Shape[1], ref.Shape[2])
		}
		head, err := decoder.New(w, "decode_head", decoderInChannels, numClasses, embedDim, locHeadBNEps, 4)
		if err != nil {
			t.Fatalf("构造定位头: %v", err)
		}
		got, err := head.Forward(feats)
		if err != nil {
			t.Fatalf("定位头前向: %v", err)
		}
		ref, err := g.ReadEndToEnd(e.Loc4File, []int{got.C, got.H, got.W})
		if err != nil {
			t.Fatalf("读取 loc4: %v", err)
		}
		d, at := maxAbsDiff(got.Data, ref)
		t.Logf("%s: 解码(黄金特征) vs loc4 maxDiff=%.3e（第 %d 个元素 go=%v ref=%v）", e.Name, d, at, got.Data[at], ref[at])
		if d > 1e-2 {
			t.Fatalf("解码头在 256² 下不一致: %.3e", d)
		}
	}
}
