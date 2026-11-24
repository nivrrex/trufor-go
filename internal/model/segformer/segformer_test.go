package segformer

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/nivrrex/trufor-go/internal/model/decoder"
	"github.com/nivrrex/trufor-go/internal/model/dncnn"
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

// tol 返回逐层相对容差（相对该层最大绝对值），可用 TRUFOR_TOL 覆盖（排错用）。
func tol() float64 {
	if v := os.Getenv("TRUFOR_TOL"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return 5e-4
}

func loadWeights(t *testing.T) *weights.File {
	t.Helper()
	ckpt := checkpointPath()
	if ckpt == "" {
		t.Skip("缺少官方权重（ref_weights/weights/trufor.pth.tar 或 $TRUFOR_WEIGHTS），跳过")
	}
	f, err := weights.OpenWithFilter(ckpt, func(name string) bool {
		for _, p := range []string{"backbone.", "dncnn.", "decode_head.", "decode_head_conf.", "detection."} {
			if len(name) >= len(p) && name[:len(p)] == p {
				return true
			}
		}
		return false
	})
	if err != nil {
		t.Fatalf("加载权重: %v", err)
	}
	return f
}

// TestBackboneMatchesPyTorch 以 PyTorch 参考实现的 backbone 逐检查点快照为基准，
// 校验双支 SegFormer + FRM/FFM 的每一级、每个 Block 的输出。
func TestBackboneMatchesPyTorch(t *testing.T) {
	goldenDir := testgolden.DefaultDir()
	if testgolden.SkipIfMissing(goldenDir) {
		t.Skip("缺少 testdata/golden，跳过（先运行 tools/dump_golden.py）")
	}
	g, err := testgolden.Load(goldenDir)
	if err != nil {
		t.Fatalf("加载黄金数据: %v", err)
	}
	if len(g.BackboneLayers) == 0 {
		t.Skip("黄金数据中没有 backbone 快照")
	}

	w := loadWeights(t)
	dnet, err := dncnn.New(w, 1)
	if err != nil {
		t.Fatalf("构造 DnCNN: %v", err)
	}
	backbone, err := New(w, "backbone", 1)
	if err != nil {
		t.Fatalf("构造 Backbone: %v", err)
	}

	want := map[string][]float32{}
	wantShape := map[string][]int{}
	var rgb01 []float32
	var rgbShape []int
	for _, l := range g.BackboneLayers {
		data := mustRead(t, g, l)
		if l.Name == "input.rgb01" {
			rgb01, rgbShape = data, l.Shape
			continue
		}
		want[l.Name] = data
		wantShape[l.Name] = l.Shape
	}
	if rgb01 == nil {
		t.Fatal("黄金数据中没有 input.rgb01")
	}
	if len(rgbShape) != 3 || rgbShape[0] != 3 {
		t.Fatalf("input.rgb01 形状异常 %v", rgbShape)
	}

	// 1) Noiseprint++ 一致性（同时验证 DnCNN 与本测试的拼装方式）
	rgbTensor := tensor.NewFromData(append([]float32(nil), rgb01...), 3, rgbShape[1], rgbShape[2])
	modal1, err := dnet.Forward(rgbTensor)
	if err != nil {
		t.Fatalf("DnCNN 前向: %v", err)
	}

	// 2) ImageNet 归一化（上游 preprc_imagenet_torch）
	normalized := rgbTensor.Clone()
	for c, m := range []float32{0.485, 0.456, 0.406} {
		s := []float32{0.229, 0.224, 0.225}[c]
		plane := normalized.H * normalized.W
		d := normalized.Data[c*plane : (c+1)*plane]
		for i := range d {
			d[i] = (d[i] - m) / s
		}
	}

	// 3) NP++ 广播成 3 通道（torch.tile(x,(3,1,1))）
	np3 := tensor.New(3, modal1.H, modal1.W)
	plane := modal1.H * modal1.W
	for c := 0; c < 3; c++ {
		copy(np3.Data[c*plane:(c+1)*plane], modal1.Data)
	}

	// golden 中的 input.modal_x 是广播后的 3 通道版本
	if ref, ok := want["input.modal_x"]; ok {
		checkLayer(t, "input.modal_x", np3.Data, ref)
	}

	seen := 0
	_, err = backbone.ForwardTrace(normalized, np3, func(name string, got *tensor.Tensor) error {
		ref, ok := want[name]
		if !ok {
			t.Fatalf("黄金数据中缺少检查点 %s", name)
		}
		if sh := wantShape[name]; len(sh) != 3 || got.C != sh[0] || got.H != sh[1] || got.W != sh[2] {
			t.Fatalf("%s 形状 (%d,%d,%d) 与黄金 %v 不符", name, got.C, got.H, got.W, sh)
		}
		checkLayer(t, name, got.Data, ref)
		seen++
		return nil
	})
	if err != nil {
		t.Fatalf("前向失败: %v", err)
	}
	expected := 0
	for name := range want {
		if len(name) > 5 && name[:5] == "stage" {
			expected++
		}
	}
	if seen != expected {
		t.Fatalf("对拍了 %d 个检查点，黄金有 %d 个", seen, expected)
	}
}

func mustRead(t *testing.T, g *testgolden.Manifest, l testgolden.LayerRef) []float32 {
	t.Helper()
	v, err := g.ReadBackbone(l.File, l.Shape)
	if err != nil {
		t.Fatalf("读取 %s: %v", l.Name, err)
	}
	return v
}

func checkLayer(t *testing.T, name string, got, want []float32) {
	t.Helper()
	if len(got) != len(want) {
		t.Fatalf("%s 长度 %d != %d", name, len(got), len(want))
	}
	var maxAbs, maxWant float64
	var at int
	for i := range want {
		d := math.Abs(float64(got[i]) - float64(want[i]))
		a := math.Abs(float64(want[i]))
		if a > maxWant {
			maxWant = a
		}
		if d > maxAbs {
			maxAbs, at = d, i
		}
	}
	scale := maxWant
	if scale < 1 {
		scale = 1
	}
	t.Logf("%-28s norm=%.3e (maxAbs=%.3e layerMax=%.3e)", name, maxAbs/scale, maxAbs, maxWant)
	if maxAbs/scale > tol() {
		t.Fatalf("%s 归一化误差 %.3e 超过 %.1e（layerMax=%.3e，第 %d 个元素 got=%v want=%v）",
			name, maxAbs/scale, tol(), maxWant, at, got[at], want[at])
	}
}

// TestInputValidation 校验输入通道与尺寸约束。
func TestInputValidation(t *testing.T) {
	w := loadWeights(t)
	b, err := New(w, "backbone", 1)
	if err != nil {
		t.Fatalf("构造 Backbone: %v", err)
	}
	if _, err := b.Forward(nil, tensor.New(3, 8, 8)); err == nil {
		t.Fatal("nil 输入应报错")
	}
	if _, err := b.Forward(tensor.New(1, 8, 8), tensor.New(3, 8, 8)); err == nil {
		t.Fatal("通道数不是 3 应报错")
	}
	if _, err := b.Forward(tensor.New(3, 8, 8), tensor.New(1, 8, 8)); err == nil {
		t.Fatal("NP++ 通道数不是 3 应报错")
	}
	if _, err := b.Forward(tensor.New(3, 8, 8), tensor.New(3, 16, 16)); err == nil {
		t.Fatal("两支尺寸不一致应报错")
	}
	if _, err := b.ForwardTrace(tensor.New(3, 8, 8), tensor.New(3, 8, 8), nil); err == nil {
		t.Fatal("nil trace 应报错")
	}
}

// TestDecoderOnGoldenFeatures 用黄金数据里的主干输出直接驱动解码头，
// 把"解码头/后处理"与"主干"的误差分开。
func TestDecoderOnGoldenFeatures(t *testing.T) {
	goldenDir := testgolden.DefaultDir()
	if testgolden.SkipIfMissing(goldenDir) {
		t.Skip("缺少 testdata/golden，跳过")
	}
	g, err := testgolden.Load(goldenDir)
	if err != nil {
		t.Fatalf("加载黄金数据: %v", err)
	}
	var feats [4]*tensor.Tensor
	var loc4, conf4 []float32
	var locShape, confShape []int
	for _, l := range g.BackboneLayers {
		switch {
		case l.Name == "logits.loc4":
			loc4, locShape = mustRead(t, g, l), l.Shape
		case l.Name == "logits.conf4":
			conf4, confShape = mustRead(t, g, l), l.Shape
		}
		for i := 1; i <= 4; i++ {
			if l.Name == fmt.Sprintf("stage%d.ffm", i) {
				feats[i-1] = tensor.NewFromData(mustRead(t, g, l), l.Shape[0], l.Shape[1], l.Shape[2])
			}
		}
	}
	if loc4 == nil || feats[0] == nil {
		t.Skip("黄金数据缺少 decode_head 基准")
	}
	w := loadWeights(t)
	loc, err := decoder.New(w, "decode_head", embedDims, 2, 512, 1e-3, 1)
	if err != nil {
		t.Fatalf("构造定位头: %v", err)
	}
	conf, err := decoder.New(w, "decode_head_conf", embedDims, 1, 512, 1e-5, 1)
	if err != nil {
		t.Fatalf("构造置信头: %v", err)
	}
	gotLoc, err := loc.Forward(feats)
	if err != nil {
		t.Fatalf("定位头前向: %v", err)
	}
	gotConf, err := conf.Forward(feats)
	if err != nil {
		t.Fatalf("置信头前向: %v", err)
	}
	if gotLoc.C != locShape[0] || gotLoc.H != locShape[1] || gotLoc.W != locShape[2] {
		t.Fatalf("定位头输出形状 (%d,%d,%d) != %v", gotLoc.C, gotLoc.H, gotLoc.W, locShape)
	}
	checkLayer(t, "decode_head.loc4", gotLoc.Data, loc4)
	if confShape[0] == gotConf.C && confShape[1] == gotConf.H {
		checkLayer(t, "decode_head_conf.conf4", gotConf.Data, conf4)
	}
}
