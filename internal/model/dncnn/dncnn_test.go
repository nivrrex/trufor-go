package dncnn

import (
	"fmt"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"testing"

	"github.com/nivrrex/trufor-go/internal/tensor"
	"github.com/nivrrex/trufor-go/internal/testgolden"
	"github.com/nivrrex/trufor-go/internal/weights"
)

func checkpointPath() string {
	root := testgolden.Root()
	candidates := []string{
		filepath.Join(root, "ref_weights", "weights", "trufor.pth.tar"),
	}
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

// loadModel 读取官方权重并构造 DnCNN。
func loadModel(t *testing.T, workers int) *Model {
	t.Helper()
	ckpt := checkpointPath()
	if ckpt == "" {
		t.Skip("缺少官方权重（ref_weights/weights/trufor.pth.tar 或 $TRUFOR_WEIGHTS），跳过")
	}
	f, err := weights.OpenWithFilter(ckpt, func(name string) bool {
		return len(name) > 6 && name[:6] == "dncnn."
	})
	if err != nil {
		t.Fatalf("加载权重: %v", err)
	}
	m, err := New(f, workers)
	if err != nil {
		t.Fatalf("构造 DnCNN: %v", err)
	}
	return m
}

// TestForwardMatchesPyTorch 是核心对拍测试：以 PyTorch 参考实现逐层导出的快照为基准，
// 校验 Go 实现每一层（conv / bn / relu）的输出以及最终 Noiseprint++。
func TestForwardMatchesPyTorch(t *testing.T) {
	golden := testgolden.DefaultDir()
	if testgolden.SkipIfMissing(golden) {
		t.Skip("缺少 testdata/golden，跳过（先运行 tools/dump_golden.py）")
	}
	g, err := testgolden.Load(golden)
	if err != nil {
		t.Fatalf("加载黄金数据: %v", err)
	}

	m := loadModel(t, 1)

	var input []float32
	var inputShape []int
	for _, l := range g.DncnnLayers {
		if l.Name == "input" {
			inputShape = l.Shape
			input = mustRead(t, g, l)
		}
	}
	if input == nil {
		t.Fatal("黄金数据中没有 input 层")
	}
	if len(inputShape) != 3 {
		t.Fatalf("input 形状 %v 不是 3 维", inputShape)
	}
	x := tensor.NewFromData(input, inputShape[0], inputShape[1], inputShape[2])

	want := map[string][]float32{}
	wantShape := map[string][]int{}
	for _, l := range g.DncnnLayers {
		if l.Name == "input" {
			continue
		}
		want[l.Name] = mustRead(t, g, l)
		wantShape[l.Name] = l.Shape
	}

	seen := 0
	_, err = m.ForwardTrace(x, func(name string, got *tensor.Tensor) error {
		w, ok := want[name]
		if !ok {
			t.Fatalf("黄金数据中缺少层 %s", name)
		}
		if sh := wantShape[name]; len(sh) != 3 || got.C != sh[0] || got.H != sh[1] || got.W != sh[2] {
			t.Fatalf("层 %s 形状 (%d,%d,%d) 与黄金 %v 不符", name, got.C, got.H, got.W, sh)
		}
		maxAbs, maxRel, at := diff(got.Data, w)
		// 容差按层的量级归一。依据（已实测）：同一模型用 PyTorch fp64 重算，
		// 最终 Noiseprint++ 与 PyTorch fp32 结果的差异为 maxAbs=3.7e-4、rel=2.3e-4，
		// 即参考实现自身的 fp32 舍入噪声就在这个量级；Go 侧与 fp32 参考的最大差异
		// 为 2.7e-4（rel 1.7e-4），小于该噪声底，说明实现精度与参考等价。
		// 真正的实现错误（eps、padding、通道顺序、缺少激活）会带来 1e-2 以上偏差，
		// 因此 5e-4 的相对阈值仍保留足够判别力。
		layerMax := maxAbsOf(w)
		scale := layerMax
		if scale < 1 {
			scale = 1
		}
		t.Logf("%-22s maxAbs=%.3e maxRel=%.3e norm=%.3e", name, maxAbs, maxRel, maxAbs/scale)
		if maxAbs/scale > layerTol() {
			t.Fatalf("层 %s 归一化误差 %.3e 超过 %.0e（layerMax=%.3e，第 %d 个元素 got=%v want=%v）",
				name, maxAbs/scale, layerTol(), layerMax, at, got.Data[at], w[at])
		}
		seen++
		return nil
	})
	if err != nil {
		t.Fatalf("前向失败: %v", err)
	}
	if seen != len(want) {
		t.Fatalf("对拍了 %d 层，黄金有 %d 层", seen, len(want))
	}
}

// TestWorkersDeterminism 校验并行度不影响输出（卷积按行分块，累加顺序固定）。
func TestWorkersDeterminism(t *testing.T) {
	golden := testgolden.DefaultDir()
	if testgolden.SkipIfMissing(golden) {
		t.Skip("缺少 testdata/golden，跳过")
	}
	g, err := testgolden.Load(golden)
	if err != nil {
		t.Fatalf("加载黄金数据: %v", err)
	}
	var input []float32
	var shape []int
	for _, l := range g.DncnnLayers {
		if l.Name == "input" {
			input, shape = mustRead(t, g, l), l.Shape
		}
	}
	if input == nil {
		t.Fatal("黄金数据中没有 input 层")
	}

	run := func(workers int) []float32 {
		m := loadModel(t, workers)
		x := tensor.NewFromData(append([]float32(nil), input...), shape[0], shape[1], shape[2])
		out, err := m.Forward(x)
		if err != nil {
			t.Fatalf("前向失败: %v", err)
		}
		return out.Data
	}
	base := run(1)
	for _, workers := range []int{2, 4} {
		got := run(workers)
		for i := range base {
			if math.Float32bits(base[i]) != math.Float32bits(got[i]) {
				t.Fatalf("workers=%d 与串行结果不一致（第 %d 个元素 %v != %v）", workers, i, got[i], base[i])
			}
		}
	}
}

// TestOutputShape 覆盖奇数尺寸输入（上游不做 32 的倍数对齐）。
func TestOutputShape(t *testing.T) {
	m := loadModel(t, 1)
	for _, tc := range []struct{ h, w int }{{1, 1}, {3, 5}, {33, 17}, {64, 48}} {
		x := tensor.New(3, tc.h, tc.w)
		out, err := m.Forward(x)
		if err != nil {
			t.Fatalf("%dx%d: %v", tc.h, tc.w, err)
		}
		if out.C != 1 || out.H != tc.h || out.W != tc.w {
			t.Fatalf("%dx%d: 输出形状 (%d,%d,%d)", tc.h, tc.w, out.C, out.H, out.W)
		}
	}
}

// TestInputValidation 校验输入通道校验生效。
func TestInputValidation(t *testing.T) {
	m := loadModel(t, 1)
	if _, err := m.Forward(tensor.New(1, 8, 8)); err == nil {
		t.Fatal("输入通道不是 3 时应报错")
	}
	if _, err := m.Forward(nil); err == nil {
		t.Fatal("nil 输入应报错")
	}
	if _, err := m.ForwardTrace(tensor.New(3, 8, 8), nil); err == nil {
		t.Fatal("nil trace 应报错")
	}
}

func mustRead(t *testing.T, g *testgolden.Manifest, l testgolden.LayerRef) []float32 {
	t.Helper()
	v, err := g.ReadDncnn(l.File, l.Shape)
	if err != nil {
		t.Fatalf("读取层 %s: %v", l.Name, err)
	}
	return v
}

// layerTol 返回逐层相对容差（相对该层最大绝对值），可用 TRUFOR_TOL 覆盖（仅用于排查）。
func layerTol() float64 {
	if v := os.Getenv("TRUFOR_TOL"); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			return f
		}
	}
	return 5e-4
}

func maxAbsOf(v []float32) float64 {
	var m float64
	for _, x := range v {
		if a := math.Abs(float64(x)); a > m {
			m = a
		}
	}
	return m
}

func diff(got, want []float32) (maxAbs, maxRel float64, at int) {
	for i := range want {
		d := math.Abs(float64(got[i]) - float64(want[i]))
		scale := math.Abs(float64(want[i]))
		if scale < 1 {
			scale = 1
		}
		if d > maxAbs {
			maxAbs, at = d, i
		}
		if r := d / scale; r > maxRel {
			maxRel = r
		}
	}
	return maxAbs, maxRel, at
}

// BenchmarkForward 给出纯 Go（无汇编）实现的性能基线，便于与 PyTorch CPU 对比。
// PyTorch 参考（i5-7600 4 线程）: 256²≈0.39s，512²≈1.40s（仅 DnCNN）。
func BenchmarkForward(b *testing.B) {
	ckpt := checkpointPath()
	if ckpt == "" {
		b.Skip("缺少官方权重，跳过")
	}
	f, err := weights.OpenWithFilter(ckpt, func(name string) bool {
		return len(name) > 6 && name[:6] == "dncnn."
	})
	if err != nil {
		b.Fatalf("加载权重: %v", err)
	}
	for _, size := range []int{256, 512} {
		for _, workers := range []int{1, 4} {
			m, err := New(f, workers)
			if err != nil {
				b.Fatalf("构造: %v", err)
			}
			x := tensor.New(3, size, size)
			for i := range x.Data {
				x.Data[i] = float32(i%251) / 251
			}
			b.Run(fmt.Sprintf("%dx%d_w%d", size, size, workers), func(b *testing.B) {
				b.ReportAllocs()
				for i := 0; i < b.N; i++ {
					if _, err := m.Forward(x); err != nil {
						b.Fatal(err)
					}
				}
			})
		}
	}
}

// TestTiledMatchesFull 校验空间分块的结果与整图计算逐位相同
// （halo 覆盖 17 层 3×3 卷积的感受野，因此每个输出像素的取值与累加顺序都一致）。
func TestTiledMatchesFull(t *testing.T) {
	m := loadModel(t, 4)
	x := tensor.New(3, 96, 128)
	for i := range x.Data {
		x.Data[i] = float32(i%251) / 251
	}
	full, err := m.forwardFull(x, nil)
	if err != nil {
		t.Fatalf("整图前向: %v", err)
	}
	for _, tileH := range []int{16, 32, 64} {
		tiled, err := m.forwardTiled(x, tileH)
		if err != nil {
			t.Fatalf("分块前向(tileH=%d): %v", tileH, err)
		}
		for i := range full.Data {
			if math.Float32bits(full.Data[i]) != math.Float32bits(tiled.Data[i]) {
				t.Fatalf("tileH=%d: 第 %d 个元素不一致 full=%v tiled=%v", tileH, i, full.Data[i], tiled.Data[i])
			}
		}
	}
}
