package weights

import (
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/nivrrex/trufor-go/internal/testgolden"
)

// checkpointPath 返回官方权重路径；不存在时返回空串（测试跳过）。
func checkpointPath() string {
	candidates := []string{
		filepath.Join(testgolden.Root(), "ref_weights", "weights", "trufor.pth.tar"),
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

// TestLoadOfficialCheckpoint 校验 Go 侧 pickle 解析结果与 PyTorch 完全一致：
// 张量名集合、形状、以及若干完整张量的逐元素位级比较。
func TestLoadOfficialCheckpoint(t *testing.T) {
	ckpt := checkpointPath()
	golden := testgolden.DefaultDir()
	if ckpt == "" || testgolden.SkipIfMissing(golden) {
		t.Skip("缺少官方权重或黄金数据，跳过")
	}
	g, err := testgolden.Load(golden)
	if err != nil {
		t.Fatalf("加载黄金数据: %v", err)
	}

	f, err := Open(ckpt)
	if err != nil {
		t.Fatalf("Open(%s): %v", ckpt, err)
	}

	if got, want := len(f.Tensors)+len(f.Skipped), 952; got != want {
		t.Errorf("张量数量 %d != %d", got, want)
	}
	if got, want := len(f.Skipped), 25; got != want {
		t.Errorf("跳过的整型张量 %d != %d", got, want)
	}

	// 官方 checkpoint：952 个张量，其中 25 个 int64（num_batches_tracked）不物化，
	// 浮点参数合计 68705485（PyTorch 侧统计）。
	var total int
	for _, t2 := range f.Tensors {
		total += t2.Len()
	}
	if want := 68705485; total != want {
		t.Errorf("浮点参数总数 %d != %d", total, want)
	}

	for name, ref := range g.Weights {
		tensor2, err := f.Get(name)
		if err != nil {
			t.Errorf("%s: %v", name, err)
			continue
		}
		if !equalInts(tensor2.Shape, ref.Shape) {
			t.Errorf("%s 形状 %v != 黄金 %v", name, tensor2.Shape, ref.Shape)
			continue
		}
		want, err := g.ReadWeights(ref.File, ref.Shape)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if len(want) != len(tensor2.Data) {
			t.Fatalf("%s 长度 %d != %d", name, len(tensor2.Data), len(want))
		}
		var maxDiff float64
		for i := range want {
			if got, exp := math.Float32bits(tensor2.Data[i]), math.Float32bits(want[i]); got != exp {
				d := math.Abs(float64(tensor2.Data[i]) - float64(want[i]))
				if d > maxDiff {
					maxDiff = d
				}
			}
		}
		if maxDiff != 0 {
			t.Errorf("%s 与 PyTorch 不一致，最大差值 %g", name, maxDiff)
		}
	}
}

// TestMeta 校验顶层标量字段解析。
func TestMeta(t *testing.T) {
	ckpt := checkpointPath()
	if ckpt == "" {
		t.Skip("缺少官方权重，跳过")
	}
	f, err := Open(ckpt)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	if got, ok := f.Meta["best_key"]; !ok || got != "loss" {
		t.Errorf("best_key = %v, 期望 \"loss\"", got)
	}
	if _, ok := f.Meta["epoch"]; !ok {
		t.Errorf("缺少 epoch 字段")
	}
}

// TestFilter 验证按名过滤可以减少内存占用且结果一致。
func TestFilter(t *testing.T) {
	ckpt := checkpointPath()
	if ckpt == "" {
		t.Skip("缺少官方权重，跳过")
	}
	f, err := OpenWithFilter(ckpt, func(name string) bool { return name == "dncnn.0.weight" })
	if err != nil {
		t.Fatalf("OpenWithFilter: %v", err)
	}
	if len(f.Tensors) != 1 {
		t.Fatalf("过滤后张量数量 %d != 1", len(f.Tensors))
	}
	only, err := f.Get("dncnn.0.weight")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	all, err := Open(ckpt)
	if err != nil {
		t.Fatalf("Open: %v", err)
	}
	full, err := all.Get("dncnn.0.weight")
	if err != nil {
		t.Fatalf("Get: %v", err)
	}
	for i := range full.Data {
		if only.Data[i] != full.Data[i] {
			t.Fatalf("第 %d 个元素不一致: %v != %v", i, only.Data[i], full.Data[i])
		}
	}
}

func equalInts(a, b []int) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}
