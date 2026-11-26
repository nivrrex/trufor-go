// Command trufor 对图像执行 TruFor 篡改检测与定位推理（CPU）。
//
// 参数与上游 PyTorch 实现（test_docker/src/trufor_test.py）保持一致：
//
//	trufor -in <文件|目录|通配符> -out <目录|文件名> [-save_np] [-gpu -1]
//
// 默认每张图像输出一个 .npz（与 numpy.savez 兼容），含：
//
//	map      (H,W) float32  定位图（伪造类概率）
//	conf     (H,W) float32  置信图（sigmoid 后）
//	score    float64        整图完整性分数
//	imgsize  (2,) int64     (H, W)
//	np++     (H,W) float32  Noiseprint++（仅 -save_np）
//
// 已存在的输出文件默认跳过，不覆盖（与上游一致）。
package main

import (
	"flag"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"runtime/debug"
	"runtime/pprof"
	"strings"

	"github.com/nivrrex/trufor-go"
	"github.com/nivrrex/trufor-go/internal/npz"
)

func main() {
	var (
		gpu     = flag.Int("gpu", 0, "设备号；本实现仅支持 CPU，请传 -1（与上游参数名保持一致）")
		input   = flag.String("in", "../images", "输入：单个文件、目录（递归）或通配符")
		output  = flag.String("out", "../output", "输出目录，或输出文件名（带扩展名时按文件名处理）")
		saveNP  = flag.Bool("save_np", false, "是否保存 Noiseprint++（np++）")
		threads = flag.Int("threads", runtime.NumCPU(), "算子并行度（本实现扩展参数）")
		wp      = flag.String("weights", defaultWeights(), "权重文件 trufor.pth.tar（本实现扩展参数）")
		cpuProf = flag.String("cpuprofile", "", "把 CPU profile 写入该文件（本实现扩展参数）")
		memProf = flag.String("memprofile", "", "推理结束后把堆 profile 写入该文件（本实现扩展参数）")
		force   = flag.Bool("force", false, "覆盖已存在的输出文件（上游默认跳过）")
	)
	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, "用法: trufor -in <文件|目录|通配符> -out <目录|文件名> [选项]\n\n")
		flag.PrintDefaults()
	}
	flag.Parse()

	// 上游把 -save_np 也写作 -save-np；Go 的 flag 只看第一个名字，这里补一个别名
	if !*saveNP {
		for _, a := range os.Args[1:] {
			if a == "-save-np" || a == "--save-np" {
				*saveNP = true
			}
		}
	}
	// 上游末尾的 opts 是 yacs 配置覆盖，本实现没有对应配置项，显式提示而不是静默忽略
	if args := flag.Args(); len(args) > 0 {
		fmt.Fprintf(os.Stderr, "提示: 忽略多余参数 %v（上游用于配置覆盖，本实现无对应配置项）\n", args)
	}
	if *gpu != -1 {
		fmt.Fprintf(os.Stderr, "错误: 本实现仅支持 CPU，请使用 -gpu -1（当前 -gpu %d）\n", *gpu)
		os.Exit(2)
	}

	if *cpuProf != "" {
		f, err := os.Create(*cpuProf)
		if err != nil {
			fmt.Fprintf(os.Stderr, "错误: %v\n", err)
			os.Exit(1)
		}
		if err := pprof.StartCPUProfile(f); err != nil {
			fmt.Fprintf(os.Stderr, "错误: %v\n", err)
			os.Exit(1)
		}
		defer func() {
			pprof.StopCPUProfile()
			f.Close()
		}()
	}

	// 推理以大块张量为主，默认 GOGC 会让堆膨胀到活跃数据的 2 倍以上；收紧以降低峰值内存。
	// 显式设置 GOGC 环境变量时以环境为准（便于按机器内存调参），否则用保守默认 50。
	if os.Getenv("GOGC") == "" {
		debug.SetGCPercent(50)
	}

	if err := run(*input, *output, *wp, *threads, *saveNP, *force); err != nil {
		fmt.Fprintf(os.Stderr, "错误: %v\n", err)
		os.Exit(1)
	}

	if *memProf != "" {
		f, err := os.Create(*memProf)
		if err != nil {
			fmt.Fprintf(os.Stderr, "错误: %v\n", err)
			os.Exit(1)
		}
		defer f.Close()
		runtime.GC() // 先回收，heap profile 反映存活对象（分配热点另见 alloc_space）
		if err := pprof.Lookup("heap").WriteTo(f, 0); err != nil {
			fmt.Fprintf(os.Stderr, "错误: %v\n", err)
			os.Exit(1)
		}
	}
}

func defaultWeights() string {
	if v := os.Getenv("TRUFOR_WEIGHTS"); v != "" {
		return v
	}
	return filepath.Join("ref_weights", "weights", "trufor.pth.tar")
}

func run(input, output, weightsPath string, threads int, saveNP, force bool) error {
	list, err := expandInput(input)
	if err != nil {
		return err
	}
	fmt.Printf("=> loading model from %s\n", weightsPath)
	model, err := trufor.Load(trufor.Config{Weights: weightsPath, Threads: threads})
	if err != nil {
		return err
	}
	defer model.Close()
	if saveNP && !model.NoiseprintSupported() {
		return fmt.Errorf("当前 ONNX 后端不输出 Noiseprint++，不能使用 -save_np；请改用 trufor.pth.tar 纯 Go 后端")
	}

	for _, path := range list {
		outPath := outputPath(input, output, path)
		if !force {
			if _, err := os.Stat(outPath); err == nil {
				fmt.Printf("%s\t已存在，跳过 (%s)\n", path, outPath)
				continue
			}
		}
		res, err := model.DetectFile(path)
		if err != nil {
			// 与上游一致：单张图像失败不影响其它图像
			fmt.Fprintf(os.Stderr, "%s: %v\n", path, err)
			continue
		}
		if err := writeNPZ(outPath, res, saveNP); err != nil {
			fmt.Fprintf(os.Stderr, "%s: %v\n", path, err)
			continue
		}
		fmt.Printf("%s\tscore=%.6f\n", path, res.Score)
		debug.FreeOSMemory()
	}
	return nil
}

// writeNPZ 按上游 np.savez 的键名与 dtype 写出结果。
func writeNPZ(path string, res *trufor.Result, saveNP bool) error {
	if dir := filepath.Dir(path); dir != "" {
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return err
		}
	}
	arrays := map[string]npz.Array{
		"map":     npz.F32([]int{res.Height, res.Width}, res.Map),
		"conf":    npz.F32([]int{res.Height, res.Width}, res.Conf),
		"score":   npz.ScalarF64(float64(res.Score)),
		"imgsize": npz.I64([]int{2}, []int64{int64(res.Height), int64(res.Width)}),
	}
	if saveNP {
		arrays["np++"] = npz.F32([]int{res.Height, res.Width}, res.Noiseprint)
	}
	return npz.Write(path, arrays)
}

// outputPath 复刻上游的输出路径规则：
//   - output 的 basename 无扩展名 → 视为目录，按输入的相对路径镜像，末尾追加 ".npz"
//     （即 <output>/<相对路径>.npz，例如 images/a.png → out/a.png.npz）
//   - output 带扩展名 → 视为单个输出文件名（缺 .npz 时补上）
func outputPath(input, output, path string) string {
	if filepath.Ext(filepath.Base(output)) != "" {
		if !strings.HasSuffix(output, ".npz") {
			return output + ".npz"
		}
		return output
	}
	root := strings.SplitN(input, "*", 2)[0]
	var sub string
	if st, err := os.Stat(input); err == nil && !st.IsDir() {
		sub = strings.TrimPrefix(path, filepath.Dir(root))
	} else {
		sub = strings.TrimPrefix(path, root)
	}
	sub = strings.TrimPrefix(sub, string(filepath.Separator))
	return filepath.Join(output, sub) + ".npz"
}

// expandInput 复刻上游的输入枚举规则：通配符递归匹配、单文件、目录递归；不含扩展名过滤
// （无法解码的文件在逐张处理时报错并跳过，与上游行为一致）。
func expandInput(input string) ([]string, error) {
	if strings.Contains(input, "*") {
		m, err := filepath.Glob(input)
		if err != nil {
			return nil, err
		}
		return filterFiles(m), nil
	}
	st, err := os.Stat(input)
	if err != nil {
		return nil, fmt.Errorf("输入既不是文件也不是目录: %s", input)
	}
	if !st.IsDir() {
		return []string{input}, nil
	}
	var out []string
	err = filepath.Walk(input, func(path string, info os.FileInfo, err error) error {
		if err != nil {
			return err
		}
		if !info.IsDir() {
			out = append(out, path)
		}
		return nil
	})
	if err != nil {
		return nil, err
	}
	return filterFiles(out), nil
}

func filterFiles(paths []string) []string {
	out := make([]string, 0, len(paths))
	for _, p := range paths {
		if st, err := os.Stat(p); err == nil && st.IsDir() {
			continue
		}
		out = append(out, p)
	}
	return out
}
