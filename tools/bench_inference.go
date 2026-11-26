// Command bench_inference 对已加载模型执行固定输入的 CPU 推理基准。
// 模型加载与预热不计入采样；输出 JSON 包含每次耗时、score、校验和与分配量。
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"hash/fnv"
	"image"
	"math"
	"os"
	"runtime"
	"strings"
	"time"

	trufor "github.com/nivrrex/trufor-go"
)

type sample struct {
	Run          int     `json:"run"`
	Seconds      float64 `json:"seconds"`
	Score        float32 `json:"score"`
	MapChecksum  uint64  `json:"map_checksum"`
	ConfChecksum uint64  `json:"conf_checksum"`
	AllocatedMiB float64 `json:"allocated_mib"`
	HeapMiB      float64 `json:"heap_mib"`
}

type report struct {
	GoVersion string   `json:"go_version"`
	GOOS      string   `json:"goos"`
	GOARCH    string   `json:"goarch"`
	CPUModel  string   `json:"cpu_model"`
	NumCPU    int      `json:"num_cpu"`
	Threads   int      `json:"threads"`
	Width     int      `json:"width"`
	Height    int      `json:"height"`
	Warmup    int      `json:"warmup"`
	Samples   []sample `json:"samples"`
}

func main() {
	weights := flag.String("weights", "", "trufor.pth.tar 或 trufor.onnx")
	width := flag.Int("width", 1200, "固定合成输入宽度")
	height := flag.Int("height", 900, "固定合成输入高度")
	threads := flag.Int("threads", runtime.NumCPU(), "推理线程数")
	warmup := flag.Int("warmup", 1, "不计时的预热次数")
	repeat := flag.Int("repeat", 3, "计时采样次数")
	out := flag.String("out", "", "JSON 输出文件；空则写 stdout")
	flag.Parse()
	if *weights == "" || *width <= 0 || *height <= 0 || *threads <= 0 || *warmup < 0 || *repeat <= 0 {
		fmt.Fprintln(os.Stderr, "参数非法：weights 必填，尺寸/线程/采样必须为正，warmup 不能为负")
		os.Exit(2)
	}

	img := syntheticImage(*width, *height)
	model, err := trufor.Load(trufor.Config{Weights: *weights, Threads: *threads})
	if err != nil {
		fmt.Fprintln(os.Stderr, "加载模型:", err)
		os.Exit(1)
	}
	defer model.Close()
	for i := 0; i < *warmup; i++ {
		if _, err := model.DetectImage(img); err != nil {
			fmt.Fprintln(os.Stderr, "预热推理:", err)
			os.Exit(1)
		}
	}

	rep := report{
		GoVersion: runtime.Version(), GOOS: runtime.GOOS, GOARCH: runtime.GOARCH,
		CPUModel: cpuModel(), NumCPU: runtime.NumCPU(), Threads: *threads,
		Width: *width, Height: *height, Warmup: *warmup,
	}
	for i := 0; i < *repeat; i++ {
		var before, after runtime.MemStats
		runtime.ReadMemStats(&before)
		start := time.Now()
		res, err := model.DetectImage(img)
		elapsed := time.Since(start)
		if err != nil {
			fmt.Fprintln(os.Stderr, "推理:", err)
			os.Exit(1)
		}
		runtime.ReadMemStats(&after)
		rep.Samples = append(rep.Samples, sample{
			Run: i + 1, Seconds: elapsed.Seconds(), Score: res.Score,
			MapChecksum: checksum(res.Map), ConfChecksum: checksum(res.Conf),
			AllocatedMiB: float64(after.TotalAlloc-before.TotalAlloc) / (1 << 20),
			HeapMiB:      float64(after.HeapAlloc) / (1 << 20),
		})
	}
	data, err := json.MarshalIndent(rep, "", "  ")
	if err != nil {
		fmt.Fprintln(os.Stderr, "编码结果:", err)
		os.Exit(1)
	}
	data = append(data, '\n')
	if *out != "" {
		if err := os.WriteFile(*out, data, 0o644); err != nil {
			fmt.Fprintln(os.Stderr, "写结果:", err)
			os.Exit(1)
		}
	}
	fmt.Print(string(data))
}

func syntheticImage(w, h int) image.Image {
	img := image.NewNRGBA(image.Rect(0, 0, w, h))
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			i := img.PixOffset(x, y)
			img.Pix[i+0] = uint8((x*17 + y*31 + 11) & 0xff)
			img.Pix[i+1] = uint8((x*29 + y*7 + 23) & 0xff)
			img.Pix[i+2] = uint8((x*13 + y*19 + 37) & 0xff)
			img.Pix[i+3] = 0xff
		}
	}
	return img
}

func checksum(v []float32) uint64 {
	h := fnv.New64a()
	var buf [4]byte
	for _, x := range v {
		buf[0] = byte(math.Float32bits(x))
		buf[1] = byte(math.Float32bits(x) >> 8)
		buf[2] = byte(math.Float32bits(x) >> 16)
		buf[3] = byte(math.Float32bits(x) >> 24)
		_, _ = h.Write(buf[:])
	}
	return h.Sum64()
}

func cpuModel() string {
	data, err := os.ReadFile("/proc/cpuinfo")
	if err != nil {
		return "unknown"
	}
	for _, line := range strings.Split(string(data), "\n") {
		if strings.HasPrefix(line, "model name") {
			if i := strings.IndexByte(line, ':'); i >= 0 {
				return strings.TrimSpace(line[i+1:])
			}
		}
	}
	return "unknown"
}
