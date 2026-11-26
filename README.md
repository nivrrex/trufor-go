# trufor-go

TruFor 的 **Go 语言 CPU 推理实现**：[TruFor](https://github.com/grip-unina/TruFor)（CVPR 2023）
用于图像篡改检测与定位，输入一张 RGB 图像，输出三样东西：

- **定位图 `map`**：逐像素的伪造概率
- **置信图 `conf`**：逐像素的可靠性估计（TCP），用于压低误报
- **整图分数 `score`**：0~1，越大越可能被篡改

附带 **Noiseprint++**（学习到的相机/编辑噪声指纹），便于人工核查为什么被判为伪造。

纯 Go 实现，不依赖 PyTorch/CGO，直接读取官方 `.pth.tar` 权重；amd64 上带有手写 AVX2+FMA
内核，其它平台自动回退到纯 Go（结果相同，速度较慢）。

可选 `trufor_ort` 构建通过 CGO 动态加载 ONNX Runtime，直接读取 `trufor.onnx`，
用于与 Python ONNX Runtime 链路保持一致的速度；默认构建不受影响。

> 上游 TruFor 为 GRIP-UNINA 的非营利/信息用途授权软件，本实现沿用同一授权，
> 详见 [LICENSE](LICENSE)。引用/发表结果时请引用原论文（见文末）。

## 结果示例

| 输入 | 定位图 map | 置信图 conf |
|---|---|---|
| 被篡改图像 | 高亮伪造区域 | 高亮预测可靠的区域 |

（图略，运行后得到的是数值图：`map`/`conf` 为 0~1 的 float32 数组，可直接用
matplotlib / Pillow / 任何图像库渲染。）

## 环境要求

- Go 1.24 及以上（构建）
- 推理：**仅 CPU**。x86-64 且支持 AVX2+FMA 时使用手写内核；其它架构自动回退纯 Go
  （正确性一致，速度约慢一个数量级）。运行时不需要 Go 工具链，单文件即可部署。

## 安装

```bash
go install github.com/nivrrex/trufor-go/cmd/trufor@latest
# 或者克隆后本地构建
git clone https://github.com/nivrrex/trufor-go && cd trufor-go && go build ./cmd/trufor
```

## 准备权重

模型权重由原作者分发，**不随本仓库提供**（授权限制），需自行下载：

```bash
wget https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip
# MD5: 7bee48f3476c75616c3c5721ab256ff8
unzip TruFor_weights.zip          # 解压得到 weights 目录，内含 trufor.pth.tar
mkdir -p ref_weights && mv weights ref_weights/
```

权重默认从 `ref_weights` 目录下的 `trufor.pth.tar` 读取，也可以用 `-weights` 参数或
`TRUFOR_WEIGHTS` 环境变量指定任意位置。

## 命令行用法

参数与上游 `trufor_test.py` 保持一致：

```bash
trufor -gpu -1 -in <文件|目录|通配符> -out <目录|文件名> [-save_np]
```

- `-in`：单张图像、目录（递归）、或通配符（如 `'images/*.png'`）
- `-out`：目录（无扩展名）时按输入相对路径镜像输出；带扩展名时作为单个输出文件名。
  已存在的输出文件**默认跳过**，加 `-force` 覆盖
- `-save_np`：额外输出 `np++`
- `-gpu`：仅为与上游参数对齐而保留，本实现只接受 `-1`（CPU）
- 扩展参数：`-threads N`（并行度，默认 CPU 核数）、`-weights PATH`、`-cpuprofile FILE`、
  `-memprofile FILE`

```bash
# 单张：输出为 <out>/<输入文件名>.npz
trufor -gpu -1 -in tampered1.png -out out

# 整个目录（递归），并额外保存 Noiseprint++
trufor -gpu -1 -in images -out out -save_np

# 指定输出文件名
trufor -gpu -1 -in tampered1.png -out result.npz
```

### 输出格式

每个输入对应一个 `.npz`（与 `numpy.savez` 兼容，键名与上游一致）：

| 键 | 形状 | dtype | 含义 |
|---|---|---|---|
| `map` | (H, W) | float32 | 伪造概率（softmax 后） |
| `conf` | (H, W) | float32 | 置信度（sigmoid 后） |
| `score` | () | float64 | 整图分数（sigmoid 后） |
| `imgsize` | (2,) | int64 | (H, W) |
| `np++` | (H, W) | float32 | Noiseprint++，仅 `-save_np` |

读取示例：

```python
import numpy as np
d = np.load("out/tampered1.png.npz")
print(float(d["score"]))          # 0.996523
print(d["map"].shape, d["map"].max())
```

```go
// Go 侧读取（第三方 npz 库均可）；本仓库只写不读
```

## Go 库用法

```go
package main

import (
	"fmt"
	"log"

	trufor "github.com/nivrrex/trufor-go"
)

func main() {
	m, err := trufor.Load(trufor.Config{
		Weights: "trufor.pth.tar", // 官方权重文件路径
		Threads: 4,                // 0 = 使用全部核心
	})
	if err != nil {
		log.Fatal(err)
	}
	defer m.Close()

	res, err := m.DetectFile("tampered1.png")
	if err != nil {
		log.Fatal(err)
	}
	fmt.Printf("score=%.6f size=%dx%d\n", res.Score, res.Width, res.Height)
	// res.Map / res.Conf / res.Noiseprint 为 row-major 的 []float32，长度 Width*Height
}
```

`*trufor.Model` 可并发调用 `DetectFile` / `DetectImage`。`DetectImage` 接受任意
`image.Image`（例如来自 `image.Decode` 或调用方自己的解码器）。

## 可选 ONNX Runtime CPU 后端（linux/amd64）

当 `-weights`/`Config.Weights` 以 `.onnx` 结尾时自动选择 ORT 后端。需要 CGO 与本机
ONNX Runtime 1.29（C API 头文件按上游 v1.29.0 固定，见 `internal/ort/include/SOURCE.md`）：

```bash
CGO_ENABLED=1 go build -tags trufor_ort -o trufor_ort ./cmd/trufor
TRUFOR_ORT_LIBRARY=/path/to/libonnxruntime.so.1 \
  ./trufor_ort -gpu -1 -in image.png -out out -weights /path/to/trufor.onnx
```

- 库查找顺序：`TRUFOR_ORT_LIBRARY` → `libonnxruntime.so.1` → `libonnxruntime.so`。
- ORT 使用固定输入/输出名 `rgb/out/conf/det`；`intra-op` 线程数由 `-threads` 控制，
  `inter-op=1`，图优化等级与 Python 版一致（`ORT_ENABLE_ALL`）。
- ONNX 模型不输出 Noiseprint++，因此 `Result.Noiseprint` 为 nil；map/conf/score 正常返回。
  CLI 在 ONNX 后端遇到 `-save_np` 会明确报错，避免写出空数组。
- 未启用标签、非 linux/amd64 或库不可用时返回明确错误；普通 `CGO_ENABLED=0` 构建仍是
  完全静态的纯 Go 版本。Windows 的 ORT DLL 加载尚未实现。

## 性能

2026-11-26 同机实测（AMD Ryzen 5 2600 6C12T、31GB；固定合成输入；12 线程；
模型加载与预热不计入采样）。复现程序与原始 JSON 见 `benchmarks/`：

| 输入 | 纯 Go 基线 c0f1284 | 纯 Go 优化 | Go + ORT | Python ORT |
|---|---:|---:|---:|---:|
| 1200×900 | 33.15s | **14.33s** | **11.88s** | 12.23s |
| 1536×2048 | 145.87s | **55.52s** | **43.30s** | 44.74s |
| 3000×4000 | 1141.97s | **332.13s** | 未测 | 未测 |

纯 Go 使用 `trufor.pth.tar`，ORT 使用 `trufor.onnx`；Go ORT 与 Python ORT 的 score
最大差约 `1.1e-7`。原生 12MP 内存需求仍高，应先缩小输入或使用 seal-go 的尺寸限制功能。

## 精度

与官方 PyTorch 实现的输出对照（256×256 PNG，同一权重）：

| 项 | 最大偏差 |
|---|---|
| `score` | 4.8e-7 |
| `map` | 1.0e-4 |
| `conf` | 3.9e-5 |
| Noiseprint++ | 3.7e-4 |

差异来自浮点累加顺序不同（参考实现自身 fp32 与 fp64 重算的噪声底为 3.7e-4，本实现处于同一量级）。

## 已知差异与限制

- **JPEG 输入**：Go 标准库的 JPEG 解码与 PIL/libturbojpeg 的 IDCT 与色度上采样不同，
  同一张 JPEG 的 `score` 可能相差 0.04~0.07，`map` 最大差约 0.16。**PNG 输入与 PIL 逐像素一致**。
  需要与 Python 端逐位对齐时，请先把图像转成 PNG 再送进来。
- **仅 CPU**，无 GPU/CUDA 实现。
- **权重不随仓库分发**，需按上文自行下载。
- 输入按上游约定归一化：像素值除以 **256**（不是 255）。
- 大图（>2 MP）会对 Noiseprint++ 做空间分块，峰值内存约 4.7 GB @ 2048×1536。

## 仓库结构

```
cmd/trufor/          命令行入口
trufor.go            对外 Go API（Load / DetectFile / DetectImage / Result）
internal/weights/    官方 .pth.tar 权重读取
internal/tensor/     张量
internal/ops/        算子（卷积/归一化/注意力/上采样…）
internal/kernels/    AVX2+FMA 汇编内核 + 纯 Go 回退
internal/model/      DnCNN(Noiseprint++)、双支 SegFormer、解码头、完整推理链
internal/imageio/    图像解码
internal/npz/        .npz 写出
testdata/golden/     与 PyTorch 参考实现的对拍基准（测试用）
tools/dump_golden.py 生成上述基准的脚本（需要上游源码与 PyTorch）
```

测试：`go test ./...`（缺少权重时相关用例自动跳过；`-short` 跳过端到端推理）。

## 引用

```bibtex
@InProceedings{Guillaro_2023_CVPR,
  author    = {Guillaro, Fabrizio and Cozzolino, Davide and Sud, Avneesh and
               Dufour, Nicholas and Verdoliva, Luisa},
  title     = {TruFor: Leveraging All-Round Clues for Trustworthy Image Forgery
               Detection and Localization},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and
               Pattern Recognition (CVPR)},
  month     = {June},
  year      = {2023},
  pages     = {20606-20615}
}
```

## 授权

本实现是上游 TruFor 的 Go 移植，沿用其授权：**仅限信息用途与非营利用途**，禁止用于
工业或营利活动；再分发/修改需保留原始声明；公开使用结果时需引用原作者。
完整条款见 [LICENSE](LICENSE)，上游仓库：<https://github.com/grip-unina/TruFor>。
