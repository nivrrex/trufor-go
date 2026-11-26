# trufor-go

TruFor 的 **Go 语言 CPU 推理实现**：[TruFor](https://github.com/grip-unina/TruFor)（CVPR 2023）
用于图像篡改检测与定位。输入一张 RGB 图像，输出：

- **定位图 `map`**：逐像素的伪造概率
- **置信图 `conf`**：逐像素的可靠性估计，用于压低误报
- **整图分数 `score`**：0~1，越大越可能被篡改
- **Noiseprint++**：可选的噪声指纹输出，便于人工核查

默认实现为纯 Go，不依赖 PyTorch/CGO，可直接读取官方 `.pth.tar` 权重。amd64 上使用
AVX2+FMA 内核；非 AVX2 环境自动回退纯 Go。另提供独立 `trufor_ort` 构建标签，可动态加载
ONNX Runtime CPU 后端。

> 上游 TruFor 为 GRIP-UNINA 的非营利/信息用途授权软件，本实现沿用同一授权，
> 详见 [LICENSE](LICENSE)。引用或发表结果时请引用原论文（见文末）。

## 环境要求

- Go 1.24 及以上（构建）
- 仅 CPU 推理
- x86-64 + AVX2/FMA 时使用手写内核；其它平台使用纯 Go 回退
- 运行时不需要 Go 工具链，默认构建为静态二进制

## 安装

```bash
go install github.com/nivrrex/trufor-go/cmd/trufor@latest

# 或从源码构建
git clone https://github.com/nivrrex/trufor-go
cd trufor-go
go build ./cmd/trufor
```

## 准备权重

模型权重由原作者分发，**不随本仓库提供**：

```bash
wget https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip
# MD5: 7bee48f3476c75616c3c5721ab256ff8
unzip TruFor_weights.zip
mkdir -p ref_weights
mv weights ref_weights/
```

默认从 `ref_weights/weights/trufor.pth.tar` 读取，也可用 `-weights` 参数或
`TRUFOR_WEIGHTS` 环境变量指定。

## 命令行

参数与上游 `trufor_test.py` 保持一致：

```bash
trufor -gpu -1 -in <文件|目录|通配符> -out <目录|文件名> [-save_np]
```

- `-in`：单张图像、目录（递归）或通配符
- `-out`：目录时按输入相对路径镜像输出；带扩展名时作为输出文件名
- 已存在的输出默认跳过，加 `-force` 覆盖
- `-save_np`：额外输出 `np++`
- `-gpu`：仅为兼容上游参数，本实现只接受 `-1`（CPU）
- 扩展参数：`-threads N`、`-weights PATH`、`-cpuprofile FILE`、`-memprofile FILE`

```bash
# 单张图像
trufor -gpu -1 -in tampered1.png -out out

# 目录递归处理并保存 Noiseprint++
trufor -gpu -1 -in images -out out -save_np

# 指定输出文件
trufor -gpu -1 -in tampered1.png -out result.npz
```

### 输出格式

每个输入对应一个 `.npz`，与 `numpy.savez` 兼容：

| 键 | 形状 | dtype | 含义 |
|---|---|---|---|
| `map` | (H, W) | float32 | 伪造概率 |
| `conf` | (H, W) | float32 | 置信度 |
| `score` | () | float64 | 整图分数 |
| `imgsize` | (2,) | int64 | (H, W) |
| `np++` | (H, W) | float32 | Noiseprint++，仅 `-save_np` |

```python
import numpy as np

d = np.load("out/tampered1.png.npz")
print(float(d["score"]))
print(d["map"].shape, d["map"].max())
```

## Go API

```go
package main

import (
	"fmt"
	"log"

	trufor "github.com/nivrrex/trufor-go"
)

func main() {
	m, err := trufor.Load(trufor.Config{
		Weights: "ref_weights/weights/trufor.pth.tar",
		Threads: 4, // 0 = 使用全部核心
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
	// res.Map / res.Conf / res.Noiseprint 为 row-major []float32，长度 Width*Height
}
```

`*trufor.Model` 可并发调用 `DetectFile` / `DetectImage`。`DetectImage` 接受任意
`image.Image`。

## 可选 ONNX Runtime CPU 后端

仅支持 **linux/amd64 + CGO**。权重文件以 `.onnx` 结尾时自动选择该后端：

```bash
CGO_ENABLED=1 go build -tags trufor_ort -o trufor_ort ./cmd/trufor
TRUFOR_ORT_LIBRARY=/path/to/libonnxruntime.so.1 \
  ./trufor_ort -gpu -1 -in image.png -out out -weights /path/to/trufor.onnx
```

- 需要 ONNX Runtime 1.29.0（C API 头文件固定版本）
- 库查找顺序：`TRUFOR_ORT_LIBRARY` → `libonnxruntime.so.1` → `libonnxruntime.so`
- 固定输入/输出名：`rgb/out/conf/det`
- ONNX 模型不输出 Noiseprint++，因此不能与 `-save_np` 同用
- 未启用标签、非 linux/amd64 或库不可用时返回明确错误
- 默认 `CGO_ENABLED=0` 构建不包含 ORT

## 性能

同一台 **AMD Ryzen 5 2600（6C12T）/ 31GB**，Go 1.27.3，12 线程，固定合成输入，
模型加载与预热不计入采样：

| 输入 | 上一版纯 Go | 当前纯 Go | Go + ORT 1.29.0 | Python ORT 1.23.2 |
|---|---:|---:|---:|---:|
| 1200×900 | 33.15 s | **14.33 s** | **11.88 s** | 12.23 s |
| 1536×2048 | 145.87 s | **55.52 s** | **43.30 s** | 44.74 s |
| 3000×4000 | 1141.97 s | **332.13 s** | 未测 | 未测 |

当前纯 Go 相对上一版：1MP 快 **2.31×**，3MP 快 **2.63×**，12MP 快 **3.44×**。
Go ORT 与 Python ORT 使用不同运行时版本，上表仅作同机跨版本参考，不宣称逐位一致。
12MP 内存需求较高，建议先缩小输入或分批处理。

## 精度

与官方 PyTorch 参考实现对照（256×256 PNG，同一权重）：

| 输出 | 最大偏差 |
|---|---:|
| `score` | 4.8e-7 |
| `map` | 1.0e-4 |
| `conf` | 3.9e-5 |
| Noiseprint++ | 3.7e-4 |

差异来自浮点累加顺序不同，处于参考实现自身 fp32 舍入噪声量级。

## 已知差异与限制

- **JPEG 输入**：Go 标准库解码与 PIL/libturbojpeg 的 IDCT、色度上采样不同，
  同一 JPEG 的 `score` 可能相差约 0.04~0.07。**PNG 输入与 PIL 逐像素一致**。
  需要与 Python 端逐位对齐时，请先转为 PNG。
- **仅 CPU**，无 GPU/CUDA 实现。
- **权重不随仓库分发**，需自行下载。
- 输入按上游约定归一化：像素值除以 **256**，不是 255。
- 大图（>2 MP）会对 Noiseprint++ 做空间分块以控制内存；主干含全局注意力，不能同样分块。
- ORT 仅支持 linux/amd64，且不输出 Noiseprint++。

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
工业或营利活动；再分发或修改需保留原始声明；公开使用结果时需引用原作者。
完整条款见 [LICENSE](LICENSE)。
