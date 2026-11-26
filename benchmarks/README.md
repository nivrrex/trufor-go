# 同机推理基准

本目录记录 **同一台 AMD Ryzen 5 2600（6C12T）/31GB** 上的可复现结果。Go 1.27.3，
12 线程，固定合成输入，模型加载不计时。基线为 `c0f1284`，优化版为当前分支。

## 复现命令

```bash
# 纯 Go：基线/优化分别构建
go build -o /tmp/tf_bench_base ./tools/bench_inference.go   # 在 c0f1284 worktree
go build -o /tmp/tf_bench_opt ./tools/bench_inference.go

TRUFOR_WEIGHTS=/path/to/trufor.pth.tar
/tf_bench_base -weights "$TRUFOR_WEIGHTS" -width 1200 -height 900 \
  -threads 12 -warmup 1 -repeat 3 -out benchmarks/1mp_baseline.json
/tf_bench_opt -weights "$TRUFOR_WEIGHTS" -width 1200 -height 900 \
  -threads 12 -warmup 1 -repeat 3 -out benchmarks/1mp_optimized.json

# Go ORT 与 Python ORT：同一 ONNX 模型、同一合成输入
CGO_ENABLED=1 go build -tags trufor_ort -o /tmp/tf_bench_ort ./tools/bench_inference.go
TRUFOR_ORT_LIBRARY=/path/to/libonnxruntime.so.1 \
  /tmp/tf_bench_ort -weights /path/to/trufor.onnx -width 1200 -height 900 \
  -threads 12 -warmup 1 -repeat 3 -out benchmarks/1mp_go_ort.json
python3 tools/bench_python.py --weights /path/to/trufor.onnx \
  --width 1200 --height 900 --threads 12 --warmup 1 --repeat 3 \
  --out benchmarks/1mp_python_ort.json
```

3MP 使用 `1536×2048 / warmup=1 / repeat=2`；12MP 使用 `3000×4000 / warmup=0 / repeat=1`。

## 结果摘要

| 输入 | 纯 Go 基线 | 纯 Go 优化 | Go ORT | Python ORT |
|---|---:|---:|---:|---:|
| 1200×900 | 33.15s | **14.33s** | **11.88s** | 12.23s |
| 1536×2048 | 145.87s | **55.52s** | **43.30s** | 44.74s |
| 3000×4000 | 1141.97s | **332.13s** | 未测 | 未测 |

- 纯 Go 使用 `trufor.pth.tar`，ORT 使用 `trufor.onnx`；两者 score/checksum 不要求相同。
- Go ORT 使用 ONNX Runtime 1.29.0，Python ORT 使用 1.23.2，二者不是同版本运行时。
  score 最大差约 `1.1e-7` 仅作为跨版本参考；checksum 差异同时来自运行时版本与后处理精度差异，
  不能据此宣称逐位一致。
- 原始逐次耗时、score、checksum、分配量和堆占用见同目录 JSON。
