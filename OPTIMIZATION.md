# TruFor Go 性能优化记录（2026-11-25 至 2026-11-26）

基线：`c0f1284`。目标：在保持黄金测试与端到端对拍通过的前提下，缩小 Go/Python CPU 推理差距。

## 1. 纯 Go 路线（默认构建，保留）

### 注意力按 query 块流水并行

原路径为 `QK-GEMM（并行）→ 屏障 → softmax（串行）→ AV-GEMM（n=head_dim≤64，串行）`。
改为每个 worker 独占 scores 缓冲，连续完成 `QK → softmax → AV`，只在 head 级同步一次。
每个输出元素的 K 累加顺序不变；scores 缓冲用 `sync.Pool` 复用，并限制新增缓冲总量。

### 完整 16 列块走微内核

`workerRange` 原来把 32/48 列等完整 16 列块整段退化为标量。现在所有完整 16 列块均走
`micro6x16s`，仅不足 16 列的尾部走标量；3/5 行尾仍走标量。尾块由标量改 FMA 会带来
极小浮点差（1MP score 变化约 1.7e-6），黄金测试通过。

### 串行大数组并行化

- 卷积输出初始化、BN/ReLU 后处理按通道并行。
- Linear bias、LayerNorm token、GELU 连续区间、双线性上采通道并行。
- 小任务保留串行，避免 goroutine 开销。

### decoder 分特征累加

原实现物化 `4×embedDim` 拼接张量，并保留 3 个全分辨率上采结果。改为按上游拼接顺序
`[_c4,_c3,_c2,_c1]` 逆序处理：投影 → 上采 → 直接累加进 fused；4 个固定权重通道块
分别预打包 A，继续走 `SgemmPrepacked`。减少约 9GB 级活跃缓冲，保持 K 方向累加顺序。

### 结果

- 1MP：约 37.1s → 约 24.0s。
- 3MP：约 80s。
- 12MP：约 540s，峰值约 15.5GB（上一轮约 1296s/16.5GB）。

## 2. 第二轮纯 Go 优化（2026-11-26）

### GEMM 访存与分块

- Direct GEMM 新增 row-major A 微内核，完整 6 行块不再做 `packAInto`；仅尾行打包。
- `kc=1024`、`nc=16`、`panelWidth=32`；`SgemmGather` 每个 worker 一次处理连续列区间，
  避免按 16 列块重复分配 B 面板。

### 调度与内存

- 深度卷积 token 路径按通道块并行，使用 AVX2 区间累加；每像素 tap 顺序不变。
- transformer block 池化 LayerNorm/FFN 中间缓冲。
- 每个 stage 的 RGB/Noiseprint 两支并行，各分一半 worker；trace 路径保持串行。
- 定位头/置信头并行，各自使用独立工作区与一半 worker。
- `attnChunk=64`，增加注意力可并行任务数并降低 scores 内存。

### 结果

同机固定输入 A/B（AMD Ryzen 5 2600、12 线程、模型加载/预热排除；原始 JSON 见 `benchmarks/`）：

- 1MP：`c0f1284` 33.15s → **14.33s**；Python ORT 12.23s。
- 3MP：`c0f1284` 145.87s → **55.52s**；Python ORT 44.74s。
- 12MP：`c0f1284` 1141.97s → **332.13s**。
- 1MP/3MP score 与各自基线一致；6 图端到端对拍结构性差异 0。

### 第二轮回退项

Linear 全切 B 打包、B 面板预打包、3×32 TapPlan、Direct 3×32、注意力/FFM scratch 扩展、
卷积块预算/segment 调优、后处理并行均无稳定收益或显著增加内存，已回退。

## 3. 可选 ONNX Runtime CPU 后端

构建标签：`trufor_ort`，仅 `cgo && linux && amd64` 启用；其它平台编译为明确错误占位。
默认 `CGO_ENABLED=0` 构建不引入 CGO 或运行时依赖。

- 通过 `dlopen` 加载 `TRUFOR_ORT_LIBRARY`、`libonnxruntime.so.1` 或 `libonnxruntime.so`。
- 使用上游 v1.29.0 C API 头文件（MIT，见 `internal/ort/include/SOURCE.md`）。
- 固定 TruFor ONNX 输入/输出名 `rgb/out/conf/det`；`intra-op=threads`、`inter-op=1`、
  `ORT_ENABLE_ALL`，与 Python 版一致。
- `.onnx` 权重自动选择 ORT；`.pth.tar` 继续选择纯 Go。
- ONNX 不输出 Noiseprint++，`Result.Noiseprint=nil`。
- `Session.Run` 可并发；`Close` 等待进行中的 Run 结束。

### 结果

| 输入 | Go + ORT | Python ORT | 数值差 |
|---|---:|---:|---|
| 1200×900 | 约 11.88s | 约 12.23s | score 最大差约 1.1e-7 |
| 1536×2048 | 约 43.30s | 约 44.74s | 端到端对拍结构性差异 0 |

原生 3000×4000 下 Python ORT 申请约 34.9GB 缓冲并失败；Go ORT 未在该尺寸宣称通过。

## 4. 已证伪/回退

- kernel 内 M 方向并行：wall +7~10%，goroutine 启动与带宽争抢。
- softmax 行并行：wall +7%，高频小调用开销。
- packA 并行：约 24.0s vs 23.7s，无收益。
- ToTokens/FromTokens 并行：A/B 无可复现收益。
- 整段 `clear` 替换：无可测收益。
- 固定 worker 卷积调度：约 22.2s vs 21.0s，回退。
- Winograd 变换布局优化：约 0.58s vs 原 0.54s，回退。
- 6×16 微内核 K 展开 2 次：约 22.2s vs 21.0s，回退。
- CrossAttention M 并行：1MP 变慢，回退。

## 审计整改（P0/P1）

- 预打包 GEMM 新增显式 `lda` 版本；旧 API 保留 `lda=k` 包装，decoder 分段权重使用
  `SgemmPrepackedStride` 传入真实行跨度。
- SIMD 检测/声明按 `amd64` 与 `trufor_scalar` 构建标签隔离；非 amd64 提供明确 panic 的
  编译期占位与标量辅助函数，正常路径由 `simdAvailable=false` 进入纯 Go。
- `TestPrepackedStrideFallback` 覆盖 `lda≠k`；`trufor_scalar` 全量 golden 通过。
- `TestDirectVsGather` 对逐位不一致执行失败；`TestDecoderParity` 对 PNG 执行 `5e-4`
  阈值断言，JPEG 明确跳过独立 decoder 对拍并交由 E2E JPEG 容差覆盖。
- ORT `-save_np` 明确拒绝；修正文档中不存在的参数与 decoder 工作区注释。
- 新增 `tools/bench_inference.go`、`tools/bench_python.py` 和 `benchmarks/` 原始 JSON。

## 5. 验证

- 默认 `go test -count=1 ./...`：全通过。
- `go test -count=1 -tags trufor_scalar ./...`：强制非 AVX2 回退全通过。
- Linux arm64 交叉编译与 vet：通过。
- 纯 Go 并发包 `CGO_ENABLED=1 go test -race`（kernels/ops/segformer/trufor）：通过。
- `CGO_ENABLED=1 go test -race -tags trufor_ort ./internal/ort`：通过。
- ORT 集成测试覆盖正常输出、并发 Run 确定性、Close 后错误、缺失模型错误。
- C shim：`-Wall -Wextra -Wconversion -Wshadow -Werror` 通过；ASan+UBSan 最小推理通过。
- seal-go 默认与 ORT 构建、Windows 默认交叉编译通过。
- 6 图端到端对拍：纯 Go 与 ORT 两条链路均 `compare.py` 退出 0（结构性差异 0）。

## 6. 已知限制

- ORT 后端仅 linux/amd64；Windows 仍使用默认纯 Go 构建。
- ORT 库与 ONNX 权重不随仓库分发；需用户提供。
- 原生 12MP 内存需求仍高，应先缩小输入或使用 seal-go 的尺寸限制功能。
- C API 使用 TruFor 固定张量名，不适用于任意 ONNX 模型。
