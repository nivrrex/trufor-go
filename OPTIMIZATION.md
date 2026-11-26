# TruFor Go 性能优化记录（2026-11-25）

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

## 2. 可选 ONNX Runtime CPU 后端

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
| 1200×900 | 约 14.1s | 约 14.1s | map 1.73e-6；conf 5.36e-7；score 1.95e-7 |
| 1536×2048 | 约 46.0s | 约 47.8s | 端到端对拍结构性差异 0 |

原生 3000×4000 下 Python ORT 申请约 34.9GB 缓冲并失败；Go ORT 未在该尺寸宣称通过。

## 3. 已证伪/回退

- kernel 内 M 方向并行：wall +7~10%，goroutine 启动与带宽争抢。
- softmax 行并行：wall +7%，高频小调用开销。
- packA 并行：约 24.0s vs 23.7s，无收益。
- ToTokens/FromTokens 并行：A/B 无可复现收益。
- 整段 `clear` 替换：无可测收益。
- 固定 worker 卷积调度：约 22.2s vs 21.0s，回退。
- Winograd 变换布局优化：约 0.58s vs 原 0.54s，回退。
- 6×16 微内核 K 展开 2 次：约 22.2s vs 21.0s，回退。
- CrossAttention M 并行：1MP 变慢，回退。

## 4. 验证

- `go test -count=1 ./...`：默认纯 Go 全通过。
- `CGO_ENABLED=1 go test -race -tags trufor_ort ./internal/ort`：通过。
- ORT 集成测试覆盖正常输出、并发 Run 确定性、Close 后错误、缺失模型错误。
- C shim：`-Wall -Wextra -Wconversion -Wshadow -Werror` 通过；ASan+UBSan 最小推理通过。
- seal-go 默认与 ORT 构建、Windows 默认交叉编译通过。
- 6 图端到端对拍：纯 Go 与 ORT 两条链路均 `compare.py` 退出 0（结构性差异 0）。

## 5. 已知限制

- ORT 后端仅 linux/amd64；Windows 仍使用默认纯 Go 构建。
- ORT 库与 ONNX 权重不随仓库分发；需用户提供。
- 原生 12MP 内存需求仍高，应显式限制 `--trufor-max-size`。
- C API 使用 TruFor 固定张量名，不适用于任意 ONNX 模型。
