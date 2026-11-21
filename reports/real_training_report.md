# 冻结条件缓存完整规模训练报告

[材料说明](../README.md) · [A100 实验计划](a100_experiment_plan.md) · [A100 实验报告](a100_training_report.md) · [Qwen A100 实验报告](qwen_training_report.md) · [PR 正文](https://github.com/redai-studio/Relax/pull/571)

本报告记录 2026-11-12（Asia/Shanghai）在官方 Docker 镜像中使用完整规模 SenseNova 模型完成的真实 SFT 训练实验。实验覆盖 R1 在线基线、R3–R5 训练数值、R6–R7 端到端性能及 R8 容量与资源。第二种模型暂时搁置，不在本报告范围内。

## 1. 环境与版本

| 项目 | 内容 |
| --- | --- |
| 机器 | 租用的单台服务器，主机名 `ubuntu22`，Ubuntu 22.04.4，16 核，主机内存 31 GB |
| GPU | 2 × NVIDIA GeForce RTX 4090 48 GB，驱动 550.78，无 NVLink |
| 镜像 | `ghcr.io/redai-studio/relaxrl@sha256:cd431e1094646c347aa97bc9cb8c5ac8315d4e72b0b6321f1cafd1ae12e680c6`（2026-11-12 查询时的 `latest`，构建于 2026-09-16），容器 `relax-task17` |
| 软件 | Python 3.12.3、PyTorch 2.11.0+cu129、torchvision 0.26.0+cu129、Transformers 5.12.1、flash-attn 2.7.4.post1、PEFT 0.20.0、Ray 2.58.0；包版本依据各实验的 `environment.txt`，Megatron 使用镜像内 `/root/Megatron-LM/` |
| 驱动处理 | 镜像声明 `cuda>=12.9`，并在 `/usr/lib/x86_64-linux-gnu` 带有 575.57.08 前向兼容 `libcuda`，前向兼容不支持 GeForce（错误 804）。[start_container.sh](../scripts/start_container.sh) 设置 `NVIDIA_DISABLE_REQUIRE=1`，把指向宿主机 550.78 驱动库的链接置于 `LD_LIBRARY_PATH` 最前面；CUDA 12.9 runtime 通过小版本兼容运行。镜像文件与驱动均未修改 |
| 代码 | HEAD `44c522b4841e0f9a0b37033bf66a3647b2fb9e9b` 加未提交改动，`git diff --binary` 摘要 `0e1ac9520f3b81b053bf06f9b1788900a4de7356a1077b60b6b42656839f0f59`；容器挂载服务器 `/root/task17/Relax` 为 `/root/Relax` |
| 模型 | `sensenova/SenseNova-U1.5-8B-MoT` revision `a2462cc1b549bace2c0896143bbbcbb3fa217c76`，32.7 GiB；从 ModelScope 下载，按 HF 该 revision 的 LFS sha256 与 git blob 摘要逐文件校验通过 |
| 数据 | `osunlp/MagicBrush` revision `6c254560879df7851233ea0fd2ae2a62764d0ef2`；[prepare_magicbrush.py](../scripts/prepare_magicbrush.py) 读取 `train-00000-of-00051-9fd9f23e2b1cb397.parquet`（sha256 `ed417a2c…21cf`），按文件顺序取前 128 行并保留原始图像字节；128 个不同条件，190 张 1024×1024、66 张 500×500 图像；metadata sha256 `3311ec0f682a80447ef1ac0f0f8567062269c944bbe9dcae488fc6b2c882def9` |

容器内 213 项相关测试通过、0 项跳过，见 [PR 正文](https://github.com/redai-studio/Relax/pull/571)的 Testing 部分；测试日志位于原始日志压缩包的 `task17-docker-experiments-20261112/validate-docker-tf5/`。

## 2. 训练配置

所有实验通过官方入口 `scripts/entrypoint/ray-job.sh` 与 [run-sensenova-i2i-sft-fixed-tiers-8xgpu.sh](https://github.com/xBei0223/Relax/blob/60706131f2b01db708dbc3f57c5c9e9f8cd8ffe5/scripts/training/diffusion/run-sensenova-i2i-sft-fixed-tiers-8xgpu.sh) 提交，额外参数为 `--fsdp-load-wave-size 1`（主机内存 31 GB，两个 rank 依次加载模型）。共同配置见 [base.env](../scripts/configs/base.env)：

| 配置 | 值 |
| --- | --- |
| 训练模式 | LoRA，rank 与 alpha 为 32，学习率 1e-4，线性 warmup 200 步，FP32 主权重，BF16 计算，FP32 归约 |
| 并行与批次 | 2 GPU 纯 FSDP2 数据并行，global batch 4，每个 rank 每步 2 个 micro-batch，每个 epoch 32 步 |
| 图像 | `MAX_PIXELS=1048576`，`CONDITIONING_IMAGES=1`，前缀 gradient checkpointing 开启 |
| 已有优化 | FlashAttention 2，图像预取深度 2、2 个 worker |
| 缓存容量 | 每个 rank `condition_cache_max_entries=256`、`condition_cache_max_bytes=17179869184`（16 GiB） |
| 随机种子 | 42 |

| 实验 | 配置 | 用途 |
| --- | --- | --- |
| `parity-off-20261112` / `parity-on-20261112` | [parity-off.env](../scripts/configs/parity-off.env) / [parity-on.env](../scripts/configs/parity-on.env)：确定性 attention，64 步，保存最终 checkpoint | 训练数值对照 |
| `baseline-plain-r{1,2,3}-20261112` | [baseline-plain.env](../scripts/configs/baseline-plain.env)：缓存与追踪器关闭，96 步 | 性能基线 |
| `baseline-tracker-r{1,2,3}-20261112` | [baseline-tracker.env](../scripts/configs/baseline-tracker.env)：缓存关闭，追踪器开启，96 步 | 在线基线统计与追踪器开销 |
| `cache-plain-r{1,2,3}-20261112` | [cache-plain.env](../scripts/configs/cache-plain.env)：缓存开启，追踪器关闭，96 步 | 缓存性能 |
| `cache-tracker-r1-20261112` | [cache-tracker.env](../scripts/configs/cache-tracker.env)：缓存与追踪器开启，96 步 | 缓存运行的阶段明细 |

运行顺序见 [queue-20261112.txt](../scripts/queue-20261112.txt)，缓存开启与关闭的运行交替执行。12 个实验均为 `SUCCEEDED`。

## 3. 训练数值（R3–R5）

两次对照使用相同数据顺序、初始权重、optimizer 状态、种子与确定性 attention，各训练 64 步（2 个 epoch）。缓存开启的运行中每个 rank 128 次请求，本地命中 36 次，组内共同复用 23 次（均在第 2 个 epoch，占该 epoch 请求的 36%），13 次因另一 rank 未命中而在线计算。

| 比较项 | 结果 |
| --- | --- |
| 64 步 `train/loss` | 64 步相同，最大绝对误差 0 |
| 64 步 `train/grad_norm` | 64 步相同，最大绝对误差 0 |
| 64 步 `train/lr` | 64 步相同 |
| 最终 checkpoint（LoRA FP32 参数与 optimizer 状态） | 2352 个张量逐元素相同，最大绝对误差 0，最大相对 L2 误差 0 |
| 各 rank RNG 状态 | 6 项相同 |

比较由 [compare_parity.py](../scripts/compare_parity.py) 完成：读取 TensorBoard 记录的逐步指标，用 `torch.distributed.checkpoint.format_utils.dcp_to_torch_save` 转换 DCP checkpoint 后逐张量比较。结果文件为 [parity-compare-20261112.json](../raw/task17-docker-experiments-20261112/parity-compare-20261112.json)。本实验直接验证逐步 loss、grad_norm、lr 和最终参数、optimizer、RNG 状态；完整规模模型每步逐参数梯度及单步 FP32 参数更新量未单独导出。逐参数梯度与单步 FP32 更新量的直接比较已在 CPU 和缩小维度双 GPU 测试中完成，范围见 [PR 正文](https://github.com/redai-studio/Relax/pull/571)的 Testing 部分。

## 4. 在线基线（R1）

`baseline-tracker-r1-20261112` 的离线汇总（96 步、384 个样本消费位置、预热 1 步）：

| 指标 | 本地（各 rank 之和） | 全局 |
| --- | --- | --- |
| 唯一数据行 | 218 | 128 |
| 重复样本 | 166（43.2%） | 256（66.7%） |
| `ref_vision_encode` / `prefix_encode` 调用 | 各 384 次，其中重复 166 次 | 重复 256 次 |
| `ref_pixel_values` H2D 搬运 | 1413.0 MiB，其中重复 609.0 MiB | 重复 942.0 MiB |
| `prefix_input_ids` H2D 搬运 | 2.6 MiB，其中重复 1.1 MiB | 重复 1.7 MiB |
| `target_input_latents` H2D 搬运 | 2304.0 MiB | — |

| 阶段（每步均值） | 耗时 ms | 占单步 | 本地重复条件耗时 ms |
| --- | --- | --- | --- |
| `ref_image_preprocess` | 311.4 | 3.26% | 133.7 |
| `ref_vision_encode` | 2.4 | 0.03% | 1.0 |
| `prefix_encode` | 3064.6 | 32.04% | 1339.3 |
| `target_image_preprocess` | 21.3 | 0.22% | — |
| `gen_vision_encode` | 2.7 | 0.03% | — |
| `dit_forward` | 3094.8 | 32.36% | — |
| `backward` | 3113.7 | 32.55% | — |
| `optimizer` | 23.1 | 0.24% | — |
| 单步合计 | 9564.7 | 100% | 1480.6 |

`past_key_values` 每个样本平均 145.8 MiB、最大 212.1 MiB；`ref_vision_embeds` 平均 4.9 MiB；`indexes_image` 24 KiB。每步已分配显存峰值 23.6 GiB，进程历史峰值为已分配 33.4 GiB、保留 37.4 GiB（加载模型阶段），每个 rank CPU RSS 峰值约 15 GB。

追踪器开销：3 次重复中，追踪器开启时 96 步总训练时间均值 923.5 s，关闭时 920.9 s，开销 0.28%；第 3 个 epoch 单步开销 0.12%。

## 5. 端到端性能（R6、R7）

计时使用 `perf/train_seconds`，覆盖每步内的数据读取、条件装载、控制通信、前向、backward 与 optimizer 更新。第 1 个 epoch 的单步均值不计首步，首步单独列出。每步 4 个样本。

| 指标 | 缓存关闭（3 次） | 缓存开启（3 次） | 变化 |
| --- | --- | --- | --- |
| 提交到首步完成（含模型加载） | 337.2 s（333.5–339.9） | 337.2 s（329.9–343.8） | 无差异 |
| 首步 | 12.86 s（12.76–12.91） | 12.57 s（12.37–12.79） | — |
| 第 1 个 epoch 单步（无复用） | 9.603 s（9.512–9.740） | 9.614 s（9.592–9.644） | 吞吐 −0.12% |
| 第 2 个 epoch 单步 | 9.531 s（9.516–9.556） | 8.445 s（8.441–8.454） | 吞吐 +12.85% |
| 第 3 个 epoch 单步（稳态） | 9.541 s（9.520–9.551） | 7.815 s（7.809–7.822） | 吞吐 **+22.09%** |
| 96 步总训练时间 | 920.9 s（916.9–925.0） | 830.9 s（829.9–832.6） | 减少 9.76% |

第 1 个 epoch 没有可复用的条件，缓存路径的单步耗时差异 0.12% 小于缓存关闭时 3 次重复之间的范围（9.512–9.740 s），作为低复用负载的开销结果。缓存构建成本以第 1 个 epoch 两组总耗时之差估算为 0.056 s，同样处于测量波动范围内；第 3 个 epoch 每个样本节省 0.43 s，按 \(N_{\mathrm{break}}=\lceil C/(t_o-t_c)\rceil\) 得到回本样本量的均值估算为 1。该估算使用接近零的构建成本差值，其不确定性受重复运行波动影响。

组内复用情况（3 次重复的计数完全相同）：

| 指标 | rank 0 | rank 1 |
| --- | --- | --- |
| 请求 | 192 | 192 |
| 本地命中 | 84（第 2–3 个 epoch 的 65.6%） | 82（64.1%） |
| 组内共同复用 | 59（第 2–3 个 epoch 的 46.1%） | 59（46.1%） |
| 本地命中但组内改为在线计算 | 25 | 23 |
| 每次请求的摘要与查询 | 2.54 ms | 2.60 ms |
| 每次请求的版本检查 | 7.60 ms | 7.50 ms |
| 每次请求的控制通信 | 0.93 ms | 0.99 ms |
| 每次复用的容器恢复 | 0.37 ms | 0.37 ms |

`cache-tracker-r1-20261112` 中 `prefix_encode` 与 `ref_vision_encode` 实际调用 266 次（缓存关闭为 384 次，减少 2 × 59 次），`ref_pixel_values` 搬运 961.5 MiB（缓存关闭为 1413.0 MiB）。该运行的 `target_image_preprocess` 主机计时为每步 102.6 ms，缓存关闭时为 21.3 ms，原因尚未分析；单步总耗时已包含该部分。

采样器每个 epoch 以 `seed + epoch` 重新打乱样本并交错分配到各 rank，一条样本再次分到同一 rank 的概率约为 1/2；组内复用要求两个 rank 在同一 micro-batch 同时命中。本实验 2 个 rank 的测量复用率为第 2–3 个 epoch 的 46.1%。按缓存设计的性能适用条件分析，rank 数增加时全部命中的概率随之减小，其他规模的复用率与收益需要另行测量。

性能汇总由 [summarize_perf.py](../scripts/summarize_perf.py) 与 [aggregate_perf.py](../scripts/aggregate_perf.py) 生成：[perf-summary-20261112.json](../raw/task17-docker-experiments-20261112/perf-summary-20261112.json)、[perf-aggregate-20261112.json](../raw/task17-docker-experiments-20261112/perf-aggregate-20261112.json)。

## 6. 容量与资源（R8）

| 指标 | 缓存关闭 | 缓存开启 |
| --- | --- | --- |
| 每个 rank 缓存计费峰值 | — | 15.47 GiB（108 个条目）/ 15.63 GiB（110 个条目），上限 16 GiB，无淘汰，无超大条目 |
| nvidia-smi 显存占用峰值 | 37.96 GiB / 37.96 GiB | 41.44 GiB / 41.75 GiB |
| 主机内存已用峰值 | 11.0–12.8 GiB | 11.2–11.3 GiB |
| `/dev/shm` 已用峰值 / 结束时 | 0.10 GiB / 0.001 GiB | 0.10 GiB / 0.001 GiB |
| 实验目录磁盘增量 | 7–19 MiB（日志） | 11–21 MiB（日志） |

资源每 5 秒采样一次，由 [monitor_resources.sh](../scripts/monitor_resources.sh) 记录到各实验的 `resources.csv`，汇总为 [resource-summary-20261112.json](../raw/task17-docker-experiments-20261112/resource-summary-20261112.json)。缓存条目约 15.6 GiB，而显存占用峰值只增加约 3.6 GiB：加载完整模型时 CUDA allocator 已保留约 37.4 GiB，训练阶段的缓存条目复用了其中的空闲块。缓存不创建磁盘文件与共享内存段。

## 7. 复跑方法

在服务器上依次执行：

```bash
bash /root/task17/Task17/bounded_condition_cache/real_run/start_container.sh
docker exec relax-task17 bash /root/Task17/bounded_condition_cache/real_run/start_ray.sh
bash /root/task17/Task17/bounded_condition_cache/real_run/queue.sh /root/task17/Task17/bounded_condition_cache/real_run/queue-20261112.txt
```

单个实验可直接执行 `docker exec relax-task17 bash /root/Task17/bounded_condition_cache/real_run/run_experiment.sh <实验名> <配置文件>`。[run_experiment.sh](../scripts/run_experiment.sh) 为每个实验建立独立目录，保存配置、实际生效的环境变量、Git 状态、依赖列表、`nvidia-smi`、提交日志、Ray 任务日志、TensorBoard 事件与资源采样。

服务器 `/root/task17/experiments/<实验名>` 保存全部产物，两次数值对照的 checkpoint 也在其中。不含 checkpoint 的产物收录在 Release 附件 `task17-4090-raw-20261112.tar.zst`，解压后位于 `raw/task17-docker-experiments-20261112/`。

## 8. 适用条件与尚未验证内容

- 本结果对应 2 GPU、MagicBrush 前 128 行、3 个 epoch 与每个 rank 16 GiB 缓存。数据集更大、单个 rank 无法容纳已出现条件时，命中率随 LRU 淘汰下降。
- 第二种模型的接入、训练对照与性能对照尚未进行。
- GPU 上的训练取消、异常退出与 DCP 恢复后的缓存行为未在完整规模模型上运行，对应行为由 CPU 测试覆盖。
- 每步逐参数梯度未单独导出；数值一致性由逐步 loss、grad_norm 与最终 checkpoint 完全一致体现。
