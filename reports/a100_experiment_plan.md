# A100 补充实验执行方案

[材料说明](../README.md) · [任务要求](https://github.com/redai-studio/community/blob/main/contributor-program/2026-cohort-2/official-task.md#task-17) · [A100 实验报告](a100_training_report.md) · [4090 训练报告](real_training_report.md) · [Qwen A100 实验报告](qwen_training_report.md)

本文件规定在 2 × NVIDIA A100-SXM4-40GB 上执行的全部实验，覆盖除第二种模型之外的 R1–R9。所有对比只在 A100 的实验之间进行；4090 的结果保留为另一套配置的记录，不与本方案的数据混合比较。GPU 数量变化对组内复用率的影响需要更多 GPU，本方案不测量，在报告中按缓存设计的性能适用条件分析说明：组内复用要求全部 rank 在同一 micro-batch 同时命中，rank 数增加时同时命中的概率随之减小。

实验结果写入[A100 实验报告](a100_training_report.md)。

## 1. 固定条件

| 项目 | 内容 |
| --- | --- |
| 代码 | 提交 `60706131f2b01db708dbc3f57c5c9e9f8cd8ffe5` 之上，第 3.2 节代码通过测试与完整 pre-commit 后形成的新提交；每个实验记录 HEAD 与 `git diff --binary` 摘要，实验期间工作区不应有未提交修改 |
| 镜像 | `ghcr.io/redai-studio/relaxrl@sha256:cd431e1094646c347aa97bc9cb8c5ac8315d4e72b0b6321f1cafd1ae12e680c6` |
| 模型 | `sensenova/SenseNova-U1.5-8B-MoT` revision `a2462cc1b549bace2c0896143bbbcbb3fa217c76`，按 HF 记录逐文件校验 |
| 数据 | `osunlp/MagicBrush` revision `6c254560879df7851233ea0fd2ae2a62764d0ef2`，由 [prepare_magicbrush.py](../scripts/prepare_magicbrush.py) 按文件名顺序从 `train-00000` 起选取；128 行子集的 metadata sha256 必须为 `3311ec0f682a80447ef1ac0f0f8567062269c944bbe9dcae488fc6b2c882def9` |
| 训练配置 | 与 [base.env](../scripts/configs/base.env) 相同：LoRA rank/alpha 32、学习率 1e-4、warmup 200、FP32 主权重、BF16 计算、global batch 4、每 epoch 32 步、`MAX_PIXELS=1048576`、FlashAttention 2、预取深度 2、种子 42、`--fsdp-load-wave-size 1` |
| 缓存预算 | 每个 rank `condition_cache_max_bytes=B`，B 由实验 A2、A3 确定后写入新的 A100 共同配置，后续实验不再改变（G2 除外） |
| 目录 | 服务器 `/root/task17` 下保持与 4090 相同的结构：`Relax`、`Task17`、`models`、`data`、`experiments`、`work`、`jobs`；实验名使用 `a100-` 前缀和执行日期 |

## 2. 新服务器准备

1. 记录 `nvidia-smi`、`nvidia-smi topo -m`、`free -g`、`df -h`、CPU 核数与驱动版本。磁盘可用空间不少于 150 GB：镜像 48 GB、模型 33 GB、数据分片约 2 GB、checkpoint 与梯度记录约 25 GB，其余为日志与余量。加载模型的 rank 主机内存峰值约 16 GB；4090 服务器主机内存为 31 GB，配合 `--fsdp-load-wave-size 1` 依次加载运行正常，新服务器主机内存不应低于该值。
2. 按[训练报告第 1 节](real_training_report.md#1-环境与版本)的方式拉取镜像，从 ModelScope 下载模型并逐文件校验，下载 MagicBrush 的 `train-00000` 至 `train-00003` 四个分片。
3. 用 git bundle 传输第 1 节规定的提交，在服务器上核对 HEAD，并确认工作区没有未提交修改。
4. 依次尝试以下启动方式，第一种通过 [check_gpu.py](../scripts/check_gpu.py) 检查的方式即为采用方式，写入报告：
   1. 按[安装文档](https://redai-studio.github.io/Relax/zh/guide/installation.html)的 `docker run --gpus all` 方式，加上 `start_container.sh` 中的挂载、`--ipc=host` 与 `--network host`，不设置其他环境变量。驱动满足镜像的 CUDA 12.9 要求，或镜像要求对 A100 所属的数据中心 GPU 允许较低驱动时，此方式可以启动；镜像自带的前向兼容库支持 A100。
   2. 上一种方式因 `cuda>=12.9` 条件拒绝启动时，增加 `NVIDIA_DISABLE_REQUIRE=1`。
   3. 前两种方式在 GPU 初始化时报错时，再使用 4090 采用的宿主机驱动库链接。
5. 在容器内运行 `torchrun --standalone --nproc_per_node=2 /root/Task17/bounded_condition_cache/real_run/check_gpu.py`，确认两张卡的矩阵乘法、FlashAttention 与 NCCL all_reduce 正常。
6. 生成 128、256、512 行三个数据子集：`--limit 128` 使用 `train-00000`；`--limit 256` 使用 `train-00000` 至 `train-00001`；`--limit 512` 使用 `train-00000` 至 `train-00003`（每个分片约 173 行，脚本在行数不足时直接报错）。每个子集的 `selection.json` 写入报告，128 行子集的 metadata 摘要与第 1 节一致。

## 3. 执行前需要完成的代码与脚本

以下修改完成后，在 CPU 或缩小维度模型上测试，并运行 `validate.sh` 与完整 pre-commit。2026-11-14 用户已授权按照本计划完成实验，授权范围包含本节三项前置实现。

### 3.1 实验脚本（Task17 目录）

- [start_container.sh](../scripts/start_container.sh)：用参数选择第 2 节第 4 步的三种启动方式，挂载目录保持不变。
- [run_experiment.sh](../scripts/run_experiment.sh)：`SAVE_INTERVAL` 改为配置可指定，未指定时保持只在最后一步保存；`effective-env.txt` 增加 `LOAD_DIR`、`SAVE_INTERVAL` 及第 3.2 节新增配置。
- [compare_parity.py](../scripts/compare_parity.py)：通过参数指定比较的步骤范围与 checkpoint 迭代编号。E2 与 D1 比较第 33–64 步及第 64 步 checkpoint。
- [summarize_perf.py](../scripts/summarize_perf.py) 与 [aggregate_perf.py](../scripts/aggregate_perf.py)：通过参数指定每个 epoch 的步数（G1 为 64 与 128 步），并提供各组的完整实验名列表。
- 新增取消脚本：等待任务日志出现指定的 `i2i SFT step=` 行后执行 `ray job stop <实验名>`，随后执行资源检查。
- 新增逐参数比较脚本：读取第 3.2 节记录的梯度与更新量，输出每一步、每个参数的比较结果。
- 新增资源检查脚本：从实验结束或终止开始，每 5 秒采样直到资源释放或达到 10 分钟，记录两张卡的显存占用、残留的训练进程、Ray actor、`/dev/shm` 与实验目录磁盘占用，以及释放所用时间。

### 3.2 Relax 仓库代码

三项都通过 `i2i_sft_config` 中默认关闭的配置启用，并在 `validate_i2i_sft_config` 中校验类型；关闭时训练执行路径与现状相同。这些配置与缓存配置一样从 checkpoint 训练一致性比较中排除，否则从 D1 恢复的 E2 会因配置不同被拒绝。记录过程只读取张量，不改变训练数值；测试需覆盖开启记录前后 loss 与参数更新一致。

| 功能 | 位置 | 行为 |
| --- | --- | --- |
| 梯度与更新量记录 | `FSDPSFTTrainer._train_step`：全部 micro-batch 的 backward 之后、`clip_sharded_gradients` 之前记录梯度；`optimizer.step()` 前后记录参数 | 每一步、每个可训练参数记录名称、shape、dtype、本地分片字节的 SHA256 与 FP64 L2 范数，梯度与更新量（FP32 主权重更新后减更新前）各一份；在配置指定的步骤额外保存完整本地分片。每个 rank 写入自己的文件 |
| 编码器权重更新 | `FSDPSFTTrainer._train_step`：指定步骤的 optimizer 更新之后 | 在无梯度上下文中把 `pipe.dit.language_model.model.embed_tokens.weight` 的本地分片乘以固定系数，然后调用 `bump_encoder_version(version_tag=...)`；缓存关闭时只修改权重。`embed_tokens` 只用于条件前缀，修改后条件特征必然变化 |
| 故障注入 | `SenseNovaSFTModel.forward` | 在指定步骤、指定 micro-batch、指定 rank 抛出 `RuntimeError`。前向阶段在条件单元执行之后直接抛出；backward 阶段在返回的 loss 上注册 tensor hook，backward 计算到该处时在 hook 中抛出 |

## 4. 实验列表

表中“次数”为独立运行次数。主配置的性能实验交替运行缓存开启与关闭。

### 4.1 环境与容量确认

| 编号 | 配置 | 次数 | 记录与判定 |
| --- | --- | --- | --- |
| A1 | 容器内 `validate.sh` | 1 | 原有 213 项、10 项诊断测试及 7 项 profiler 回归全部通过；产物目录 `/experiments/a100-validate-<日期>` |
| A2 | 缓存关闭，追踪器开启，4 步 | 1 | 单步耗时、各阶段占比、每步已分配显存峰值 \(M_0\)、加载阶段历史峰值、CPU RSS 峰值 |
| A3 | 缓存开启，追踪器开启，预算 B，64 步 | 1 | 取 \(B=\lfloor(\text{单卡总显存}-M_0-4\ \text{GiB})/1\ \text{GiB}\rfloor\ \text{GiB}\)，单卡总显存取 `nvidia-smi --query-gpu=memory.total` 的值。运行到第 2 个 epoch 结束，缓存计费接近 B 时没有显存不足；出现显存不足时 B 减少 2 GiB 后重新运行。最终 B 写入新的 A100 共同配置 |

### 4.2 在线基线（R1）

| 编号 | 配置 | 次数 | 记录与判定 |
| --- | --- | --- | --- |
| B1 | 128 行，3 epoch，缓存关闭，追踪器开启 | 3 | 编码调用次数、本地与全局重复率、重复搬运字节、阶段耗时、特征大小、显存与主机内存；离线汇总 `baseline_summary.json` |
| B2 | 128 行，3 epoch，缓存关闭，追踪器关闭 | 3 | 每 epoch 单步耗时、总训练时间、提交到首步完成时间；与 B1 比较得到追踪器开销，目标低于 2% |

### 4.3 主性能对照（R6、R7）

| 编号 | 配置 | 次数 | 记录与判定 |
| --- | --- | --- | --- |
| C1 | 128 行，3 epoch，缓存开启（B），追踪器关闭 | 3 | 与 B2 比较：每 epoch 单步耗时与吞吐变化、第 1 个 epoch（低复用）开销、总训练时间、冷启动、构建成本、回本样本量、本地命中率、组内复用率、淘汰次数、各项缓存开销；稳态吞吐提升需达到 5% |
| C2 | 同 C1，追踪器开启 | 1 | 阶段耗时明细、实际编码次数与搬运字节的减少；对比 B1 的 `target_image_preprocess` 主机计时，并用 `torch.cuda.synchronize` 前后计时或 profiler 确认差异来源 |

性能统计使用第 3.1 节修改后的 `summarize_perf.py` 与 `aggregate_perf.py`；资源统计使用 [summarize_resources.py](../scripts/summarize_resources.py)。

C2 的 profiler 记录窗口由 [select_profile_window.py](../scripts/select_profile_window.py) 根据 A3 实际记录选择：使用第二个 epoch 中最后一个同时包含在线编码与复用的完整步骤，并核对 micro-batch 编码记录与两个 rank 的共同复用计数。保存步骤、样本编号、复用状态和配置；C2 结束后核对实际记录与窗口选择一致。profiler 的采集和导出耗时保留在对应独立运行的总耗时中，主性能收益使用 B2 与 C1。

本次 C2 使用三个独立目录保存阶段统计与 profiler 采集：`a100-C2-20261114` 保存完整 96 步阶段统计；`a100-C2-profile-single-rank0-20261114`、`a100-C2-profile-single-rank1-20261114` 分别传入 `--profile-ranks 0`、`--profile-ranks 1`，并使用 `--profile-step-start 63 --profile-step-end 63` 采集第 64 步，控制主机内存占用。每次保持双 GPU 训练，使用同一代码、数据、训练参数和缓存预算完成 96 步。各次采集的 trace、双 rank 样本对应关系及资源释放单独审查；所有既有产物保留。B1 与阶段统计运行使用相同的第 2–96 步比较，两次 profiler 运行用于核查各 rank 的 CPU、CUDA API 和 GPU 活动。

### 4.4 训练数值（R3、R4、R5）

| 编号 | 配置 | 次数 | 记录与判定 |
| --- | --- | --- | --- |
| D1-off / D1-on | 128 行，确定性 attention，64 步，缓存关闭 / 开启，开启梯度与更新量记录，`SAVE_INTERVAL=32`；完整张量保存第 1、33、48、64 步 | 各 1 | 每一步、每个参数的梯度与更新量：SHA256 相同即逐元素一致；不同时用完整张量计算 \(\|X-X_{\mathrm{ref}}\|_2/\|X_{\mathrm{ref}}\|_2\leq 0.01\)、最大绝对误差、有限性及参考为零时的零值要求。没有保存完整张量的步骤出现差异时，指定该步重新运行两组。另用 [compare_parity.py](../scripts/compare_parity.py) 比较逐步 loss、grad_norm、lr、最终 checkpoint 与 RNG。缓存开启的运行在第 2 个 epoch 必须出现组内复用 |

D1 的第 32 步 checkpoint 与第 33–64 步结果是 E2 的起点与参考。

### 4.5 权重版本更新与 DCP 恢复（R3、R8）

| 编号 | 配置 | 次数 | 记录与判定 |
| --- | --- | --- | --- |
| E1-off / E1-on | 128 行，确定性 attention，64 步，第 40 步 optimizer 更新后修改 `embed_tokens` 并通知版本更新，缓存关闭 / 开启，两组均开启追踪器，保存最终 checkpoint | 各 1 | 第 40 步 optimizer 更新后的事件中，两个 rank 的 `version_invalidations` 均增加，`entries` 与计费字节变为 0；该状态计入第 40 步结束时的追踪记录。第 41 步全部在线计算，条目随新计算增长，之后的复用只能来自更新后写入的条目；E1-off 与 E1-on 的逐步 loss、grad_norm 与最终 checkpoint 逐元素一致。另检查修改后的 loss 与 D1 在第 41 步开始出现差异，说明修改确实影响了条件编码 |
| E2-on-on | 从 D1-on 的第 32 步恢复，缓存开启，追踪器开启，训练到 64 步 | 1 | 追踪器记录的第 33 步缓存统计中条目从 0 开始，第 33 步全部在线计算；第 33–64 步 loss、grad_norm 与最终 checkpoint 与 D1-on、D1-off 逐元素一致 |
| E2-off-on | 从 D1-off 的第 32 步恢复，缓存开启，追踪器开启 | 1 | 同上；恢复时缓存开关可以改变 |
| E2-on-off | 从 D1-on 的第 32 步恢复，缓存关闭，追踪器开启 | 1 | 第 33–64 步 loss、grad_norm、最终 checkpoint 与 RNG 与 D1-on、D1-off 逐元素一致 |

追踪器在已有测试中不改变训练数值，E 组开启追踪器只用于读取逐步缓存统计。

恢复训练加载 `LOAD_DIR` 中编号最大的已提交 checkpoint，D1 的目录同时包含第 32 步与第 64 步。执行 E2 前把 D1 的 `checkpoints/i2i_sft/iter_0000032` 复制到独立目录的 `i2i_sft/` 下，作为 `LOAD_DIR`；`SAVE_DIR` 使用 E2 自己的实验目录。

### 4.6 取消与异常的资源释放（R8）

| 编号 | 配置 | 次数 | 记录与判定 |
| --- | --- | --- | --- |
| F1 | 128 行，缓存开启，配置 64 步，在第 40 步日志出现后执行 `ray job stop` | 3 | 任务状态为 `STOPPED`；记录从停止到资源释放的时间；释放后两张卡的显存恢复到训练前水平，没有残留的训练进程与 Ray actor，`/dev/shm` 与实验目录磁盘无新增缓存文件；下一次实验正常启动 |
| F2 | 128 行，缓存开启，配置 64 步，第 40 步 rank 0 的第 1 个 micro-batch 前向抛出异常 | 3 | 任务状态为 `FAILED`，日志包含注入的错误；记录 rank 1 的退出方式（Ray 终止 actor 或通信超时）与从出错到全部资源释放的时间；释放后的检查项同 F1，下一次实验正常启动 |
| F3 | 同 F2，异常在 backward 中抛出 | 3 | 同 F2 |

### 4.7 收益的适用范围（R6）

除 G2 外，每组在相同数据与分辨率下运行缓存关闭与开启各 1 次；G2 使用 B2 作为对照。每个点记录本地命中率、组内复用率、淘汰次数与稳态吞吐变化，并核对“稳态收益 ≈ 组内复用率 × 条件分支占单步比例”。主配置 3 次重复的范围作为测量波动参考。

G1、G3、G4 的两次配对运行均开启追踪器，以相同计时方式取得对应数据量和分辨率下的条件分支占比，并从逐步累计计数计算稳态复用率。G2 保持追踪器关闭，使用 B1 的条件分支占比和 B2 的性能作为主配置参考。计算时区分单步耗时减少比例 \(rf\) 与吞吐提升比例 \(rf/(1-rf)\)，其中 \(r\) 为组内复用率、\(f\) 为条件分支占单步耗时的比例。

| 编号 | 变化因素 | 取值 | 预算 | 轮数 |
| --- | --- | --- | --- | --- |
| G1 | 数据量与缓存容量之比 | 256、512 行 | B | 3 epoch |
| G2 | 缓存预算 | 2、4、8 GiB | 变化 | 128 行，3 epoch |
| G3 | 训练轮数，同时作为长时间多轮资源曲线 | 6 epoch | B | 128 行 |
| G4 | 分辨率 | `MAX_PIXELS=262144` | B | 128 行，3 epoch |

128 行与预算 B 的结果来自 B2、C1。G1 的 256 行与 512 行每个 epoch 分别为 64 步与 128 步，3 epoch 共 192 步与 384 步；G3 共 192 步。

## 5. 判定与证据要求

1. 每个实验目录保存配置、实际生效的环境变量、Git 状态与改动摘要、依赖列表、`nvidia-smi`、提交日志、Ray 任务日志、TensorBoard 事件与每 5 秒的资源采样；状态必须为预期值（SUCCEEDED，F 组为 STOPPED 或 FAILED）。
2. R3 的内容、预处理、dtype 与普通参数修改失效，以及 R4 的开关与资格检查，由 A1 中的测试覆盖；完整规模的 R3 判定使用 E1 与 E2。当前 SenseNova SFT 配置不包含随机裁剪、随机增强与 caption dropout，R4 的随机数流判定使用 D1 与 E2 的 RNG 比较。
3. R5 判定使用 D1 的逐参数比较结果，确定性场景要求逐元素一致；出现非确定性差异时，按[任务要求](https://github.com/redai-studio/community/blob/main/contributor-program/2026-cohort-2/official-task.md#task-17)的数值验收要求重复运行基线，先确定近零绝对容差再判定。
4. R6 判定使用 C1 相对 B2 的稳态吞吐提升（3 次重复的均值与范围），同时报告低复用开销与 G 组结果。
5. R7 判定报告冷启动、构建、总耗时与回本条件；构建成本处于测量波动内时，按估算值报告并注明波动范围。
6. R8 判定使用 A3、C1、G3 的容量与资源曲线及 F 组的释放结果。
7. R9 判定使用第 1 节的版本记录、第 2 节的准备记录、各实验目录与 A100 训练报告中的复跑命令。
8. 结束时把全部实验产物复制到本地 `Relax/output/task17-a100-experiments-<日期>`，包括训练 checkpoint 与独立恢复目录。逐文件核对服务器与本地的 SHA256，并使用 `rsync -anzc --itemize-changes` 检查完整目录；校验记录、文件数量及保存路径写入报告。确认全部实验产物保存完整且实验进程结束后，实例可以退租。

## 6. 执行顺序

1. 第 2 节准备、第 3 节代码与脚本及其测试。
2. A1、A2、A3，确定 B。
3. D1-off、D1-on，随后 E2 三次运行与 E1 两次运行。
4. B 组、C 组交替：B2、C1、B1 依次重复 3 轮，最后运行 C2。
5. F 组 9 次运行，每次运行后执行资源检查。
6. G 组。
7. 汇总、报告与进度更新。

数值与恢复实验放在性能实验之前，出现不一致时可以先修复再进行耗时更长的性能测量。
