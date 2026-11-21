# A100 补充实验记录

[材料说明](../README.md) · [实验计划](a100_experiment_plan.md) · [4090 训练报告](real_training_report.md) · [Qwen A100 实验报告](qwen_training_report.md)

## 环境准备

日期：2026-11-14（Asia/Shanghai）。服务器为租用的单台服务器，主机名 `ubuntu22`，Ubuntu 22.04.3，16 个 CPU 核心、31 GiB 主机内存。GPU 为 2 × NVIDIA A100-SXM4-40GB，每张卡报告 40960 MiB，驱动 550.127.05；`nvidia-smi topo -m` 报告 GPU 间连接为 PHB。

开始时文件系统可用空间为 159772389376 bytes。Docker 29.1.3 和 NVIDIA Container Toolkit 已安装，Docker 使用 overlay2，固定镜像已导入。模型下载期间使用已验证可正常传输的 ModelScope CDN 地址 `36.163.116.58`；下载入口结束后，已用 `cmp` 确认 `/etc/hosts` 与原内容一致。模型全部文件已按 HF 指定 revision 的文件大小、LFS SHA256 或 Git blob SHA1 校验通过，证据见 [model-verified-files.json](../raw/task17-a100-experiments-20261114/preparation/model-verified-files.json)。MagicBrush 的四个分片均通过 SHA256 校验，证据见 [data-verified-files.json](../raw/task17-a100-experiments-20261114/preparation/data-verified-files.json)。下载、安装与环境记录保存在 `/root/task17/jobs/`。

采用 `bash start_container.sh direct` 启动容器，保持镜像自带的驱动兼容方式。`torchrun --standalone --nproc_per_node=2 check_gpu.py` 已在两个 rank 上通过 BF16 矩阵乘法、FlashAttention 和 NCCL all_reduce 检查，归约结果均为 3。实际环境为 Python 3.12.3、PyTorch 2.11.0+cu129、torchvision 0.26.0+cu129、CUDA 12.9、Transformers 5.12.1、FlashAttention 2.7.4.post1、Ray 2.58.0、NumPy 1.26.4；日志为服务器 `a100-preparation-20261114/gpu-check-direct.log`，每个实验的 `environment.txt` 保存完整依赖列表。

固定版本为镜像 `ghcr.io/redai-studio/relaxrl@sha256:cd431e1094646c347aa97bc9cb8c5ac8315d4e72b0b6321f1cafd1ae12e680c6`、模型 `sensenova/SenseNova-U1.5-8B-MoT` revision `a2462cc1b549bace2c0896143bbbcbb3fa217c76`、数据 `osunlp/MagicBrush` revision `6c254560879df7851233ea0fd2ae2a62764d0ef2`。服务器模型目录为 `/root/task17/models/SenseNova-U1.5-8B-MoT`，数据目录为 `/root/task17/data/magicbrush-{128,256,512}`。

## 前置实现与验证

镜像的 88 个层记录已在本地与服务器完成 SHA256 校验，总计 22207091234 bytes，两份结果经 `cmp` 检查逐字节一致，证据见 [image-verified-layers.json](../raw/task17-a100-experiments-20261114/preparation/image-verified-layers.json) 与 [server-image-verified-layers.json](../raw/task17-a100-experiments-20261114/preparation/server-image-verified-layers.json)。生成的 [OCI index](../raw/task17-a100-experiments-20261114/preparation/image-oci-index.json) 保留计划指定的 manifest digest。服务器使用 Docker archive 和 `docker load` 导入，image ID 与原始 config digest `sha256:9b42d559b0e6ec3b20b69fc6b079ea2e256f6c9e8c898c3148082fb8d95ae7e8` 一致。

代码提交：`9783ebe924323c9742fbd735fb850f93a0a157f0`，分支 `feat/task17-frozen-conditioning-cache`。提交经过完整 pre-commit 的 16 个检查，工作区没有未提交修改。20 个受测文件 SHA256 与提交后的文件逐个一致。相同提交已通过 Git bundle 传入服务器 `/root/task17/Relax`。

- `experiment_diagnostics` 为默认关闭的字典配置，从 checkpoint 训练一致性比较中排除。配置包括 `output_dir`、`record_parameters`、`tensor_steps`、`encoder_update_step`、`encoder_update_scale` 和 `fault`。
- 梯度在全部 micro-batch backward 完成、梯度裁剪开始之前记录；更新量由 optimizer 更新后的 FP32 本地参数分片减去更新前的 FP32 副本计算。每个 rank 独立保存 JSONL 和指定步骤的完整张量，记录过程检查数值有限性。
- 编码器更新在指定 optimizer 步骤完成后修改 `embed_tokens.weight` 的本地分片并通知缓存版本更新；事件文件保存更新前后的缓存状态。
- 故障配置使用从 1 开始的 `step`、`micro`，以及从 0 开始的 `rank`；前向故障在条件单元执行结束后触发，反向故障由 loss 的 Tensor hook 触发。
- FSDP profiler 根据 WORLD rank 生成 trace 与内存记录名称，DP rank 使用 global rank，TP 与 PP rank 均为 0。已有 profiler 回归包含真实 Gloo 初始化和 Python 调用记录导出。
- 本地完整相关回归：228 项通过、2 项 CUDA 测试因本机没有可用 GPU 跳过，耗时 13.97 秒。独立环境为 Python 3.13.15、PyTorch 2.15.0+cpu、Transformers 5.12.1；日志、JUnit、环境、源码摘要位于 [cpu-validation-profiler](../raw/task17-a100-experiments-20261114/cpu-validation-profiler)。实验工具测试另有 3 项通过，覆盖步骤选择、不同 epoch 长度和逐参数比较；日志见 [preparation](../raw/task17-a100-experiments-20261114/preparation)。
- A1：官方镜像内 230 项全部通过、0 项跳过，100.41 秒；Ruff 检查、格式检查和 Git 改动格式检查通过。实验工具另有 3 项通过，5.54 秒。日志、JUnit、环境、源码与摘要已复制到本地 [a100-validate-20261114](../raw/task17-a100-experiments-20261114/a100-validate-20261114)。

CPU 验证命令：

```bash
VALIDATION_PYTHON=/home/xbei/Relax/Relax/output/task17-a100-experiments-20261114/preparation/venv/bin/python \
ARTIFACT_DIR=/home/xbei/Relax/Relax/output/task17-a100-experiments-20261114/cpu-validation-profiler \
PYTHONPYCACHEPREFIX=/home/xbei/Relax/Task17/.work/a100-bytecode \
bash Task17/bounded_condition_cache/validate.sh
```

## 实验状态

A1–G4 全部完成。39 次主计划运行均通过服务器与本地统一审查，C2 两个 rank 的独立 profiler 采集另外通过运行、样本与 trace 检查。共同缓存预算为每个 rank 12 GiB；全部数值对照、主性能、容量、取消、故障及适用范围实验的证据已保存。顺序实验入口于 2026-11-14 23:37:37（Asia/Shanghai）完成，资源曲线、全部 checkpoint 及其他实验产物已复制至本地，2026-11-15 的逐文件 SHA256 与完整目录校验全部通过。

三个数据子集均按分片文件名顺序选取，128、256、512 行分别具有相同数量的不同条件。metadata SHA256 分别为 `3311ec0f682a80447ef1ac0f0f8567062269c944bbe9dcae488fc6b2c882def9`、`5d5e37539587d9fe74ab712aebd15458c0d99f245dbc0c3d34ac7df4950ef20b`、`de199e8498f611efe3a39e9f3f95ca721c9adc3b6ff80d9e64cb2248747c34cd`。每个子集的 `selection.json` 位于服务器 `/root/task17/data/magicbrush-<行数>/`，其内容也保存在环境准备目录的 `data-<行数>.log`。

## A2：在线编码与候选容量

运行标识为 `a100-A2-20261114`，状态 `SUCCEEDED`，完整完成 4 步，资源检查的 8 个项目全部通过。原始记录见本地 [A2 产物](../raw/task17-a100-experiments-20261114/a100-A2-20261114)；预算计算保存为 [capacity.json](../raw/task17-a100-experiments-20261114/a100-orchestration-20261114/capacity.json)。

| 指标 | 实测值 |
| --- | --- |
| GPU 0、1 总显存 | 各 40 GiB |
| rank 0 单步已分配显存峰值 | 25329218560 bytes |
| rank 1 单步已分配显存峰值 | 25329265664 bytes |
| 加载阶段历史已分配显存峰值 | 35863955968 bytes，33.40 GiB |
| rank 主机 RSS 历史峰值 | 15583002624 bytes，14.51 GiB |
| 第一步耗时 | 13.1515 秒 |
| 4 步训练总耗时 | 43.8976 秒 |
| 提交至第一步完成 | 350.14 秒 |

预算按计划计算为 \(\lfloor40-23.5897-4\rfloor=12\ \mathrm{GiB}\)，每个 rank 对应 12884901888 bytes，已通过 A3 两轮容量验证并写入共同配置。

去除首步后，两个 rank 的阶段计时均值为：参考图预处理 313.01 ms、参考图编码 4.40 ms、条件前缀编码 3050.88 ms、目标图预处理 159.61 ms、生成图编码 4.64 ms、DiT 前向 3587.90 ms、反向 3156.48 ms、optimizer 26.72 ms；完整单步均值 10248.58 ms。该窗口仅用于容量确认和短程阶段记录，正式性能对照使用 B、C 组。

## A3：容量与两轮运行

`a100-A3-b12-20261114` 的 64 步全部完成，Ray 状态为 `SUCCEEDED`，8 项资源释放检查全部通过。单步已分配显存峰值为 38200532480 bytes（35.58 GiB），GPU 实际占用采样峰值为每卡 38930 MiB。缓存预算在第 2 个 epoch 内开始限制条目数量，容量未超出上限。

| 指标 | rank 0 | rank 1 |
| --- | --- | --- |
| 缓存计费峰值 / bytes | 12846170112 | 12846268416 |
| 最多条目数量 | 86 | 82 |
| 128 次请求中的本地命中次数 | 33 | 35 |
| 共同复用次数 | 21 | 21 |
| 淘汰次数 | 9 | 11 |
| 条目超过预算的跳过次数 | 0 | 0 |

第 2 个 epoch 的每个 rank 共有 64 次请求，共同复用率为 32.8125%。第 1 个 epoch 去除首步后的单步均值为 9.6510 秒，第 2 个 epoch 为 8.5927 秒；这两组时间对应不同训练轮数，正式性能判定使用相同窗口的 B2、C1 对照。

本地 [A3 原始产物](../raw/task17-a100-experiments-20261114/a100-A3-b12-20261114)、[容量与步骤审查](../raw/task17-a100-experiments-20261114/preparation/a100-calibration-audit.json)、[资源曲线](../raw/task17-a100-experiments-20261114/preparation/a100-A3-resource-curves/a100-A3-b12-20261114-resources.svg)已保存。缓存退出日志记录清理之前的条目数量，最终释放状态由 `release-samples.jsonl` 中的实际采样确认；包括本次 JobSupervisor 在内的全部相关 actor 均为 `DEAD`，GPU 占用恢复为每卡 14 MiB，共享内存目录没有新增文件。

C2 的 profiler 窗口选定为第 64 步，对应 `profile_step_start=63`、`profile_step_end=63`。每个 rank 的两个 micro-batch 分别包含一次共同复用和一次在线编码；rank 0 的样本编号为 6、35，rank 1 为 31、39。选择规则及 A3 记录见 [profile-window-single-step.json](../raw/task17-a100-experiments-20261114/preparation/profile-window-single-step.json)。

## D1：逐参数数值记录

D1-off 与 D1-on 均已完成 64 步，保存第 32、64 步 checkpoint，资源检查全部通过。[关闭缓存时的完整性审查](../raw/task17-a100-experiments-20261114/a100-orchestration-20261114/parameter-audit-off.json)与[开启缓存时的完整性审查](../raw/task17-a100-experiments-20261114/a100-orchestration-20261114/parameter-audit-on.json)确认每组两个 rank、588 个可训练参数的梯度与 FP32 更新量全部存在，各有 \(64\times2\times588\times2=150528\) 条记录。每组第 1、33、48、64 步的 16 个完整张量文件中，共 9408 个张量的 shape、dtype 与 SHA256 均和摘要一致。

梯度在有效 batch 的 backward 完成后、梯度裁剪之前记录；更新量为 optimizer 更新后的 FP32 参数减去更新前的参数，即 \(\Delta\theta=\theta_{\mathrm{after}}-\theta_{\mathrm{before}}\)。[逐参数比较](../raw/task17-a100-experiments-20261114/a100-orchestration-20261114/parameter-parity.json)的 150528 条记录全部一致；[训练状态比较](../raw/task17-a100-experiments-20261114/a100-orchestration-20261114/compare-a100-D1-off-20261114-vs-a100-D1-on-20261114.json)中，64 步 loss、grad_norm、lr、最终 checkpoint 的 2352 个张量和 6 个 RNG 张量全部一致，最大绝对误差与最大相对 L2 误差均为 0。

比较命令分别为 `python compare_parameters.py --reference /experiments/a100-D1-off-20261114/diagnostics --candidate /experiments/a100-D1-on-20261114/diagnostics --output /experiments/a100-orchestration-20261114/parameter-parity.json`，以及 `python compare_parity.py --reference /experiments/a100-D1-off-20261114 --candidate /experiments/a100-D1-on-20261114 --work /work/parity-a100 --iteration 64 --start-step 1 --end-step 64 --output /experiments/a100-orchestration-20261114/compare-a100-D1-off-20261114-vs-a100-D1-on-20261114.json`。两条命令均在上述固定镜像与代码提交下退出 0；完整性审查使用 `audit_parameters.py`，两组命令均退出 0。checkpoint 已保存于本地各实验目录的 `checkpoints/i2i_sft/`，文件内容与服务器完全一致。

[记录类型统计](../raw/task17-a100-experiments-20261114/preparation/D1-record-dtypes.json)确认全部梯度与更新量使用 `torch.float32`。其中 588 条梯度记录、1059 条更新量记录的 L2 范数为零，对应两组内容一致，满足参考值为零时的零值要求。

D1-on 的两个 rank 各完成 21 次共同复用，第 2 个 epoch 的共同复用率为 32.8125%；本地命中次数分别为 33、35，淘汰次数分别为 9、11。[训练记录汇总](../raw/task17-a100-experiments-20261114/preparation/a100-D1-training-records.json)保存实际计数。两组全部产物及第 32、64 步 checkpoint 已保存到本地，并通过[本地目录审查](../raw/task17-a100-experiments-20261114/preparation/a100-D1-on-audit.json)和完整归档的逐文件 SHA256 校验。

## E2：恢复训练

E2-on-on 从 D1-on 的第 32 步 checkpoint 恢复，缓存保持开启，完成第 33–64 步，资源检查通过。与 [D1-off 的比较](../raw/task17-a100-experiments-20261114/a100-orchestration-20261114/compare-a100-D1-off-20261114-vs-a100-E2-on-on-20261114.json)和与 [D1-on 的比较](../raw/task17-a100-experiments-20261114/a100-orchestration-20261114/compare-a100-D1-on-20261114-vs-a100-E2-on-on-20261114.json)均确认：32 步 loss、grad_norm、lr 全部一致，最终 checkpoint 的 2352 个张量及 6 个 RNG 张量全部一致，最大绝对误差和最大相对 L2 误差均为 0。比较使用 `compare_parity.py --iteration 64 --start-step 33 --end-step 64`，完整参数与输出保存在服务器 `/root/task17/experiments/a100-orchestration-20261114/compare-a100-D1-{off,on}-20261114-vs-a100-E2-on-on-20261114.{log,json}`。

本地 [E2-on-on 原始产物](../raw/task17-a100-experiments-20261114/a100-E2-on-on-20261114)与[独立审查](../raw/task17-a100-experiments-20261114/preparation/a100-E2-on-on-audit.json)已保存。两个 rank 在第 33 步均记录两次请求、两次插入和零次复用，各 micro-batch 均执行 `ref_vision_encode` 和 `prefix_encode`。rank 0 的样本编号为 68、121，rank 1 为 38、40。

E2-off-on 从 D1-off 的第 32 步恢复并开启缓存，同样完成第 33–64 步。[D1-off 参考比较](../raw/task17-a100-experiments-20261114/a100-orchestration-20261114/compare-a100-D1-off-20261114-vs-a100-E2-off-on-20261114.json)与 [D1-on 参考比较](../raw/task17-a100-experiments-20261114/a100-orchestration-20261114/compare-a100-D1-on-20261114-vs-a100-E2-off-on-20261114.json)中的逐步指标、2352 个 checkpoint 张量及 6 个 RNG 张量全部一致，最大误差均为 0；[本地原始产物](../raw/task17-a100-experiments-20261114/a100-E2-off-on-20261114)通过[独立审查](../raw/task17-a100-experiments-20261114/preparation/a100-E2-off-on-audit.json)。

E2-on-off 从 D1-on 的第 32 步恢复并关闭缓存，完成第 33–64 步。[D1-off 参考比较](../raw/task17-a100-experiments-20261114/a100-orchestration-20261114/compare-a100-D1-off-20261114-vs-a100-E2-on-off-20261114.json)与 [D1-on 参考比较](../raw/task17-a100-experiments-20261114/a100-orchestration-20261114/compare-a100-D1-on-20261114-vs-a100-E2-on-off-20261114.json)中的逐步指标、2352 个 checkpoint 张量及 6 个 RNG 张量全部一致，最大误差均为 0；[本地原始产物](../raw/task17-a100-experiments-20261114/a100-E2-on-off-20261114)通过[独立审查](../raw/task17-a100-experiments-20261114/preparation/a100-E2-on-off-audit.json)。[缓存语义检查](../raw/task17-a100-experiments-20261114/a100-orchestration-20261114/cache-semantics.json)确认 E2-on-on、E2-off-on 的两个 rank 均在恢复后的首次请求执行在线编码并建立新条目。

## E1：编码器版本更新

E1-off 已完成 64 步，资源检查及[本地目录审查](../raw/task17-a100-experiments-20261114/preparation/a100-E1-off-audit.json)通过。两个 rank 在第 40 步 optimizer 更新后均记录 `encoder_update` 事件；修改对象为 `embed_tokens.weight`，系数为 1.01，实际配置与事件保存在 [E1-off 产物](../raw/task17-a100-experiments-20261114/a100-E1-off-20261114)。

[前 40 步比较](../raw/task17-a100-experiments-20261114/preparation/encoder-update-prefix.json)确认 loss、grad_norm、lr 与 D1-off 全部一致。[修改时刻附近的记录](../raw/task17-a100-experiments-20261114/preparation/encoder-update-off-window.json)显示，第 41 步 loss 从 D1 的 `0.027182362973690033` 变化为 E1-off 的 `0.027171628549695015`，说明修改已经影响条件编码。E1-on 的第 40 步事件确认：rank 0 的 71 个条目、10937475072 bytes，以及 rank 1 的 72 个条目、11162861568 bytes，均在更新后清零；两个 rank 的 `version_invalidations` 从 0 增至 1，该状态与第 40 步追踪记录完全一致。

第 41 步的每个 micro-batch 均执行参考图编码与条件前缀编码，各 rank 新增两个条目。第 41–64 步的每个 rank 共 48 次请求全部在线编码，最终各有 48 个条目；窗口内保存的条目全部来自参数更新之后的计算。[19 项语义检查](../raw/task17-a100-experiments-20261114/a100-orchestration-20261114/cache-semantics.json)全部通过，并在本地独立执行 `verify_a100_semantics.py` 获得[相同结果](../raw/task17-a100-experiments-20261114/preparation/cache-semantics-local.json)。

[E1 两组数值比较](../raw/task17-a100-experiments-20261114/a100-orchestration-20261114/compare-a100-E1-off-20261114-vs-a100-E1-on-20261114.json)确认 64 步 loss、grad_norm、lr、2352 个 checkpoint 张量和 6 个 RNG 张量全部精确一致，最大绝对误差与最大相对 L2 误差均为 0。比较使用 `compare_parity.py --iteration 64 --start-step 1 --end-step 64`，资源检查通过。[E1-on 全部产物](../raw/task17-a100-experiments-20261114/a100-E1-on-20261114)及最终 checkpoint 已保存到本地，并通过[本地目录审查](../raw/task17-a100-experiments-20261114/preparation/a100-E1-on-audit.json)和完整归档的逐文件 SHA256 校验。

## B/C：在线基线与主性能

B1、B2、C1 各三次运行均完成 96 步，每次 8 项资源检查通过；九次运行的完整产物已复制到本地，[统一审查](../raw/task17-a100-experiments-20261114/preparation/main-audit.json)确认版本、配置、步骤和资源证据有效。[逐次计时](../raw/task17-a100-experiments-20261114/preparation/main-performance-summary.json)、[三轮统计](../raw/task17-a100-experiments-20261114/preparation/main-performance-aggregate.json)与[配对范围](../raw/task17-a100-experiments-20261114/preparation/main-performance-repeats.json)已保存。性能汇总使用本地 Python 3.13.15、TensorBoard 事件文件和 `TZ=Asia/Shanghai`，执行命令示例为：

```bash
env TZ=Asia/Shanghai Relax/output/task17-a100-experiments-20261114/preparation/venv/bin/python \
  Task17/bounded_condition_cache/real_run/summarize_perf.py \
  Relax/output/task17-a100-experiments-20261114/a100-B2-r1-20261114 \
  --output Relax/output/task17-a100-experiments-20261114/preparation/a100-B2-r1-performance.json
```

| 运行 | 首步 / 秒 | 提交至首步完成 / 秒 | epoch 1 单步 / 秒 | epoch 2 单步 / 秒 | epoch 3 单步 / 秒 | 96 步总时间 / 秒 |
| --- | --- | --- | --- | --- | --- | --- |
| B2 第一轮 | 14.5023 | 348.7059 | 9.68802 | 9.67529 | 9.69745 | 934.7586 |
| C1 第一轮 | 13.1841 | 347.5958 | 9.65856 | 8.59719 | 8.56335 | 861.7368 |
| B1 第一轮 | 13.9559 | 354.8109 | 9.66821 | 9.61952 | 9.61934 | 929.3138 |
| B2 第二轮 | 14.5797 | 349.4995 | 9.60185 | 9.55769 | 9.55445 | 923.8252 |
| C1 第二轮 | 14.0534 | 358.7925 | 9.72659 | 8.64750 | 8.62962 | 868.4456 |
| B1 第二轮 | 14.1561 | 355.8382 | 9.67278 | 9.64729 | 9.65896 | 931.8123 |
| B2 第三轮 | 14.0831 | 351.7309 | 9.64624 | 9.63609 | 9.66502 | 930.7522 |
| C1 第三轮 | 13.8030 | 359.4541 | 9.70410 | 8.61036 | 8.57258 | 864.4842 |
| B1 第三轮 | 14.2131 | 354.6960 | 9.69026 | 9.65880 | 9.66666 | 933.0255 |

epoch 1 单步均值排除首步，总训练时间保留首步。`train_seconds` 的计时从样本编号分发完成后开始，覆盖图像读取等待、条件处理、前向、反向、optimizer 更新、指标归约及 GPU 同步；指标上报、步骤间 Ray 调度和 checkpoint 写入在该区间之外。性能组关闭 checkpoint 保存。

相邻步骤的完成间隔另行计入调度与记录开销；第三个 epoch 使用 TensorBoard 中第 64 步至第 96 步的完成时间差除以 32。提交至任务结束的时间包含模型加载、训练和退出过程。

三次 C1 的全部非计时缓存字段完全一致：rank 0、1 分别记录 68、69 次本地命中，各完成 43 次共同复用，分别淘汰 42、37 个条目。第 2–3 个 epoch 每个 rank 共 128 次请求，共同复用率为 \(43/128=33.59375\%\)，本地命中率分别为 53.125%、53.90625%。缓存计费峰值分别为 12846170112、12846268416 bytes，均低于 12 GiB；最多各保存 86 个条目，条目超过预算的跳过次数均为 0。

三次运行、两个 rank 的每请求开销均值分别为：查找 4.019 ms、跨 rank 控制 1.722 ms、版本检查 9.348 ms；对应观测范围分别为 3.716–4.289、1.532–1.931、9.150–9.591 ms。恢复操作按实际复用次数计算，平均每次 0.534 ms，范围为 0.494–0.583 ms。各项原始累计时间和计数保存在逐次计时汇总中。

C1 第一轮的[资源曲线](../raw/task17-a100-experiments-20261114/preparation/a100-C1-r1-resource-curves/a100-C1-r1-20261114-resources.svg)及[数据数量清单](../raw/task17-a100-experiments-20261114/preparation/a100-C1-r1-resource-curves/resource-curves.json)已生成，两张 GPU 各有 250 次采样，覆盖 1261.661 秒。该组关闭追踪器，GPU、主机、共享内存和磁盘曲线使用资源采样，缓存容量与条目峰值使用训练结束统计。本地 Python 3.13.15、Matplotlib 3.11.2 的执行命令为：

```bash
env MPLCONFIGDIR=/home/xbei/Relax/Task17/.work/matplotlib \
  Relax/output/task17-a100-experiments-20261114/preparation/venv/bin/python \
  Task17/bounded_condition_cache/real_run/plot_resources.py \
  Relax/output/task17-a100-experiments-20261114/a100-C1-r1-20261114 \
  --output Relax/output/task17-a100-experiments-20261114/preparation/a100-C1-r1-resource-curves
```

B1 三轮 96 步总训练时间均值为 931.3839 秒，相对 B2 增加 0.1726%；第三个 epoch 单步均值为 9.64832 秒，增加 0.0970%。三轮配对的总时间变化范围为 −0.5825% 至 +0.8646%，达到追踪器平均开销低于 2% 的目标。

三组 B1 的 `sample_stats`、`encoding_stats`、`transfer_stats`、`feature_tensor_bytes` [逐项比较完全一致](../raw/task17-a100-experiments-20261114/preparation/online-repeat-checks.json)。每组 [baseline_summary.json](../raw/task17-a100-experiments-20261114/a100-B1-r1-20261114/baseline_stats/baseline_summary.json)记录 384 次样本处理、128 个全局不同样本、166 次 rank 本地重复和 256 次全局重复，对应重复率 43.2292%、66.6667%。参考图编码与条件前缀编码分别执行 384 次，每项包含 166 次本地重复编码及 256 次全局重复编码；生成图编码执行 384 次。

| 搬运项目 | 总量 / MiB | 本地重复 / MiB | 全局重复 / MiB |
| --- | --- | --- | --- |
| `ref_pixel_values` | 1413.0000 | 609.0000 | 942.0000 |
| `ref_grid_hw` | 0.0059 | 0.0025 | 0.0039 |
| `prefix_input_ids` | 2.6039 | 1.1231 | 1.7360 |
| `target_input_latents` | 2304.0000 | 996.0000 | 1536.0000 |
| 合计 | 3719.6098 | 1606.1256 | 2479.7399 |

排除首步后，三次运行中两个 rank 的单步阶段均值为：参考图预处理 308.011 ms、参考图编码 3.908 ms、条件前缀编码 3052.609 ms、目标图预处理 26.740 ms、生成图编码 4.690 ms、DiT 前向 3143.235 ms、反向 3140.297 ms、optimizer 27.853 ms。条件阶段前三项之和占完整计时的范围为 34.8071%–34.8780%。目标图预处理的三轮均值范围为 25.462–28.066 ms；[与 C2 相同窗口的阶段汇总](../raw/task17-a100-experiments-20261114/preparation/comparison-stages.json)使用第 2–96 步。

`past_key_values` 的平均大小为 145.8208 MiB、最大为 212.1328 MiB，`indexes_image` 每份 24 KiB；按平均大小估算，12 GiB 可保存约 84 个条目，实际条目数量还受样本大小与淘汰顺序影响。参考图视觉特征平均为 4.90625 MiB、最大为 8 MiB。三次运行的训练单步已分配显存峰值均为 23.58991 GiB，加载历史峰值均为 33.40091 GiB，rank 主机 RSS 历史峰值范围为 14.25063–14.60886 GiB。

[三轮主性能汇总](../raw/task17-a100-experiments-20261114/preparation/main-performance-summary.json)记录，B2、C1 第三个 epoch 单步均值分别为 9.63897、8.58852 秒，按均值计算吞吐提高 12.2309%；三轮配对提升范围为 10.7169%–13.2436%。第二个 epoch 吞吐提高 11.6573%，96 步训练总时间均值从 929.7786 秒减少至 864.8889 秒，减少 6.9791%。主配置达到计划要求的 5% 稳态吞吐提升。

[三轮实际经过时间](../raw/task17-a100-experiments-20261114/preparation/main-walltime-summary.json)记录，第三个 epoch 的步骤完成间隔均值从 10.18368 秒减少至 9.13725 秒，包含步骤间调度与记录开销的吞吐提高 11.4523%，三轮配对范围为 10.0808%–12.2803%。提交至任务结束的均值从 1323.1067 秒减少至 1263.2073 秒，减少 4.5272%。提交至首步完成的均值分别为 349.9787、355.2808 秒，首步训练耗时分别为 14.3884、13.6802 秒。

首个 epoch 全部条件首次出现，含首步的时间均值增加 0.2790%，对应构建成本差值 0.8742 秒；三轮差值为 −2.2316、3.3408、1.5135 秒。该差值处于观测波动范围内。按第三个 epoch 每样本平均节省 0.2626 秒估算，回本量为 \(\lceil0.8742/0.2626\rceil=4\) 个样本当量，表示进入相同复用状态后覆盖构建成本所需的估算处理量；实际首次复用位置取决于样本顺序和共同命中情况。

[summarize_walltime.py](../scripts/summarize_walltime.py) 已在本地 Python 3.13.15 下读取 B2 第一轮产物执行通过，命令为：

```bash
Relax/output/task17-a100-experiments-20261114/preparation/venv/bin/python \
  Task17/bounded_condition_cache/real_run/summarize_walltime.py \
  Relax/output/task17-a100-experiments-20261114/preparation/a100-B2-r1-performance.json \
  --root Relax/output/task17-a100-experiments-20261114 \
  --output Relax/output/task17-a100-experiments-20261114/preparation/a100-B2-r1-walltime.json
```

## C2：阶段统计与 profiler

[阶段统计运行](../raw/task17-a100-experiments-20261114/a100-C2-20261114)完成 96 步，[本地审查](../raw/task17-a100-experiments-20261114/preparation/a100-C2-audit.json)及 8 项资源检查通过，30 个文件、3640567 bytes 已复制到本地。[与三次 B1 相同窗口的比较](../raw/task17-a100-experiments-20261114/preparation/comparison-stages.json)使用第 2–96 步，每个 rank 各 95 步。C2 的 294 次在线编码 micro-batch 中，目标图预处理主机时间平均为 11.057 ms，中位数为 8.332 ms；86 次复用 micro-batch 的均值为 133.940 ms，中位数为 111.072 ms。

完整 96 步中的参考图编码和条件前缀编码各为 298 次，相对 B1 各减少 86 次；生成图编码为 384 次。参考图像素搬运由 1413 MiB 减少至 1077 MiB，条件前缀 token ID 搬运由 2.60394 MiB 减少至 1.99559 MiB，目标图 latent 搬运为 2304 MiB；总搬运量由 3719.60980 MiB 减少至 3383.00014 MiB。实际计数见 [C2 baseline_summary.json](../raw/task17-a100-experiments-20261114/a100-C2-20261114/baseline_stats/baseline_summary.json)。

| 阶段 | B1 三轮平均 / ms | C2 / ms |
| --- | --- | --- |
| 参考图预处理 | 308.011 | 223.944 |
| 参考图编码 | 3.908 | 3.134 |
| 条件前缀编码 | 3052.609 | 2354.342 |
| 目标图预处理 | 26.740 | 77.735 |
| 生成图编码 | 4.690 | 4.752 |
| DiT 前向 | 3143.235 | 3127.142 |
| backward | 3140.297 | 3107.272 |
| optimizer | 27.853 | 27.879 |

表格使用每个 rank 的单步阶段均值，再对三个 B1 运行取平均；目标图预处理使用主机计时。C2 的单步已分配显存峰值为 35.5770 GiB，rank 主机 RSS 历史峰值为 14.2827 GiB。

独立运行 `a100-C2-profile-single-rank0-20261114`、`a100-C2-profile-single-rank1-20261114` 使用相同的双 GPU 训练配置，每次选择一个 rank 采集第 64 步，以控制主机内存占用。参数选择使用实际镜像中的双 Gloo rank 检查，分别验证选中的 rank 与未选中的 rank；步骤参数为 `profile_step_start=63`、`profile_step_end=63`。rank 0 的检查命令为：

```bash
PYTHONPATH=/root/Relax:/root/Megatron-LM torchrun --standalone --nproc_per_node=2 \
  /root/Task17/bounded_condition_cache/real_run/check_profile_selection.py \
  --use-pytorch-profiler --profile-ranks 0 --profile-step-start 63 --profile-step-end 63
```

rank 0 的独立采集完成 96 步，服务器与[本地独立审查](../raw/task17-a100-experiments-20261114/preparation/profile-capture-audit-local.json)均通过。两个 rank 的实际样本与 A3 选择一致，114554428 bytes 的 trace 已复制到本地，SHA256 为 `5fba8ce2d553c4802038289133cd2bfc3840399cb80867a8342bc2ac336adfd9`，与服务器相同。[资源记录](../raw/task17-a100-experiments-20261114/preparation/profile-single-rank0-resources.json)中的主机使用峰值为 24933326848 bytes，最低可用内存为 8677257216 bytes；[释放检查](../raw/task17-a100-experiments-20261114/preparation/profile-single-rank0-release.json)全部通过，确认释放的时间上界为 17.0242 秒。

rank 1 的独立采集同样完成 96 步，服务器与本地审查均通过，完整采集审查记录的 `complete_capture_set=true`。114326385 bytes 的 trace 已复制到本地，SHA256 为 `89db5bee8848d7af22edfa1e3e31b03415c252ba6b8d9836ea2f23c7248abd71`，与服务器相同。[资源记录](../raw/task17-a100-experiments-20261114/preparation/profile-single-rank1-resources.json)中的主机使用峰值为 26019880960 bytes，最低可用内存为 7590703104 bytes；[释放检查](../raw/task17-a100-experiments-20261114/preparation/profile-single-rank1-release.json)全部通过，确认释放的时间上界为 16.1428 秒。两个采集运行的最终缓存非计时统计均与 C1 相同。

[rank 0 原始事件提取](../raw/task17-a100-experiments-20261114/preparation/profile-single-rank0-events.json)和 [rank 1 原始事件提取](../raw/task17-a100-experiments-20261114/preparation/profile-single-rank1-events.json)分别完整读取 trace，各定位两个 `SenseNovaU1ImageUnit_InputImageEmbedder.process` 调用。[rank 0 关联分析](../raw/task17-a100-experiments-20261114/preparation/profile-single-rank0-analysis.json)和 [rank 1 关联分析](../raw/task17-a100-experiments-20261114/preparation/profile-single-rank1-analysis.json)使用相同的 correlation ID 与 External id 连接 CPU 复制调用和 GPU 复制事件。

| 第 64 步 rank / 样本 | 条件路径 | 预处理函数 / ms | `cudaMemcpyAsync` 主机调用 / ms | `cudaStreamSynchronize` / ms | 6 MiB GPU 复制 / ms |
| --- | --- | --- | --- | --- | --- |
| 0 / 6 | 共同复用 | 107.743 | 99.155 | 0.043 | 0.394 |
| 0 / 35 | 在线编码 | 8.984 | 0.576 | 0.045 | 0.493 |
| 1 / 31 | 共同复用 | 179.025 | 0.589 | 0.045 | 0.458 |
| 1 / 39 | 在线编码 | 8.545 | 0.767 | 0.040 | 0.642 |

以 rank 0 复用样本的预处理函数开始时间为 0，NCCL AllGather 从 −3.792 ms 执行至 102.760 ms；参数复制从 102.773 ms 执行至 106.710 ms；目标图像复制从 106.741 ms 执行至 107.134 ms。CPU 上的 `cudaMemcpyAsync` 在 7.945 ms 开始，在 107.100 ms 返回。AllGather 的事件包含两个 rank、BF16、1325354176 个输出元素，并通过 External id 对应到 FSDP 前向之前提交的 `record_param_comms`。

CUDA stream 是按顺序执行 GPU 工作的任务队列。[FSDP 的参数汇集实现](../raw/task17-a100-experiments-20261114/preparation/torch-2.11-fsdp-collectives.py)中的 `foreach_all_gather_copy_out` 在当前计算 stream 中等待参数汇集的完成事件，再执行参数复制；目标图像复制也进入该 stream。该复用样本跳过参考图与条件前缀编码后，目标图预处理承接了此前参数汇集和复制的等待。trace 将主机 API 时间与 GPU 实际复制时间分别记录，支持将这项阶段增长解释为等待位置的变化；主性能结论使用完整训练时间。条件阶段的主机计时可以包含后续仍然需要等待的通信，因此 \(rf\) 使用条件分支占比估算节省时间时存在这一适用限制。

该复制事件标记为 `Pageable -> Device`。已完整阅读的 [CUDA 12.9 API 同步说明](https://docs.nvidia.com/cuda/archive/12.9.0/cuda-runtime-api/api-sync-behavior.html)规定，涉及 pageable host memory 的异步复制可以在主机侧等待，准备 pinned memory 中转缓冲区时也可能同步 stream。pageable 表示操作系统可以调度内存页面；pinned 表示页面固定在物理内存中。此处较长的主机等待直接发生在 `cudaMemcpyAsync` 调用内。

rank 1 复用样本中的 NumPy `array` 调用持续 176.838 ms，在线样本为 6.385 ms。以该复用样本的函数开始时间为 0，参数 AllGather 在 98.021 ms 完成，参数复制在 101.998 ms 完成，`cudaMemcpyAsync` 在 177.907 ms 开始，持续 0.589 ms。该 trace 将主要主机耗时定位到图像转换调用，未采集到 NumPy 内部与操作系统调度的详细事件，无法继续确定这段调用时间的具体组成。两次独立采集显示不同的主机耗时位置；profiler 会影响时序，正式性能数据使用 B2/C1 三轮未开启 profiler 的训练。

两种 rank 选择及单步骤参数均已在镜像内执行通过，日志为 [rank 0](../raw/task17-a100-experiments-20261114/preparation/profile-selection-single-rank0.log)、[rank 1](../raw/task17-a100-experiments-20261114/preparation/profile-selection-single-rank1.log)。两步窗口的[双 rank 资源记录](../raw/task17-a100-experiments-20261114/preparation/profile-resource-comparison.json)与[单 rank 资源记录](../raw/task17-a100-experiments-20261114/preparation/profile-rank0-resource-summary.json)分别记录主机使用峰值 30970122240、31873994752 bytes；两个诊断任务均触发主机内存限制，状态为 `FAILED`，原始产物完整保留。手动 Serve 清理后的[第一份快照](../raw/task17-a100-experiments-20261114/preparation/c2-profile-cleanup-snapshot.json)与[第二份快照](../raw/task17-a100-experiments-20261114/preparation/c2-profile-rank0-cleanup-snapshot.json)均确认其 actor、placement group 和 GPU 计算进程已释放；这些记录单独保存，不纳入 F 组的自动释放证据。

[实际 JobSupervisor 环境](../raw/task17-a100-experiments-20261114/preparation/rank-profile-runtime.json)已核查：`RAY_JOB_STOP_WAIT_TIME_S=60`、`RELAX_PROFILE_GZIP_LEVEL=1`、`TMPDIR=/work/tmp`。采集文件使用 gzip level 1，临时 JSON 使用容器挂载的工作目录。

## 资源释放记录

D1 两组、E1 两组、E2 三组及九组 B1、B2、C1，共 16 次正常训练的[释放汇总](../raw/task17-a100-experiments-20261114/preparation/normal-release-summary.json)已在本地重新核查。各次任务的 8 项检查全部通过，训练 actor 与对应 JobSupervisor 均已退出，placement group 已移除，GPU 计算进程、共享内存及实验目录检查通过。任务结束至全部查询完成的时间上界为 9.0751–20.3533 秒，该数值包含采样等待与查询时间。

实际命令使用本地 Python 3.14.10，执行目录为 `/home/xbei/Relax`：

```bash
python Task17/bounded_condition_cache/real_run/summarize_release.py \
  Relax/output/task17-a100-experiments-20261114/a100-B1-r*-20261114 \
  Relax/output/task17-a100-experiments-20261114/a100-B2-r*-20261114 \
  Relax/output/task17-a100-experiments-20261114/a100-C1-r*-20261114 \
  Relax/output/task17-a100-experiments-20261114/a100-D1-*-20261114 \
  Relax/output/task17-a100-experiments-20261114/a100-E*-20261114 \
  --output Relax/output/task17-a100-experiments-20261114/preparation/normal-release-summary.json
```

F1 三次取消实验已完成。`ray job stop` 均在观察到第 40 步日志后发出，服务器确认接受请求，最终状态为 `STOPPED`，完整日志各记录 40 步。各次八项检查全部通过，两张 GPU 均从训练前的 14 MiB 恢复至 14 MiB。两名 `FSDPTrainRayActor` 的退出记录显示 placement group 被移除后由 Ray 终止，相关 Serve actor 及 JobSupervisor 均已退出，两个 placement group 状态为 `REMOVED`。

| 运行 | 训练资源释放上界 / s | 包含 JobSupervisor 的完整释放上界 / s |
| --- | --- | --- |
| F1-r1 | 5.8472 | 63.9241 |
| F1-r2 | 8.0382 | 65.9377 |
| F1-r3 | 11.4678 | 64.0997 |

原始目录均已复制到本地，[独立审查](../raw/task17-a100-experiments-20261114/preparation/F1-audit-local.json)与[释放汇总](../raw/task17-a100-experiments-20261114/preparation/F1-release-summary.json)均通过。训练资源释放时，GPU、训练进程、共享内存、实验文件、placement group 及本次训练所属 actor 均满足检查条件；完整释放还要求 JobSupervisor 退出和任务状态终止。三次 JobSupervisor 日志均已保存：[第一次](../raw/task17-a100-experiments-20261114/preparation/F1-r1-supervisor.log)、[第二次](../raw/task17-a100-experiments-20261114/preparation/F1-r2-supervisor.log)、[第三次](../raw/task17-a100-experiments-20261114/preparation/F1-r3-supervisor.log)。各次均记录等待 SIGTERM 60 秒后使用 SIGKILL，完整释放上界包含这一等待。每次释放检查后，后续实验均正常启动。

F2 三次前向异常实验已完成，[独立审查](../raw/task17-a100-experiments-20261114/preparation/F2-audit-local.json)与[释放汇总](../raw/task17-a100-experiments-20261114/preparation/F2-release-summary.json)均通过。各次故障事件均记录 `step=40`、`micro=1`、`rank=0`、`phase=forward`；完整日志各记录前 39 步，包含指定异常消息，最终状态为 `FAILED`，driver 退出码为 1。各次两个训练 actor 的退出原因为 `OWNER_DIED`，owner 的退出类型为 `INTENDED_SYSTEM_EXIT`；训练 actor 由 Ray 随 owner 退出而终止。各次八项检查全部通过，两张 GPU 均恢复至 14 MiB。

| 运行 | rank 0 故障 PID | rank 1 PID | 训练资源释放上界 / s | 完整释放上界 / s |
| --- | --- | --- | --- | --- |
| F2-r1 | 335876 | 336181 | 6.1117 | 11.4434 |
| F2-r2 | 345365 | 345678 | 3.8993 | 9.1524 |
| F2-r3 | 354704 | 355006 | 3.6394 | 8.9625 |

三次原始产物及 JobSupervisor 日志均已保存：[第一次](../raw/task17-a100-experiments-20261114/preparation/F2-r1-supervisor.log)、[第二次](../raw/task17-a100-experiments-20261114/preparation/F2-r2-supervisor.log)、[第三次](../raw/task17-a100-experiments-20261114/preparation/F2-r3-supervisor.log)。

F3 三次 backward 异常已完成，[独立审查](../raw/task17-a100-experiments-20261114/preparation/F3-audit-local.json)与[释放汇总](../raw/task17-a100-experiments-20261114/preparation/F3-release-summary.json)均通过。各次故障事件均记录 `step=40`、`micro=1`、`rank=0`、`phase=backward`；日志各记录前 39 步并包含指定异常，最终 `FAILED`，driver 退出码为 1。各次两个训练 actor 的退出原因均为 `OWNER_DIED`，owner 退出类型为 `INTENDED_SYSTEM_EXIT`。各次八项资源检查全部通过，两张 GPU 均恢复至 14 MiB。

| 运行 | rank 0 故障 PID | rank 1 PID | 训练资源释放上界 / s | 完整释放上界 / s |
| --- | --- | --- | --- | --- |
| F3-r1 | 364049 | 364367 | 9.1166 | 9.1166 |
| F3-r2 | 373493 | 373795 | 11.7531 | 11.7531 |
| F3-r3 | 382903 | 383214 | 8.5662 | 8.5662 |

三次原始产物及 JobSupervisor 日志均已保存：[第一次](../raw/task17-a100-experiments-20261114/preparation/F3-r1-supervisor.log)、[第二次](../raw/task17-a100-experiments-20261114/preparation/F3-r2-supervisor.log)、[第三次](../raw/task17-a100-experiments-20261114/preparation/F3-r3-supervisor.log)。F 组完整的[九次审查](../raw/task17-a100-experiments-20261114/preparation/F-audit-local.json)与[九次释放记录](../raw/task17-a100-experiments-20261114/preparation/F-release-summary.json)均已在本地验证；每次完成检查后，下一次实验均正常启动。

## G：适用范围

[主配置参考](../raw/task17-a100-experiments-20261114/preparation/scope-main.json)使用 B2、C1 各三次运行的第 33–96 步，B1 相同窗口提供条件分支计时。共同复用率 \(r=0.3359375\)，条件分支占单步时间的比例 \(f=0.3488303\)。忽略缓存控制成本和样本编码耗时差异时，单步时间减少比例近似为 \(rf\)，吞吐提高比例为 \(rf/(1-rf)\)。该窗口的预测吞吐提升为 13.2740%，实测为 11.9436%，实测单步时间减少 10.6693%。

共同复用要求两个 rank 在同一次 micro-batch 中均满足复用条件。设各 rank 的本地命中事件为 \(H_0,H_1\)，共同复用率为 \(r=P(H_0\cap H_1)=P(H_0)P(H_1\mid H_0)\)。样本分配和淘汰顺序可能使两者相关，全部计算使用实际共同计数；本地命中率分别记录每个 rank 自身持有条目的比例。

G1 的 256 行配对实验各完成 192 步，状态均为 `SUCCEEDED`，八项资源检查全部通过，完整释放上界分别为 17.1544、17.0836 秒。[关闭缓存审查](../raw/task17-a100-experiments-20261114/preparation/G1-n256-off-audit-local.json)与[开启缓存审查](../raw/task17-a100-experiments-20261114/preparation/G1-n256-on-audit-local.json)通过，完整原始产物已复制到本地。[配对计算](../raw/task17-a100-experiments-20261114/preparation/scope-G1-n256.json)使用第三个 epoch，即第 129–192 步。

| 指标 | 缓存关闭 | 缓存开启 |
| --- | --- | --- |
| 首步耗时 / s | 13.9238 | 14.4441 |
| 提交至首步 / s | 346.8434 | 351.2399 |
| 第一个 epoch 单步均值，排除首步 / s | 9.64460 | 9.69496 |
| 第二个 epoch 单步均值 / s | 9.65027 | 9.55274 |
| 第三个 epoch 单步均值 / s | 9.66023 | 9.47455 |
| 192 步训练累计 / s | 1857.4055 | 1842.9734 |

第三个 epoch 中，各 rank 的 128 次请求均有 7 次共同复用，\(r=5.46875\%\)；本地命中率分别为 15.625%、14.0625%，淘汰次数分别为 111、110。条件分支比例 \(f=34.7568\%\)，预测吞吐提高 1.9376%，实测提高 1.9598%，单步时间减少 1.9221%。该组各执行一次，数值为本次配对观测；主配置的三轮变化范围提供测量波动参考。

[逐 epoch 资源记录](../raw/task17-a100-experiments-20261114/preparation/G1-n256-epoch-resources.json)检查了两个 rank 的全部 192 步，计费均未超过 12 GiB；计费峰值分别为 12884877312、12855115776 bytes，完整运行各有 262、265 次淘汰。第一个 epoch 各有 128 次请求且没有共同复用。192 步训练累计时间减少 0.7770%，第一个 epoch 的训练累计时间增加 3.6932 秒，即 0.5942%。

512 行数据的配对实验各完成 384 步，状态均为 `SUCCEEDED`。[关闭缓存审查](../raw/task17-a100-experiments-20261114/preparation/G1-n512-off-audit-local.json)与[开启缓存审查](../raw/task17-a100-experiments-20261114/preparation/G1-n512-on-audit-local.json)通过，八项资源释放检查全部通过，完整释放上界分别为 18.0621、16.8591 秒。两份数据选择记录逐字节相同，生效配置的差异为缓存开关与统计输出目录。[配对计算](../raw/task17-a100-experiments-20261114/preparation/scope-G1-n512.json)使用第三个 epoch，即第 257–384 步。

| 指标 | 缓存关闭 | 缓存开启 |
| --- | --- | --- |
| 首步耗时 / s | 14.6923 | 13.8043 |
| 提交至首步 / s | 366.5916 | 359.1339 |
| 第一个 epoch 单步均值，排除首步 / s | 9.68057 | 9.70178 |
| 第二个 epoch 单步均值 / s | 9.68993 | 9.67506 |
| 第三个 epoch 单步均值 / s | 9.69190 | 9.70520 |
| 384 步训练累计 / s | 3724.9990 | 3726.6037 |

第三个 epoch 中，各 rank 的 256 次请求仅有 1 次共同复用，\(r=0.390625\%\)；本地命中率分别为 2.34375%、3.515625%，淘汰次数分别为 241、255。条件分支比例 \(f=34.9124\%\)，近似式预测吞吐提高 0.1366%；实测吞吐变化为 −0.1370%，单步时间增加 0.1372%。该近似式没有计入缓存身份、版本检查及控制通信等开销，实测净变化以完整训练计时为依据。该数据量各执行一次，这一变化幅度小于主配置三次重复的观测范围。

全程训练累计时间增加 0.0431%；第一个 epoch 累计时间增加 1.8056 秒，即 0.1451%。[逐 epoch 资源记录](../raw/task17-a100-experiments-20261114/preparation/G1-n512-epoch-resources.json)核查了两个 rank 的全部 384 步，缓存计费均未超过 12 GiB，峰值分别为 12851134464、12884877312 bytes；完整运行各有 666、672 次淘汰。GPU 整体占用采样峰值分别为 39070、39310 MiB。全部原始产物已复制到本地。

G2 使用 128 行数据、三个 epoch，关闭追踪器，并以 B2 三次运行的第 33–96 步均值 9.63100 秒作为对照；条件分支比例由 B1 相同窗口取得，\(f=34.8830\%\)。本地命中率与共同复用率对应第 33–96 步，淘汰次数对应完整 96 步。2、4、8 GiB 各执行一次，12 GiB 参考来自 C1 三次运行，三次缓存计数相同。使用 `diff -u` 比较各组 `effective-env.txt`，与 C1-r1 的差异均仅为 `CONDITION_CACHE_MAX_BYTES`；`cmp` 检查三份 `data-selection.json` 与 C1-r1 逐字节相同。

| 预算 / GiB | 本地命中率，rank 0 / 1 | 共同复用率 | 单步均值 / s | 实测吞吐变化 | 近似式预测变化 | 全程淘汰次数，rank 0 / 1 |
| --- | --- | --- | --- | --- | --- | --- |
| 2 | 0.78125% / 2.34375% | 0% | 9.76304 | −1.3524% | 0% | 177 / 174 |
| 4 | 4.6875% / 9.375% | 1.5625% | 9.65647 | −0.2637% | +0.5480% | 158 / 149 |
| 8 | 24.21875% / 21.875% | 7.03125% | 9.45039 | +1.9111% | +2.5144% | 109 / 107 |
| 12，C1 参考 | 53.125% / 53.90625% | 33.59375% | 8.60343 | +11.9436% | +13.2740% | 42 / 37 |

2 GiB 预算的[独立审查](../raw/task17-a100-experiments-20261114/preparation/G2-b2-audit-local.json)与[性能计算](../raw/task17-a100-experiments-20261114/preparation/scope-G2-b2.json)均通过，96 步状态为 `SUCCEEDED`，八项释放检查全部通过，完整释放上界 11.9642 秒。两个 rank 的计费峰值分别为 2129338368、2129264640 bytes，均低于 2147483648 bytes；最多分别保存 18、16 个条目，条目超过预算的跳过次数均为 0。全程没有共同复用，96 步训练累计 944.0682 秒，相对 B2 三次均值增加 1.5369%。该数值为本次预算配置相对三次基线均值的观测结果。

4 GiB 预算的[独立审查](../raw/task17-a100-experiments-20261114/preparation/G2-b4-audit-local.json)与[性能计算](../raw/task17-a100-experiments-20261114/preparation/scope-G2-b4.json)均通过，96 步状态为 `SUCCEEDED`，八项释放检查全部通过，完整释放上界 12.9901 秒。两个 rank 的计费峰值分别为 4294803456、4294729728 bytes，均低于 4294967296 bytes；最多分别保存 31、33 个条目，条目超过预算的跳过次数均为 0。每个 rank 的 128 次后续请求中有 2 次共同复用，96 步训练累计 933.9370 秒，相对 B2 三次均值增加 0.4472%。

8 GiB 预算的[独立审查](../raw/task17-a100-experiments-20261114/preparation/G2-b8-audit-local.json)与[性能计算](../raw/task17-a100-experiments-20261114/preparation/scope-G2-b8.json)均通过，96 步状态为 `SUCCEEDED`，八项释放检查全部通过，完整释放上界 11.1152 秒。两个 rank 的计费峰值分别为 8589434880、8551440384 bytes，均低于 8589934592 bytes；最多分别保存 60、59 个条目，条目超过预算的跳过次数均为 0。每个 rank 的 128 次后续请求中有 9 次共同复用，96 步训练累计 919.4505 秒，相对 B2 三次均值减少 1.1108%。三个预算点的完整原始产物均已复制到本地。

G3 两组各完整完成六个 epoch、192 步，状态均为 `SUCCEEDED`，[缓存关闭审查](../raw/task17-a100-experiments-20261114/preparation/G3-off-audit-local.json)、[缓存开启审查](../raw/task17-a100-experiments-20261114/preparation/G3-on-audit-local.json)及[逐 epoch 资源检查](../raw/task17-a100-experiments-20261114/preparation/G3-epoch-resources.json)均通过。两组生效配置的差异为缓存开关和统计输出目录，数据选择记录逐字节相同。各 epoch 的单步均值及共同复用率如下，第一个 epoch 的均值排除首步。

| epoch | 缓存关闭单步均值 / s | 缓存开启单步均值 / s | 共同复用率 |
| --- | --- | --- | --- |
| 1 | 9.67355 | 9.69036 | 0% |
| 2 | 9.66513 | 8.62480 | 32.8125% |
| 3 | 9.67225 | 8.59522 | 34.375% |
| 4 | 9.66034 | 8.94732 | 23.4375% |
| 5 | 9.68802 | 8.55819 | 35.9375% |
| 6 | 9.66237 | 8.40883 | 40.625% |

[第六个 epoch 配对计算](../raw/task17-a100-experiments-20261114/preparation/scope-G3.json)使用第 161–192 步：每个 rank 的 64 次请求有 26 次共同复用，本地命中率分别为 59.375%、62.5%，各淘汰 25 次。条件分支占比 \(f=34.6436\%\)，近似式预测吞吐提高 16.3791%，实测提高 14.9074%，单步时间减少 12.9734%。两组各执行一次，这些数值为本次配对观测。

首步耗时分别为 14.6656、14.0893 秒，提交至首步分别为 361.9533、360.6370 秒；192 步训练累计时间由 1861.6852 秒减少至 1694.7898 秒，减少 8.9647%。完整运行中，缓存开启组每个 rank 各有 384 次请求、107 次共同复用，淘汰次数为 128、124；条目超过预算的跳过次数均为 0。

两个 rank 的全部逐步缓存记录均符合 12 GiB 预算，计费峰值分别为 12882198528、12846268416 bytes，最多保存 86、88 个条目。缓存开启时的单步已分配显存峰值分别为 35.58706、35.57702 GiB；reserved 显存峰值分别为 37.36914、37.54492 GiB。rank 1 的 reserved 显存从第一个 epoch 的 37.36914 GiB 增至第六个 epoch 的 37.54492 GiB；GPU 整体占用采样峰值为 38930、39110 MiB。

缓存关闭组的单步已分配显存峰值均为 23.58991 GiB，reserved 显存峰值均为 37.36719 GiB。两个 rank 在第一个 epoch 的当前 RSS 峰值分别为 2.59849、2.60408 GiB，第六个 epoch 为 2.61393、2.61905 GiB。缓存开启组对应值由 2.72292、2.73246 GiB 增至 2.74342、2.74898 GiB。当前 RSS 的增长与缓存计费分别记录，六个 epoch 的有限观测用于本次容量和生命周期验证。两组八项释放检查均通过，完整释放上界分别为 16.7302、17.2674 秒，全部原始产物已复制到本地。

G4 两组在 `MAX_PIXELS=262144` 下各完整完成 96 步，状态均为 `SUCCEEDED`，纳入[完整审查](../raw/task17-a100-experiments-20261114/preparation/complete-plan-audit-local.json)。两组生效配置的差异为缓存开关与统计输出目录，数据选择记录逐字节相同。八项释放检查均通过，完整释放上界分别为 15.2404、17.1037 秒。

| 指标 | 缓存关闭 | 缓存开启 |
| --- | --- | --- |
| 首步耗时 / s | 13.6898 | 13.7925 |
| 提交至首步 / s | 359.1340 | 353.3958 |
| 第一个 epoch 单步均值，排除首步 / s | 9.27671 | 9.28982 |
| 第二个 epoch 单步均值 / s | 9.25330 | 8.18247 |
| 第三个 epoch 单步均值 / s | 9.27752 | 7.59743 |
| 96 步训练累计 / s | 894.2543 | 806.7335 |

[第三个 epoch 配对计算](../raw/task17-a100-experiments-20261114/preparation/scope-G4.json)使用第 65–96 步：每个 rank 的 64 次请求有 36 次共同复用，\(r=56.25\%\)，本地命中率为 75%、71.875%，淘汰次数均为 0。条件分支占比 \(f=34.9746\%\)，近似式预测吞吐提高 24.4915%，实测提高 22.1140%，单步时间减少 18.1093%；全程累计训练时间减少 9.7870%。两组各执行一次，这些数值为本次配对观测。

[逐 epoch 容量检查](../raw/task17-a100-experiments-20261114/preparation/G4-epoch-resources.json)通过。两个 rank 的计费峰值分别为 9603661824、9783447552 bytes，最多保存 108、110 个条目；完整运行各完成 59 次共同复用，淘汰与条目超过预算的跳过次数均为 0。缓存开启时单步已分配显存峰值分别为 32.43300、32.68331 GiB，reserved 显存峰值均为 37.36719 GiB。

[实际尺寸检查](../raw/task17-a100-experiments-20261114/preparation/G4-dimensions.json)覆盖两组各 384 次 micro-batch：目标图均为 512×512，每张目标图搬运 1572864 bytes；B1-r1 对应的全部目标图为 1024×1024，每张搬运 6291456 bytes。G4 开启缓存后，参考图编码及条件前缀编码各由 384 次减少至 266 次；参考图像素搬运由 576 MiB 减少至 399 MiB，目标图搬运均为 576 MiB，全部记录的搬运量由 1153.5200 MiB 减少至 976.0533 MiB。完整计时与调用数据见[关闭缓存阶段记录](../raw/task17-a100-experiments-20261114/preparation/G4-off-stages.json)及[开启缓存阶段记录](../raw/task17-a100-experiments-20261114/preparation/G4-on-stages.json)，两组原始产物均已复制到本地。

## 结果分析与验收

本节的时间与缓存计数由 [recompute_a100.py](../scripts/recompute_a100.py) 直接读取 24 次运行的 TensorBoard 事件、`effective-env.txt` 与任务日志重新计算，结果保存为 [independent-recompute-20261116.json](../raw/task17-a100-experiments-20261114/preparation/independent-recompute-20261116.json)（SHA256 `39d843203eabed68ba6482676f56e0b87817c48393abde3081a475a01ff91eca`），与上文各节的单步均值、总训练时间和缓存计数逐项一致。2026-11-16（Asia/Shanghai）在 `/home/xbei/Relax` 使用本地 Python 3.13.15 与 TensorBoard 2.21.0 执行：

```bash
R=Relax/output/task17-a100-experiments-20261114
$R/preparation/venv/bin/python -I Task17/bounded_condition_cache/real_run/recompute_a100.py $R \
  a100-B2-r{1,2,3}-20261114 a100-C1-r{1,2,3}-20261114 a100-B1-r{1,2,3}-20261114 \
  a100-G1-n256-{off,on}-20261114 a100-G1-n512-{off,on}-20261114 a100-G2-b{2,4,8}-20261114 \
  a100-G3-{off,on}-20261114 a100-G4-{off,on}-20261114 \
  a100-D1-on-20261114 a100-E1-on-20261114 a100-E2-on-on-20261114 a100-E2-off-on-20261114 \
  > $R/preparation/independent-recompute-20261116.json
```

数值对照文件 `parameter-parity.json` 与八份 `compare-*.json`、`cache-semantics.json`、`complete-plan-audit-local.json`、A1 的 JUnit 及 F 组 `status.txt` 也已逐项读取核对。

### 性能收益

主配置稳态吞吐提高 12.2309%，满足计划的 5% 要求；第二个 epoch 提高 11.6573%，96 步训练总时间减少 6.9791%。第一个 epoch 不含首步时，C1 单步均值 9.69642 秒，B2 为 9.64537 秒，耗时增加 0.53%，小于 B2 三次之间 9.60185–9.68802 秒的范围（约 0.9%）。每次请求的查找、版本检查和控制通信合计约 15 ms，约为一个 micro-batch 计时的 0.3%。

A100 主配置的收益低于 4090 记录的 22.09%，原因是缓存容量：12 GiB 按平均条目大小约可保存 84 个条目，而相同数据、采样与种子下，4090 每个 rank 在三个 epoch 内保存的条目峰值为 108、110 个且没有淘汰；A100 两个 rank 分别淘汰 42、37 个条目，第 2–3 个 epoch 的共同复用率为 33.59375%；4090 每个 rank 16 GiB，没有淘汰，共同复用率为 46.1%。G4 在 `MAX_PIXELS=262144` 下全部条目可以保存，计费峰值约 9.6 GiB、无淘汰，第三个 epoch 共同复用率 56.25%，吞吐提高 22.1140%。

### 数值一致性与证据范围

D1 的 150528 条逐参数梯度与 FP32 更新量记录、9408 个完整张量，以及 D1、E1、E2 八组对照的逐步 loss、grad_norm、lr、2352 个 checkpoint 张量和 6 个 RNG 张量全部逐元素一致，最大误差为 0。D1-on 每个 rank 在第 2 个 epoch 完成 21 次共同复用，复用路径的数值一致性由 D1 直接证明。

E1-on 每个 rank 的 5 次共同复用全部发生在第 33–40 步，即编码器更新之前；第 41–64 步的 48 次请求全部在线编码，没有发生复用。E2-on-on 与 E2-off-on 第 33–64 步的每个 rank 64 次请求中，本地命中与共同复用均为 0：恢复后缓存为空，第 2 个 epoch 中每个样本只出现一次。E1 与 E2 因此证明版本更新清空旧条目并重新编码、恢复后从空缓存开始，以及这两种情况下训练数值与缓存关闭完全一致；更新或恢复之后再次使用新条目复用的路径没有在完整规模运行中出现，需要训练进入第 3 个 epoch 才会覆盖。

### 适用范围

各配置的条件分支占比 \(f\) 均约为 35%（34.6436%–34.9746%），收益主要由共同复用率 \(r\) 决定。\(r\) 取决于每个 rank 已接触的条件能否保存在缓存中：128 行数据下，预算从 12 GiB 降到 8 GiB，\(r\) 由 33.59375% 降至 7.03125%；预算 12 GiB 时，数据增加到 256、512 行，第三个 epoch 的 \(r\) 分别为 5.46875%、0.390625%。训练轮数增加时 \(r\) 总体上升，G3 第 2、3、5、6 个 epoch 分别为 32.8125%、34.375%、35.9375%、40.625%；第 4 个 epoch 为 23.4375%，低于相邻轮次，单次运行未区分其组成。

近似式 \(rf/(1-rf)\) 在复用率较高时比测量值高 1.2–2.4 个百分点（主配置 13.2740% 对 11.9436%，G3 16.3791% 对 14.9074%，G4 24.4915% 对 22.1140%）。C2 的 trace 显示条件阶段主机计时包含 FSDP 参数汇集的等待，跳过条件编码后该等待转移到目标图预处理，条件阶段计时的减少量不会全部成为训练时间的节省。

复用率接近零时（G1 512 行、G2 4 GiB），全程训练时间变化为 +0.0431%、+0.4472%，与每次请求约 0.3% 的缓存开销相符。G2 2 GiB 全程增加 1.5369%，944.0682 秒超出 B2 三次 923.8252–934.7586 秒的范围；该运行第一个 epoch 不含首步的单步均值 9.84855 秒，比 C1 三次均值 9.69642 秒高 1.57%，第 2–3 个 epoch 比 B2 均值高 1.37%。第一个 epoch 没有复用，各预算之间的差异只在淘汰次数（4 GiB、8 GiB 同期比 C1 高 0.33%、0.04%），该运行在各 epoch 均较慢；该点只执行一次，不能区分机器波动与频繁淘汰的影响。

### 对照实验计划的验收

| 计划项 | 结论 | 依据与说明 |
| --- | --- | --- |
| A1 | 满足 | JUnit 230 项通过、0 项跳过 |
| A2、A3 | 满足 | B 取 12 GiB；64 步内无显存不足，GPU 占用采样峰值 38930 MiB |
| B1、B2（R1） | 满足 | 三轮统计逐项一致；追踪器总时间开销 0.1726%，低于 2% |
| C1（R6、R7） | 满足 | 稳态吞吐 +12.2309%，配对范围 10.7169%–13.2436%；冷启动、构建成本与回本样本量已报告 |
| C2 | 部分满足 | 阶段统计、编码次数与搬运字节减少已完成；rank 0 的目标图预处理增长定位为参数汇集等待转移到 `cudaMemcpyAsync`，rank 1 的 176.838 ms 位于 NumPy `array` 调用，trace 未覆盖其内部组成，原因未确定 |
| D1（R5） | 满足 | 逐参数梯度、FP32 更新量、完整张量、checkpoint 与 RNG 全部一致，复用发生在比较窗口内 |
| E1（R3） | 满足，证据范围见上 | 判定条件全部成立；更新后的第 41–64 步没有复用，更新后条目被再次复用的路径未出现 |
| E2（R3、R4） | 满足，证据范围见上 | 三种组合的数值全部一致，恢复后从空缓存开始；第 33–64 步没有复用 |
| F1–F3（R8） | 满足 | 九次状态、故障位置与资源释放全部通过；取消时 driver 未在 SIGTERM 后 60 秒内退出，完整释放包含 JobSupervisor 的 SIGKILL 等待，训练资源在 5.8472–11.4678 秒内释放 |
| G1–G4（R6） | 满足 | 四组全部完成，近似式与测量值的关系已核对；G2 2 GiB 单次运行整体较慢，见上 |
| 第 5 节第 8 条（归档） | 满足 | 1830 个文件逐个 SHA256 通过，12 份 checkpoint 全部保存在本地 |

按任务要求，除第二种模型外，R1、R3–R9 在 A100 上均有对应证据，R6 的 SenseNova 部分达到要求；GPU 数量变化的影响按决定未测量。尚未覆盖的内容为 C2 中 rank 1 的计时组成、E1 与 E2 之后再次复用的路径，以及 G2 2 GiB 单次运行较慢的原因。

## 统一审查与产物

[本地完整审查](../raw/task17-a100-experiments-20261114/preparation/complete-plan-audit-local.json)与[服务器完整审查](../raw/task17-a100-experiments-20261114/a100-orchestration-20261114/experiment-audit.json)均为 `complete_plan=true`、`runs_checked=39`，两份 JSON 经 `cmp` 检查逐字节相同。审查逐项核对固定代码版本、无未提交修改、预期任务状态、完整步骤、TensorBoard、资源采样及释放结果；七份 G 组配对计算的本地与服务器结果也全部逐字节相同。A1 的 230 项回归及两个 profiler 采集的独立审查分别保留。

资源曲线保存于[曲线目录](../raw/task17-a100-experiments-20261114/a100-orchestration-20261114/resource-curves)，包含六组 SVG 和六组 PNG，PNG 尺寸均为 2080×1760。[采样清单](../raw/task17-a100-experiments-20261114/a100-orchestration-20261114/resource-curves/resource-curves.json)中，A3、C1 三轮、G3-off、G3-on 的每张 GPU 采样次数依次为 188、250、253、252、461、428。A3 与 G3 同时提供逐步缓存容量和条目曲线；C1 关闭追踪器，曲线记录系统资源，其缓存容量依据最终累计峰值。SVG 经 XML 解析检查、PNG 经 Pillow `verify()` 检查，全部通过；证据为 [SVG 检查](../raw/task17-a100-experiments-20261114/preparation/resource-svg-validation.json)和 [PNG 检查](../raw/task17-a100-experiments-20261114/preparation/resource-png-validation.json)。

[完整归档校验](../raw/task17-a100-experiments-20261114/preparation/full-archive-verification-20261115.log)于 2026-11-15（Asia/Shanghai）使用 `rsync -anzc --itemize-changes --stats` 比较服务器与本地，未设置目录排除规则，覆盖 1830 个普通文件、347 个目录、48 个符号链接。普通文件合计 23814277111 bytes；包含符号链接目标文本的 rsync 总量为 23814281191 bytes。新增文件、待传输普通文件及待传输字节均为 0，未报告文件或目录差异。[服务器 SHA256 清单](../raw/task17-a100-experiments-20261114/preparation/full-server-sha256-20261115.txt)中的 1830 个文件在本地执行 `sha256sum -c` 全部通过，退出码为 0，完整证据见[逐文件校验日志](../raw/task17-a100-experiments-20261114/preparation/full-local-sha256-verification-20261115.log)。

[checkpoint 清单](../raw/task17-a100-experiments-20261114/preparation/checkpoint-archive-inventory-20261115.json)包含 12 份 checkpoint、108 个内部文件、14763927057 bytes：D1-off/on 各有第 32、64 步，E1-off/on 和 E2 三组各有第 64 步，另有三份第 32 步独立恢复副本。每份 checkpoint 的 `COMMITTED`、模型分片及 `.metadata`、两个 rank 的 RNG、`trainer_state.json`、`adapter_contract.json` 和 `weight_sync_manifest.json` 均已保存；JSON 解析检查通过。全部路径位于[本地产物根目录](../raw/task17-a100-experiments-20261114)下，对应原实验目录结构。48 个测试符号链接保留原容器路径，其目标内容均已保存在对应本地目录。

任务提交与准备目录另外完整保存为 [server-jobs](../raw/task17-a100-experiments-20261114/server-jobs)，53 个文件、15089922 bytes 的 SHA256 [全部通过](../raw/task17-a100-experiments-20261114/preparation/server-jobs-verification-20261115.log)；辅助工具测试的 14 个文件保存在 [server-tool-tests](../raw/task17-a100-experiments-20261114/server-tool-tests)，SHA256 [全部通过](../raw/task17-a100-experiments-20261114/preparation/server-tool-tests-verification-20261115.log)。本地准备、分析结果及原有 `server-job-logs/` 同时保留。原始模型与数据使用本报告记录的固定公开版本；本次训练生成的 checkpoint 和全部实验产物已完整保存。

本次归档在本地 Linux、Python 3.14.10、GNU SHA256 工具及 rsync 环境执行，实验代码版本保持 `9783ebe924323c9742fbd735fb850f93a0a157f0`。最后一次完整同步退出码为 0，日志见[同步记录](../raw/task17-a100-experiments-20261114/preparation/full-archive-sync-20261115.log)。实际同步和复核命令如下，校验命令在本地产物根目录执行：

```bash
env LD_LIBRARY_PATH=/home/xbei/Relax/Task17/.work/usr/lib \
  /home/xbei/Relax/Task17/.work/usr/bin/rsync -azc --stats \
  --log-file=/home/xbei/Relax/Relax/output/task17-a100-experiments-20261114/preparation/full-archive-sync-20261115.log \
  -e 'ssh -F /dev/null -S /home/xbei/Relax/Task17/.work/a100-control -p <服务器端口>' \
  root@<服务器地址>:/root/task17/experiments/ \
  /home/xbei/Relax/Relax/output/task17-a100-experiments-20261114/
sha256sum -c preparation/full-server-sha256-20261115.txt \
  > preparation/full-local-sha256-verification-20261115.log
```

完整目录复核使用相同源路径、目标路径与 SSH 参数，将同步选项设置为 `-anzc --itemize-changes --stats`。2026-11-15 00:00:25（Asia/Shanghai）的[进程与 GPU 核查](../raw/task17-a100-experiments-20261114/preparation/full-archive-process-check-20261115.txt)确认实验训练和监控进程全部结束，两张 GPU 均为 14 MiB、利用率为 0%。本次实验产物已具备完整本地副本，可以退租该实例；未执行实例退租操作。

71 个实验工具及配置文件的 [SHA256 清单](../raw/task17-a100-experiments-20261114/preparation/experiment-tools-sha256.txt)已在本地与服务器分别核对通过。本地和服务器的 Relax HEAD 均为 `9783ebe924323c9742fbd735fb850f93a0a157f0`，工作区均无未提交修改。持续主机采样进程 `51990` 与顺序实验进程 `587288` 均已结束，完整主机日志保存在 [host-monitor.log](../raw/task17-a100-experiments-20261114/a100-preparation-20261114/host-monitor.log)；[结束时 GPU 记录](../raw/task17-a100-experiments-20261114/preparation/final-gpu.txt)确认两张卡均为 14 MiB。

最终统一审查在本地 Python 3.13.15、TensorBoard 环境执行，实际命令如下；资源及尺寸统计使用本地 Python 3.14.10。全部命令在 `/home/xbei/Relax` 执行。

```bash
env TMPDIR=/home/xbei/Relax/Task17/.work/tmp \
  Relax/output/task17-a100-experiments-20261114/preparation/venv/bin/python \
  Task17/bounded_condition_cache/real_run/audit_a100.py \
  --root Relax/output/task17-a100-experiments-20261114 \
  --date 20261114 --budget-gib 12 \
  --commit 9783ebe924323c9742fbd735fb850f93a0a157f0 \
  --output Relax/output/task17-a100-experiments-20261114/preparation/complete-plan-audit-local.json
python Task17/bounded_condition_cache/real_run/summarize_epoch_resources.py \
  Relax/output/task17-a100-experiments-20261114/a100-G3-off-20261114 \
  Relax/output/task17-a100-experiments-20261114/a100-G3-on-20261114 \
  --epochs 6 --budget-gib 12 \
  --output Relax/output/task17-a100-experiments-20261114/preparation/G3-epoch-resources.json
```

## 执行入口

[run_a100.py](../scripts/run_a100.py) 按 `--stage A2/A3/D/E/BC/C2/F/G` 执行对应组别，接收 `--date` 和 `--budget-gib`；每次运行使用独立目录。`C2` 阶段读取已经完成的九次 B1、B2、C1 参考及 C2 阶段统计，分别执行 `C2-profile-single-rank0`、`C2-profile-single-rank1` 并核查 trace、双 rank 样本与运行完整性。D、E 组执行后自动比较训练结果。F 组按预期状态检查，并每 5 秒采样资源，最长观察 600 秒。

[run_experiment.sh](../scripts/run_experiment.sh) 保存实际环境变量、Git 状态、依赖、任务信息、日志、TensorBoard 和资源记录。正式训练要求工作区没有未提交修改。`RELAX_SKIP_CLUSTER_CLEANUP=1` 使下一次提交不执行集群范围的进程清理；资源释放检查在下一项实验之前完成。

[run_a100_campaign.sh](../scripts/run_a100_campaign.sh) 顺序执行 D、E、B/C、F、G，并在结束时审查全部实验和生成资源曲线。F 组的释放时间从取消请求或故障事件开始，结束于全部资源查询及最终任务状态查询完成的时刻；该数值包含采样等待，作为释放时间上限。actor 检查同时覆盖训练 driver 的 job_id 和本次 JobSupervisor 的名称。

接续入口支持在日期与预算之后传入阶段列表，本次命令为 `bash run_a100_campaign.sh 20261114 12 C2 F G`。两个 Shell 脚本的 `bash -n`、Python 入口与选择检查工具的 `py_compile` 均通过；参数选择另由实际镜像中的两个 Gloo rank 验证。

[audit_a100.py](../scripts/audit_a100.py) 独立核对全部实验的状态、代码版本、步骤、TensorBoard 与资源证据；[audit_parameters.py](../scripts/audit_parameters.py) 检查 D1 的逐参数记录覆盖和指定步骤完整张量的摘要。

[summarize_release.py](../scripts/summarize_release.py) 按实际任务 ID、JobSupervisor 名称及诊断事件中的 PID 提取相关 actor，重新计算从触发事件到最终查询完成的时间，保存 GPU、共享内存、实验目录和 actor 退出原因，并分别记录训练资源与完整任务的释放上界。该工具已读取 E1-off 的真实资源记录，全部 8 项检查通过，释放时间上界为 16.1645 秒，证据为 [release-summary-validation.json](../raw/task17-a100-experiments-20261114/preparation/release-summary-validation.json)。F 组九次实际运行的本地验证同样通过；命令为 `python Task17/bounded_condition_cache/real_run/summarize_release.py Relax/output/task17-a100-experiments-20261114/a100-F*-20261114 --output Relax/output/task17-a100-experiments-20261114/preparation/F-release-summary.json`。

[compare_scope.py](../scripts/compare_scope.py) 按指定步骤范围计算 G 组吞吐、复用率和条件分支占比。G1、G3、G4 使用最后一个 epoch 的逐步记录；G2 使用第 2–3 个 epoch 及 B1、B2 参考。工具已读取已有 4090 的追踪器开启与关闭产物完成验证，输出保存于本地 `preparation/scope-tool-validation.json` 和 `preparation/scope-plain-validation.json`；这些数据仅验证统计工具，不作为 A100 结果。

[compare_repeated_perf.py](../scripts/compare_repeated_perf.py) 保存三轮配对收益、构建成本、回本估算及全部跨运行吞吐范围；范围为实际观测的描述统计。该工具已读取 4090 的三轮真实结果完成验证，产物为 [repeated-perf-validation.json](../raw/task17-a100-experiments-20261114/preparation/repeated-perf-validation.json)。A100 九次运行的正式结果保存为 [main-performance-repeats.json](../raw/task17-a100-experiments-20261114/preparation/main-performance-repeats.json)。

[summarize_stages.py](../scripts/summarize_stages.py) 按相同窗口汇总两个 rank 的阶段计时，并分别统计在线编码与复用 micro-batch 的目标图预处理主机时间，通过 `--exclude` 指定排除步骤，同时保留全程编码次数、搬运字节和被排除步骤的 micro-batch 记录。本次 B1 与 C2 阶段比较使用 `--exclude 1`。工具使用 A3 真实产物执行通过，输出为 [stage-summary-validation.json](../raw/task17-a100-experiments-20261114/preparation/stage-summary-validation.json)；该验证产物排除第 1、63、64 步，每个 rank 选择 61 步，共 204 次在线编码与 40 次复用 micro-batch。

统计工具于 2026-11-14 在本地 `/usr/bin/python`（Python 3.14.10）执行通过，Relax HEAD 保持 `9783ebe924323c9742fbd735fb850f93a0a157f0`。实际命令如下，执行目录为 `/home/xbei/Relax`；三个工具的 `py_compile` 检查亦通过。

```bash
python Task17/bounded_condition_cache/real_run/summarize_release.py \
  Relax/output/task17-a100-experiments-20261114/a100-E1-off-20261114 \
  --output Relax/output/task17-a100-experiments-20261114/preparation/release-summary-validation.json
python Task17/bounded_condition_cache/real_run/compare_repeated_perf.py \
  Relax/output/task17-docker-experiments-20261112/perf-summary-20261112.json \
  --groups Relax/output/task17-a100-experiments-20261114/preparation/repeated-perf-validation-groups.json \
  --output Relax/output/task17-a100-experiments-20261114/preparation/repeated-perf-validation.json
python Task17/bounded_condition_cache/real_run/summarize_stages.py \
  Relax/output/task17-a100-experiments-20261114/a100-A3-b12-20261114 \
  --output Relax/output/task17-a100-experiments-20261114/preparation/stage-summary-validation.json
```

[plot_resources.py](../scripts/plot_resources.py) 生成 GPU、主机、共享内存、磁盘和缓存容量曲线，保存 SVG、PNG 与采样数量清单。该工具已使用已有 4090 的两组 96 步记录完成数据与生成检查，产物位于本地 `preparation/resource-curve-validation/`。主机 RSS 曲线为 Ray worker RSS 之和，共享页可能重复计量；磁盘曲线为包含实验目录的文件系统已用空间变化。

逐步记录中的 `cpu_rss_bytes` 表示采样时刻该训练进程的 RSS；`cpu_peak_rss_bytes` 表示进程运行以来的历史峰值，包含模型加载阶段。逐 epoch 资源分析采用前者。`step_peak_allocated_bytes` 记录当前训练步骤中 Tensor 已分配显存的峰值，`step_peak_reserved_bytes` 还包含 PyTorch allocator 保留的显存；二者与 `nvidia-smi` 统计的 GPU 整体占用分别报告。

[summarize_epoch_resources.py](../scripts/summarize_epoch_resources.py)核查每个 rank 的连续步骤及 epoch 编号，保存各 epoch 的缓存计费、条目数量、复用率、淘汰次数、单步已分配显存峰值、reserved 显存和主机 RSS 范围。每条缓存记录均核对其计费峰值不超过指定预算。该工具已在本地 Python 3.14.10 使用 B1-r1 和 C2 的真实 96 步产物执行通过，每个 rank 各三个 epoch；证据为 [epoch-resource-validation.json](../raw/task17-a100-experiments-20261114/preparation/epoch-resource-validation.json)。实际命令为 `python Task17/bounded_condition_cache/real_run/summarize_epoch_resources.py Relax/output/task17-a100-experiments-20261114/a100-B1-r1-20261114 Relax/output/task17-a100-experiments-20261114/a100-C2-20261114 --epochs 3 --budget-gib 12 --output Relax/output/task17-a100-experiments-20261114/preparation/epoch-resource-validation.json`；`py_compile` 同样通过。

[summarize_dimensions.py](../scripts/summarize_dimensions.py)逐条读取两个 rank 的 micro-batch，统计实际目标图尺寸和目标图搬运字节分布。2026-11-14 在本地 Python 3.14.10 使用 B1-r1 与 G4-off 的真实产物执行通过，代码版本为 `9783ebe924323c9742fbd735fb850f93a0a157f0`。实际命令为 `python Task17/bounded_condition_cache/real_run/summarize_dimensions.py Relax/output/task17-a100-experiments-20261114/a100-B1-r1-20261114 Relax/output/task17-a100-experiments-20261114/a100-G4-off-20261114 --output Relax/output/task17-a100-experiments-20261114/preparation/G4-dimensions-baseline.json`。

[select_profile_window.py](../scripts/select_profile_window.py) 从 A3 第二个 epoch 选择同时包含在线编码与复用的窗口，通过 `--window-steps` 指定一个或两个步骤，保存样本编号与复用状态供 C2 核对。A100 使用单步骤窗口，在 15 个符合条件的步骤中选择第 64 步，micro-batch 编码记录与两个 rank 的共同复用计数一致；证据为本地 `preparation/profile-window-single-step.json`。两个实际采集运行的样本检查及 trace 分析均通过。

[inspect_trace.py](../scripts/inspect_trace.py) 使用 `ijson` 流式读取 profiler JSON，按实际 Python 函数区间提取 CPU 算子、CUDA API 及邻近 GPU 活动，并保存函数开始前 500 ms 的主机事件与覆盖该区间的父调用，包含 FSDP 的 `user_annotation`。主机事件的汇总时间包含子调用，不能直接相加解释为函数总时间。工具已在本地 Python 3.13.15、ijson 3.6.0 下读取真实 CPU trace，定位一个函数调用及其 8 个主机事件、132 个前序或父调用事件，证据为 [trace-prelude-validation.json](../raw/task17-a100-experiments-20261114/preparation/trace-prelude-validation.json)。

窗口选择与 trace 提取工具的实际验证命令如下，执行目录为 `/home/xbei/Relax`，代码保持 `9783ebe924323c9742fbd735fb850f93a0a157f0`：

```bash
python Task17/bounded_condition_cache/real_run/select_profile_window.py \
  Relax/output/task17-a100-experiments-20261114/a100-A3-b12-20261114 \
  --budget-gib 12 --window-steps 1 \
  --output Relax/output/task17-a100-experiments-20261114/preparation/profile-window-single-step.json
env TMPDIR=/home/xbei/Relax/Task17/.work/tmp \
  Relax/output/task17-a100-experiments-20261114/preparation/venv/bin/python \
  Task17/bounded_condition_cache/real_run/inspect_trace.py \
  Relax/output/task17-a100-experiments-20261114/cpu-validation-profiler/pytest-regression/test_train_profiler_exports_py1/train_overall_rank0_dp0_tp0_pp0.1794608086854489010.pt.trace.json.gz \
  --function 'test_profile_utils.py(50): _python_stack_probe_leaf' --expected-count 1 \
  --output Relax/output/task17-a100-experiments-20261114/preparation/trace-prelude-validation.json
```

C2 的分析依据包含已完整阅读的 [PyTorch 复制教程](https://docs.pytorch.org/tutorials/intermediate/pinmem_nonblock.html)，以及从实验镜像读取的 [PyTorch 2.11 CUDAFunctions.h](../raw/task17-a100-experiments-20261114/preparation/torch-2.11-CUDAFunctions.h)。该版本的 `memcpy_and_sync` 依次调用 `cudaMemcpyAsync` 与 `cudaStreamSynchronize`；trace 将分别记录复制、同步和同期 GPU 活动的时间。头文件 SHA256 为 `b73b5ad35f5d077dd3643d4719d474bb8960a7f1b8d7a3af087dd4eef27435e9`。2026-11-14 的 Google 页面访问失败，官方教程通过 Parallel 读取。

[analyze_preprocess_trace.py](../scripts/analyze_preprocess_trace.py)通过 correlation ID 与 External id 核对 CPU、GPU 复制事件，并检查函数时间与实际追踪记录一致，保存 NumPy 图像转换、AllGather、参数复制及对应主机算子的时间。[audit_profile_captures.py](../scripts/audit_profile_captures.py)核查 96 步运行、两个 rank 的样本、trace 数量、采集 rank、文件大小与 SHA256。两项工具已使用两个 rank 的实际产物在本地执行通过，Relax 代码版本为 `9783ebe924323c9742fbd735fb850f93a0a157f0`，命令如下：

```bash
python Task17/bounded_condition_cache/real_run/analyze_preprocess_trace.py \
  Relax/output/task17-a100-experiments-20261114/preparation/profile-single-rank0-events.json \
  --samples Relax/output/task17-a100-experiments-20261114/a100-orchestration-20261114/profile-samples.json \
  --rank 0 --output Relax/output/task17-a100-experiments-20261114/preparation/profile-single-rank0-analysis.json
python Task17/bounded_condition_cache/real_run/analyze_preprocess_trace.py \
  Relax/output/task17-a100-experiments-20261114/preparation/profile-single-rank1-events.json \
  --samples Relax/output/task17-a100-experiments-20261114/a100-orchestration-20261114/profile-samples.json \
  --rank 1 --output Relax/output/task17-a100-experiments-20261114/preparation/profile-single-rank1-analysis.json
env TMPDIR=/home/xbei/Relax/Task17/.work/tmp \
  Relax/output/task17-a100-experiments-20261114/preparation/venv/bin/python \
  Task17/bounded_condition_cache/real_run/audit_profile_captures.py \
  --root Relax/output/task17-a100-experiments-20261114 --date 20261114 \
  --commit 9783ebe924323c9742fbd735fb850f93a0a157f0 \
  --output Relax/output/task17-a100-experiments-20261114/preparation/profile-capture-audit-local.json
```
