# Qwen-Image-Edit SFT 完整规模训练与对照报告

[材料说明](../README.md) · [4090 训练报告](real_training_report.md) · [A100 实验报告](a100_training_report.md) · [PR 正文](https://github.com/redai-studio/Relax/pull/571)

本报告记录 2026-11-20（Asia/Shanghai）在 `2 × NVIDIA A100-SXM4-80GB` 环境中基于完整规模 `Qwen/Qwen-Image-Edit-2509` 完成的 GPU 单元测试、DiffSynth 独立参考对照与 13 次完整规模 SFT 实验队列。实验覆盖 R1 在线基线、R2 第二种模型共用公共缓存、R3–R5 训练数值一致、R6–R7 端到端性能与回本条件、R8 容量与资源以及 R9 独立复跑材料。

## 1. 环境与版本

| 项目 | 内容 |
| --- | --- |
| 机器 | 租用的单台服务器，Ubuntu 22.04.3 LTS，`2 × NVIDIA A100-SXM4-80GB`（每张 `81920 MiB`，拓扑 `PHB`），驱动 `550.127.05`，主机物理内存 `67382091776` bytes（`62 GiB`） |
| 基础镜像与容器 | 基于 `ghcr.io/redai-studio/relaxrl@sha256:cd431e1094646c347aa97bc9cb8c5ac8315d4e72b0b6321f1cafd1ae12e680c6`，在 ABI 约束下升级 `transformers==5.16.0` 与 `tokenizers==0.23.2`，对 `/sgl-workspace/sglang` 应用仓库补丁 `docker/patch/sglang/v0.5.17-transformers5-compat.patch`，提交为固定镜像 `relaxrl:task17-qwen`（ID `728f53049933`，`sha256:728f530499333398c2cbfddfb5064769da69252e2e874b49d2827ab196a71a7b`）；由 [start_qwen_container.sh](../scripts/qwen/start_qwen_container.sh) 启动容器 `relax-task17-qwen` |
| 软件环境 | 系统 Python `3.12.3`、`torch 2.11.0+cu129`、`torchvision 0.26.0+cu129`、`transformers 5.16.0`、`tokenizers 0.23.2`、`diffusers 0.37.0`、`peft 0.20.0`、`flash-attn 2.7.4.post1`、`ray 2.58.0`、`Pillow 12.3.0`；完整 `pip freeze` 保存在各实验目录的 `environment.txt` |
| 独立参考环境 | `/opt/diffsynth-venv/bin/python`（基于 `--system-site-packages` 覆盖安装 `transformers==4.57.1`、`tokenizers==0.22.2`、`huggingface-hub==0.36.2`、`kernels==0.11.7`，加载 `DiffSynth-Studio` 提交 `f7b7db96a2572e69693d85e78a1d29d9e8489ef3`） |
| 实验代码 | 分支 [`task17/qwen-experiments`](https://github.com/xBei0223/Relax/tree/task17/qwen-experiments)，提交 [`44a54ab67c5332fcbb32d83703849a2e31fe5766`](https://github.com/xBei0223/Relax/commit/44a54ab67c5332fcbb32d83703849a2e31fe5766)（工作区无未提交修改） |
| 模型 | `Qwen/Qwen-Image-Edit-2509`（Hugging Face revision `d3968ef930e841f4c73640fb8afa3b306a78167e`，34 个模型与配置文件，总计 `53.76 GiB`；通过 ModelScope 下载，全部 34 个文件经 LFS SHA256 与 Git blob SHA1 校验通过，0 不匹配） |
| 数据 | `osunlp/MagicBrush` revision `6c254560879df7851233ea0fd2ae2a62764d0ef2`；[prepare_magicbrush.py](../scripts/prepare_magicbrush.py) 按 `train-*` 分片顺序取前 128 行，`train.json` SHA256 为 `3311ec0f682a80447ef1ac0f0f8567062269c944bbe9dcae488fc6b2c882def9` |

双卡通信检查 `torchrun --standalone --nproc_per_node=2 /root/Task17/bounded_condition_cache/real_run/check_gpu.py` 在两个 rank 上均完成 BF16 矩阵乘法、FlashAttention 2 与 NCCL `all_reduce`（归约结果 `3.0`）。

## 2. 双 GPU 单元测试

在容器 `/root/Relax` 目录执行：

```bash
mkdir -p /qwen/tmp/gpu-tests /experiments/qwen-tests
cd /root/Relax
TMPDIR=/qwen/tmp PYTHONPATH=/root/Relax python -m pytest \
    tests/models/test_qwen_image_edit_sft.py \
    tests/gpux4/models/test_qwen_image_edit_sft.py \
    -v -rs --tb=short \
    --basetemp=/qwen/tmp/gpu-tests \
    --junitxml=/experiments/qwen-tests/qwen-gpu.xml
```

全部 5 项测试通过、0 项跳过，耗时 `98.36 s`；原始证据为 [qwen-gpu.xml](../raw/task17-qwen-experiments-20261120/qwen-tests/qwen-gpu.xml) 与 [qwen-gpu.log](../raw/task17-qwen-experiments-20261120/qwen-tests/qwen-gpu.log)（包含 `test_qwen_flash_attention_outputs_gradients_determinism` 的 4 个输出与输入梯度张量误差 `relative_l2=0.0`、`max_absolute_error=0.0`）：

| 测试用例 | 结果 |
| --- | --- |
| `tests/models/test_qwen_image_edit_sft.py::test_qwen_single_gpu_fsdp_cache_and_checkpoint[full]` | PASSED |
| `tests/models/test_qwen_image_edit_sft.py::test_qwen_single_gpu_fsdp_cache_and_checkpoint[lora]` | PASSED |
| `tests/models/test_qwen_image_edit_sft.py::test_qwen_flash_attention_outputs_gradients_determinism` | PASSED |
| `tests/gpux4/models/test_qwen_image_edit_sft.py::test_qwen_two_gpu_fsdp_cache_gradients_rng_and_checkpoint[full]` | PASSED |
| `tests/gpux4/models/test_qwen_image_edit_sft.py::test_qwen_two_gpu_fsdp_cache_gradients_rng_and_checkpoint[lora]` | PASSED |

## 3. 完整规模 DiffSynth 参考对照

在单张 80 GB GPU（`CUDA_VISIBLE_DEVICES=0`）上依次运行 [run_diffsynth_step.py](../scripts/qwen/reference/run_diffsynth_step.py) 与 [compare_diffsynth_step.py](../scripts/qwen/reference/compare_diffsynth_step.py)，加载完整规模 `Qwen/Qwen-Image-Edit-2509` 与 MagicBrush 128 行子集前两条样本（LoRA rank 32）。

1. **同注意力实现对照（`--attention-implementation torch`）**：DiffSynth 参考侧的 `QwenImageTextEncoder` 与 `QwenImageDiT` 均使用 PyTorch SDPA。候选侧使用 `--attention-implementation torch` 时退出码 `0`，全部 8 组用例 `"passed": true`；`position_ids`、`prompt_embeds`、`target_latents`、`edit_latents_*`、`noise`、`timestep`、`index`、`sigma`、`weight`、`after.rng_cpu` 与 `after.rng_cuda` 均逐元素相等（误差 `0.0`）。汇总结果见 [candidate/summary.json](../raw/task17-qwen-experiments-20261120/qwen-reference-diffsynth/candidate/summary.json) 与 [compare_diffsynth_torch.log](../raw/task17-qwen-experiments-20261120/qwen-reference-diffsynth/compare_diffsynth_torch.log)：

| 用例（`torch`） | `passed` | 参考损失 | 损失最大绝对误差 | 损失相对误差 | `prediction` 相对 L2 误差 | 最大梯度相对 L2 误差 | 梯度误差超过 1% 的 Tensor 数量 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `images-1-timestep-1` | `true` | `0.0` | `0.0` | `0.0` | `0.0029922370164212225` | `0.0` | `0` |
| `images-1-timestep-2` | `true` | `0.0` | `0.0` | `0.0` | `0.0029922370164212225` | `0.0` | `0` |
| `images-1-timestep-499` | `true` | `0.025502298027276993` | `0.0001193564385175705` | `0.004680222872068552` | `0.0036417183747326992` | `1.025639000127079` | `1428` |
| `images-1-timestep-random` | `true` | `0.03457467630505562` | `8.966773748397827e-05` | `0.0025934512500661294` | `0.003706411031307153` | `1.8034108476297812` | `1414` |
| `images-2-timestep-1` | `true` | `0.0` | `0.0` | `0.0` | `0.003647988877834467` | `0.0` | `0` |
| `images-2-timestep-2` | `true` | `0.0` | `0.0` | `0.0` | `0.003647988877834467` | `0.0` | `0` |
| `images-2-timestep-499` | `true` | `0.020335260778665543` | `5.369633436203003e-05` | `0.002640553024939065` | `0.0038151747804686187` | `10.571674850111167` | `1428` |
| `images-2-timestep-random` | `true` | `0.03639347106218338` | `1.3560056686401367e-05` | `0.0003725958610332083` | `0.004138280225279025` | `2.187893225340267` | `1432` |

   DiT 前向与反向中的梯度差异来自标准 `diffusers 0.37.0` MLP 的 `tanh` GELU 近似与 DiffSynth MLP 的 `sigmoid(1.702x)` 近似在导数上的固有区别。

2. **跨注意力实现对照（`--attention-implementation flash_attention_2`）**：候选侧在 60 层 BF16 文本编码器与 DiT 中使用 FlashAttention 2，对比参考侧 PyTorch SDPA 时，`prompt_embeds` 相对 L2 为 `0.13227776`（1 图）与 `0.12859482`（2 图），8 组中有 7 组损失相对误差低于 `0.01`，`images-1-timestep-499` 为 `0.01357315`；结果保存在 [candidate-fa2/summary.json](../raw/task17-qwen-experiments-20261120/qwen-reference-diffsynth/candidate-fa2/summary.json) 与 [compare_diffsynth_fa2.log](../raw/task17-qwen-experiments-20261120/qwen-reference-diffsynth/compare_diffsynth_fa2.log)。

## 4. 完整规模训练配置与队列执行

实验配置固定在 [config.json](../scripts/qwen/real_run/config.json)：

| 配置 | 值 |
| --- | --- |
| 训练模式 | LoRA，rank 与 alpha 为 32，学习率 `1e-4`，线性 warmup 200 步，FP32 主权重，BF16 计算，FP32 归约 |
| 并行与批次 | 2 GPU FSDP2 数据并行（`--fsdp-load-wave-size 1`），`GLOBAL_BATCH_SIZE=4`，每个 rank 每步 2 个 micro-batch，每个 epoch 32 步 |
| 图像与优化 | `MAX_PIXELS=1048576`、`CONDITIONING_IMAGES=1`、FlashAttention 2、`TRANSFORMER_GRADIENT_CHECKPOINTING=1`、`IMAGE_PREFETCH_DEPTH=2`、`IMAGE_PREFETCH_WORKERS=2` |
| 缓存容量 | 每个 rank `condition_cache_max_entries=256`、`condition_cache_max_bytes=2147483648`（`2 GiB`） |
| 随机种子 | `42` |

队列由 [run_campaign.py](../scripts/qwen/real_run/run_campaign.py) 顺序执行（2026-11-20 `17:16:57 CST` 至 `23:54:41 CST`），随后由 [summarize.py](../scripts/qwen/real_run/summarize.py) 汇总。全部 13 次实验状态均为 `SUCCEEDED`，原始执行日志见 [run-campaign-20261120.log](../raw/task17-qwen-experiments-20261120/qwen-campaign-20261120/run-campaign-20261120.log)：

| 实验 | 起止时间（CST） | 状态 | `nvidia-smi` 显存峰值（MiB） | 缓存计费峰值（rank 0 / rank 1） |
| --- | --- | --- | --- | --- |
| `qwen-smoke` | `17:16:57`–`17:21:55` | `SUCCEEDED` | `59954` / `59954` | `20624384` / `20724736` bytes（各 8 条目） |
| `qwen-parity-off` | `17:21:57`–`17:48:18` | `SUCCEEDED` | `60050` / `60050` | — |
| `qwen-parity-on` | `17:48:24`–`18:14:11` | `SUCCEEDED` | `60142` / `60142` | `238144512` / `238316544` bytes（各 92 条目） |
| `qwen-baseline-plain-r1` | `18:14:16`–`18:49:10` | `SUCCEEDED` | `59946` / `59946` | — |
| `qwen-cache-plain-r1` | `18:49:16`–`19:21:23` | `SUCCEEDED` | `60054` / `60056` | `279579648` / `284837888` bytes（108 / 110 条目） |
| `qwen-baseline-tracker-r1` | `19:21:28`–`19:56:05` | `SUCCEEDED` | `59946` / `59946` | — |
| `qwen-baseline-plain-r2` | `19:56:09`–`20:30:42` | `SUCCEEDED` | `59946` / `59946` | — |
| `qwen-cache-plain-r2` | `20:30:48`–`21:03:51` | `SUCCEEDED` | `60054` / `60056` | `279579648` / `284837888` bytes（108 / 110 条目） |
| `qwen-baseline-tracker-r2` | `21:03:56`–`21:38:37` | `SUCCEEDED` | `59946` / `59946` | — |
| `qwen-baseline-plain-r3` | `21:38:42`–`22:13:05` | `SUCCEEDED` | `59946` / `59946` | — |
| `qwen-cache-plain-r3` | `22:13:10`–`22:46:23` | `SUCCEEDED` | `60054` / `60056` | `279579648` / `284837888` bytes（108 / 110 条目） |
| `qwen-baseline-tracker-r3` | `22:46:29`–`23:21:47` | `SUCCEEDED` | `59946` / `59946` | — |
| `qwen-cache-tracker-r1` | `23:21:52`–`23:54:41` | `SUCCEEDED` | `60054` / `60056` | `279579648` / `284837888` bytes（108 / 110 条目） |

## 5. 训练数值对照（R3–R5）

`qwen-parity-off` 与 `qwen-parity-on` 使用确定性 attention、LoRA rank 与 alpha 32、相同种子各训练 64 步（2 个 epoch）并保存最终 checkpoint。[parity.json](../raw/task17-qwen-experiments-20261120/qwen-campaign-20261120/summary/parity.json) 显示两侧全部指标、checkpoint 张量与 RNG 状态逐元素相同：`qwen-parity-on` 每个 rank `128` 次请求，本地命中 `36` 次，组内共同复用 `23` 次（第 2 个 epoch 的 `35.94%`），`13` 次因另一 rank 未命中转为在线计算，无淘汰。

| 比较项 | 结果 |
| --- | --- |
| 64 步 `train/loss` | `64/64` 步一致，`max_abs=0.0`，`max_relative=0.0` |
| 64 步 `train/grad_norm` | `64/64` 步一致，`max_abs=0.0`，`max_relative=0.0` |
| 64 步 `train/lr` | `64/64` 步一致，`max_abs=0.0`，`max_relative=0.0` |
| 最终 checkpoint（LoRA 参数与 FP32 optimizer 状态） | `5748` 个张量，`mismatched=0`，`max_abs=0.0`，`worst_relative_l2=0.0` |
| 各 rank RNG 状态 | `6` 个张量，`mismatched=0`，`max_abs=0.0`，`worst_relative_l2=0.0` |

## 6. 在线基线与追踪器（R1）

`qwen-smoke`（4 步、预热 1 步）记录单步阶段均值 `18740.31 ms`、加载期间历史显存峰值 `55.08 GiB` allocated / `57.90 GiB` reserved、单 rank CPU RSS 峰值 `17.05 GiB`。96 步（`384` 个消费位置、唯一数据行全局 `128`、本地之和 `218`、全局重复 `256` 即 `66.67%`、本地重复 `166` 即 `43.23%`）对比见下表，原始汇总位于 [qwen-baseline-tracker-r1/baseline_stats/baseline_summary.json](../raw/task17-qwen-experiments-20261120/qwen-campaign-20261120/qwen-baseline-tracker-r1/baseline_stats/baseline_summary.json) 与 [qwen-cache-tracker-r1/baseline_stats/baseline_summary.json](../raw/task17-qwen-experiments-20261120/qwen-campaign-20261120/qwen-cache-tracker-r1/baseline_stats/baseline_summary.json)：

| 指标 | `qwen-baseline-tracker-r1` | `qwen-cache-tracker-r1` | 变化 |
| --- | --- | --- | --- |
| `prompt_encode` / `ref_vae_encode` / `target_vae_encode` 调用次数 | 各 `384` 次（全局重复 `256` 次） | 各 `266` 次（全局重复 `138` 次） | 各减少 `118` 次（`2 × 59`） |
| 条件 H2D 搬运总量 | `5961.02 MiB` | `4129.25 MiB` | 减少 `1831.77 MiB`（`30.73%`） |
| `prompt_encode_ms` 单步均值 / 占比 | `2464.81 ms`（`13.18%`） | `1687.86 ms`（`9.61%`） | 减少 `776.96 ms` |
| `ref_vae_encode_ms` 单步均值 / 占比 | `186.31 ms`（`1.00%`） | `128.21 ms`（`0.73%`） | 减少 `58.09 ms` |
| `target_vae_encode_ms` 单步均值 / 占比 | `185.74 ms`（`0.99%`） | `127.77 ms`（`0.73%`） | 减少 `57.97 ms` |
| `dit_forward_ms` / `backward_ms` 单步均值 | `8065.57 ms` / `7690.42 ms` | `7985.52 ms` / `7504.12 ms` | — |
| 追踪器覆盖 `train_ms` 单步均值 | `18696.85 ms` | `17555.44 ms` | 减少 `1141.41 ms` |

`baseline-tracker` 三轮的 `prompt_embeds` 均值为 `1.47 MiB`（最大 `1.52 MiB`），`edit_latents` 与 `target_latents` 均为 `0.50 MiB`。三组 `baseline-tracker` 与 `baseline-plain` 比较中，追踪器总训练时间差异均值为 `−1.23%`（范围 `−2.45%` 至 `+0.36%`），第 3 个 epoch 单步差异均值为 `−1.27%`（范围 `−2.44%` 至 `+0.24%`）。

## 7. 端到端性能与资源（R6、R7、R8）

三轮重复对比来自 [performance.json](../raw/task17-qwen-experiments-20261120/qwen-campaign-20261120/summary/performance.json)、[walltime.json](../raw/task17-qwen-experiments-20261120/qwen-campaign-20261120/summary/walltime.json) 与 [repetitions.json](../raw/task17-qwen-experiments-20261120/qwen-campaign-20261120/summary/repetitions.json)：

| 指标 | 缓存关闭均值及范围（`baseline-plain` 3 次） | 缓存开启均值及范围（`cache-plain` 3 次） | 配对比较均值及范围 |
| --- | --- | --- | --- |
| 提交至首步完成 | `208.93 s`（`206.42–213.00`） | `212.40 s`（`195.67–223.33`） | — |
| 首步 | `25.39 s`（`25.30–25.47`） | `25.11 s`（`24.50–25.44`） | — |
| 第 1 个 epoch 单步（无复用） | `19.029 s`（`18.928–19.164`） | `18.750 s`（`18.496–18.945`） | 吞吐 `+1.50%`（`−0.09%` 至 `+3.61%`） |
| 第 2 个 epoch 单步 | `19.027 s`（`18.919–19.180`） | `17.704 s`（`17.463–17.884`） | 吞吐 `+7.49%`（`+5.79%` 至 `+9.83%`） |
| 第 3 个 epoch 单步（稳态） | `19.028 s`（`18.930–19.164`） | `17.139 s`（`16.918–17.323`） | 吞吐 **+11.04%**（`+9.28%` 至 `+13.28%`） |
| 96 步总训练时间 | `1833.08 s`（`1823.24–1846.49`） | `1721.35 s`（`1698.07–1739.36`） | 减少 `6.09%`（`4.60%` 至 `8.04%`） |
| 提交至作业终止 | `2074.21 s`（`2061.74–2092.81`） | `1965.34 s`（`1923.92–1990.04`） | — |

- 配对与复用统计：三轮第 3 个 epoch 吞吐提升分别为 `+13.28%`、`+10.55%`、`+9.28%`；第 1 个 epoch 缓存构建成本差值均值为 `−8.94 s`（三对为 `−21.61 s`、`−5.86 s`、`+0.66 s`），回本样本估算为 `0`、`0`、`2`。四次 96 步缓存运行的计数一致：每个 rank `192` 次请求，rank 0 本地命中 `84` 次（第 2–3 个 epoch 的 `65.63%`）、写入 `108` 条目（`266.63 MiB`），rank 1 本地命中 `82` 次（第 2–3 个 epoch 的 `64.06%`）、写入 `110` 条目（`271.64 MiB`），组内共同复用 `59` 次（第 2–3 个 epoch 的 `46.09%`），`evictions=0`，`oversized_skips=0`；单次请求均耗 `lookup_ms` `0.24–0.26 ms`、`control_ms` `0.40–0.51 ms`、`version_check_ms` `5.05–5.15 ms`、`restore_ms` 低于 `0.001 ms`。
- 适用条件分析：Qwen-Image-Edit SFT 的 DiT 前向（`43.1%`）与反向（`41.1%`）占单步约 `84.3%`，三种条件编码合计占单步约 `15.2%`（`2836.86 ms / 18696.85 ms`）。在 `46.09%` 组内共同复用率下，四个缓存运行自身从第 1 个 epoch 到第 3 个 epoch 的单步耗时减少稳定在 `1.58–1.63 s`（内部吞吐提升 `9.33%`–`9.52%`），配对测量稳态吞吐提升均值为 `+11.04%`。
- 资源与归档（[resources.json](../raw/task17-qwen-experiments-20261120/qwen-campaign-20261120/summary/resources.json)）：`nvidia-smi` 显存峰值从 `59946 MiB` 变为 `60054 / 60056 MiB`（增加 `108–110 MiB`）；宿主机已用内存峰值分别为 `11.94–13.62 GiB`（关闭）与 `11.03–14.11 GiB`（开启）；`/dev/shm` 峰值为 `103.22 MiB`（`parity` 为 `140.03 MiB`），结束时为 `1.14 MiB`（`qwen-baseline-plain-r1` 末次采样为 `103.22 MiB`）；96 步目录磁盘增量为 `11.55–13.90 MiB`。

## 8. 复跑方法

1. 准备 `2 × NVIDIA A100-SXM4-80GB`（主机内存不少于 `64 GB`），在官方基础镜像内按 `docker/Dockerfile` ABI 约束升级 `transformers==5.16.0` 与 `tokenizers==0.23.2`，对 `/sgl-workspace/sglang` 应用 `docker/patch/sglang/v0.5.17-transformers5-compat.patch`，并可选创建 `/opt/diffsynth-venv`（`transformers==4.57.1`、`tokenizers==0.22.2`、`huggingface-hub==0.36.2`、`kernels==0.11.7` 与 `DiffSynth-Studio` `f7b7db96a2572e69693d85e78a1d29d9e8489ef3`），提交镜像为 `relaxrl:task17-qwen`。
2. 检出分支 `task17/qwen-experiments`（`44a54ab`）到 `/root/task17/Relax`；将本分支 `scripts/` 置于 `/root/task17/Task17/bounded_condition_cache/real_run`，将 `scripts/qwen/` 置于 `/root/task17/Task17/qwen_image_edit_sft/`；下载并校验 `Qwen/Qwen-Image-Edit-2509`（revision `d3968ef930e841f4c73640fb8afa3b306a78167e`）到 `/root/task17/qwen/models/Qwen-Image-Edit-2509`，准备 128 行 MagicBrush 子集到 `/root/task17/data/magicbrush-128`。
3. 运行 [start_qwen_container.sh](../scripts/qwen/start_qwen_container.sh) 启动容器 `relax-task17-qwen`，在容器内启动 Ray head 后依次运行测试、参考对照、完整队列与汇总脚本：

```bash
mkdir -p /qwen/ray-tmp
RAY_JOB_STOP_WAIT_TIME_S=60 ray start --head \
    --port=6379 --dashboard-host=127.0.0.1 --dashboard-port=8265 \
    --temp-dir=/qwen/ray-tmp

TMPDIR=/qwen/tmp TZ=Asia/Shanghai PYTHONPATH=/root/Relax:/root/Megatron-LM \
    python /root/Task17/qwen_image_edit_sft/real_run/run_campaign.py \
    --repo /root/Relax --model /models/Qwen-Image-Edit-2509 \
    --data /data/magicbrush-128 --output /experiments/qwen-campaign-20261120

TMPDIR=/qwen/tmp PYTHONPATH=/root/Relax:/root/Megatron-LM \
    python /root/Task17/qwen_image_edit_sft/real_run/summarize.py \
    /experiments/qwen-campaign-20261120 \
    --output /experiments/qwen-campaign-20261120/summary
```

队列原始产物校验清单见 [remote-sha256.txt](../raw/task17-qwen-experiments-20261120/qwen-campaign-20261120/remote-sha256.txt)（列出 208 份已归档文件及 4 个未打入压缩包的 `*.distcp` checkpoint 分片摘要）。
