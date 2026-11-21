# Task17 实验材料

本分支保存 [redai-studio/Relax#571](https://github.com/redai-studio/Relax/pull/571)（冻结条件特征复用与有界缓存）的真实 GPU 实验材料：实验脚本、配置、汇总与审查工具，以及实验报告。原始日志与汇总数据作为 Release [`task17-experiments-20261116`](https://github.com/xBei0223/Relax/releases/tag/task17-experiments-20261116) 的附件发布。

## 内容

| 目录 | 内容 |
| --- | --- |
| [`reports/`](reports) | [4090 训练报告](reports/real_training_report.md)、[A100 实验计划](reports/a100_experiment_plan.md)、[A100 实验报告](reports/a100_training_report.md)、[Qwen A100 实验报告](reports/qwen_training_report.md) |
| [`scripts/`](scripts) | SenseNova 的容器启动、Ray 启动、实验执行与资源采样脚本，[`configs/`](scripts/configs) 中的实验配置，MagicBrush 子集生成脚本，性能、数值、资源与释放的汇总及审查脚本；[`qwen/`](scripts/qwen) 下存放 Qwen-Image-Edit 的容器启动脚本、完整规模实验队列与汇总脚本（[`qwen/real_run/`](scripts/qwen/real_run)）及 DiffSynth 参考对照脚本（[`qwen/reference/`](scripts/qwen/reference)） |
| `raw/` | 不在分支中；从 Release 下载原始日志压缩包并解压到此目录后，报告中的日志链接即可打开 |

## 版本

| 项目 | 版本 |
| --- | --- |
| 4090 实验代码（SenseNova） | PR 提交 [`60706131f2b01db708dbc3f57c5c9e9f8cd8ffe5`](https://github.com/xBei0223/Relax/commit/60706131f2b01db708dbc3f57c5c9e9f8cd8ffe5) |
| A100 实验代码（SenseNova） | 分支 [`task17/a100-experiments`](https://github.com/xBei0223/Relax/tree/task17/a100-experiments)，提交 [`9783ebe924323c9742fbd735fb850f93a0a157f0`](https://github.com/xBei0223/Relax/commit/9783ebe924323c9742fbd735fb850f93a0a157f0)：PR 的缓存实现，加默认关闭的实验诊断与 FSDP profiler rank 名称修复 |
| A100 实验代码（Qwen-Image-Edit） | 分支 [`task17/qwen-experiments`](https://github.com/xBei0223/Relax/tree/task17/qwen-experiments)，提交 [`44a54ab67c5332fcbb32d83703849a2e31fe5766`](https://github.com/xBei0223/Relax/commit/44a54ab67c5332fcbb32d83703849a2e31fe5766)：Qwen-Image-Edit SFT 适配、公共缓存接入、DCP 导出、启动脚本与离线测试夹具 |
| 基础镜像 | `ghcr.io/redai-studio/relaxrl@sha256:cd431e1094646c347aa97bc9cb8c5ac8315d4e72b0b6321f1cafd1ae12e680c6`；Qwen 实验在其基础上按 ABI 约束升级 `transformers==5.16.0` 与 `tokenizers==0.23.2`、应用 `docker/patch/sglang/v0.5.17-transformers5-compat.patch`，提交为 `relaxrl:task17-qwen`（ID `728f53049933`） |
| 模型 | `sensenova/SenseNova-U1.5-8B-MoT` revision `a2462cc1b549bace2c0896143bbbcbb3fa217c76`；`Qwen/Qwen-Image-Edit-2509` revision `d3968ef930e841f4c73640fb8afa3b306a78167e`（DiffSynth 参考仓库提交 `f7b7db96a2572e69693d85e78a1d29d9e8489ef3`） |
| 数据 | `osunlp/MagicBrush` revision `6c254560879df7851233ea0fd2ae2a62764d0ef2`；[`prepare_magicbrush.py`](scripts/prepare_magicbrush.py) 按 `train-*` 分片文件名顺序取前 128、256、512 行，128 行子集 metadata 的 SHA256 为 `3311ec0f682a80447ef1ac0f0f8567062269c944bbe9dcae488fc6b2c882def9` |

## 原始日志

| Release 附件 | SHA256 | 内容 |
| --- | --- | --- |
| `task17-4090-raw-20261112.tar.zst` | `569a9c6296a82397b49d7ef311aefc03ceb86617bfbd3080ab426a3d836e9bff` | 4090 的 12 次训练、冒烟运行与官方镜像测试，解压后为 `task17-docker-experiments-20261112/`，共 277 项、8.63 MiB |
| `task17-a100-raw-20261114.tar.zst` | `7110fad6cc0e2df5a9e95121d1fd7eb7bcb0fe172bb8b8d60b181412d8787a8b` | A100 SenseNova 的全部实验、准备记录、CPU 验证与统一审查，解压后为 `task17-a100-experiments-20261114/`，共 2371 项、2.25 GiB |
| `task17-qwen-a100-raw-20261120.tar.zst` | `20cdec91bda72003d4977ec671137fa9d466a9e97a01c10cb4f356c61c6af4ab` | A100 Qwen-Image-Edit 的 5 项 GPU 单元测试、8 组完整规模 DiffSynth 参考对照与 13 次完整规模实验队列（含汇总 JSON 与 `remote-sha256.txt`），解压后为 `task17-qwen-experiments-20261120/`，共 305 项、19.55 MiB |

每个实验目录包含生效配置、Git 状态、依赖列表、`nvidia-smi`、Ray 任务日志、TensorBoard 事件、5 秒资源采样与追踪器统计；比较与审查结果为同目录、`a100-orchestration-20261114/` 或 `qwen-campaign-20261120/summary/` 中的 JSON。A100 产物中的 `preparation/full-server-sha256-20261115.txt` 与 Qwen 产物中的 `qwen-campaign-20261120/remote-sha256.txt` 为服务器端文件的 SHA256 清单。

压缩包不包含：训练 checkpoint 与恢复实验 checkpoint 副本中的模型与 optimizer 分片（`*.distcp`，SenseNova 约 14.7 GB、Qwen `parity` 两组各约 2.7 GB；同目录的 RNG 状态、训练状态与 TensorBoard 事件保留）、D1 的逐参数完整张量（`diagnostics/*.pt`）、DiffSynth 对照大体积 `*.safetensors`、profiler trace 原始文件、本地 CPU 验证使用的虚拟环境与 pytest 临时目录。逐参数记录摘要（`diagnostics/*.jsonl`）、trace 提取结果与各项比较结果包含在压缩包中。

解压方法：

```bash
mkdir -p raw
zstd -dc task17-4090-raw-20261112.tar.zst | tar -xf - -C raw
zstd -dc task17-a100-raw-20261114.tar.zst | tar -xf - -C raw
zstd -dc task17-qwen-a100-raw-20261120.tar.zst | tar -xf - -C raw
```

## 复跑

脚本按以下服务器目录结构编写，容器内分别挂载为 `/root/Relax`、`/root/Task17`、`/models`、`/data`、`/experiments`、`/work`（Qwen 容器额外挂载 `/qwen`）：

| 服务器路径 | 内容 |
| --- | --- |
| `/root/task17/Relax` | 对应实验代码版本的 Relax 仓库，工作区没有未提交修改 |
| `/root/task17/Task17/bounded_condition_cache/real_run` | 本分支 `scripts/` 目录的内容（不含 `qwen/`） |
| `/root/task17/Task17/qwen_image_edit_sft` | 本分支 `scripts/qwen/` 目录的内容（`real_run/`、`reference/` 与 `start_qwen_container.sh`） |
| `/root/task17/models/SenseNova-U1.5-8B-MoT` | 上表 revision 的 SenseNova 模型文件 |
| `/root/task17/qwen/models/Qwen-Image-Edit-2509` | 上表 revision 的 Qwen-Image-Edit 模型文件 |
| `/root/task17/data` | `prepare_magicbrush.py` 生成的子集 |
| `/root/task17/experiments`、`/root/task17/qwen/experiments`、`/root/task17/work` | 实验产物与临时文件 |

4090 实验的执行步骤见 [4090 训练报告第 7 节](reports/real_training_report.md#7-复跑方法)。SenseNova A100 实验的环境准备、各组配置与执行入口见 [A100 实验计划](reports/a100_experiment_plan.md)和 [A100 实验报告](reports/a100_training_report.md)：[`run_a100.py`](scripts/run_a100.py) 按 `--stage` 执行一组实验，[`run_a100_campaign.sh`](scripts/run_a100_campaign.sh) 按日期、每个 rank 的缓存预算（GiB）和阶段列表顺序执行并在结束时审查全部实验。Qwen-Image-Edit A100 实验的镜像构建、GPU 单元测试、DiffSynth 参考对照及 13 次完整规模队列复跑步骤见 [Qwen A100 实验报告第 8 节](reports/qwen_training_report.md#8-复跑方法)：[`run_campaign.py`](scripts/qwen/real_run/run_campaign.py) 顺序执行全部 13 次运行，[`summarize.py`](scripts/qwen/real_run/summarize.py) 生成汇总 JSON。

报告中记录的服务器地址与本地同步命令中的连接参数已替换为占位文字；本地命令中的 `/home/xbei/Relax` 为实验时的本地目录，对应本分支的上一级目录结构 `Task17/bounded_condition_cache/real_run`（即 `scripts/`）与 `Relax/output/`（即 `raw/`）。
