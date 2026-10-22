# TINA 实验文件索引

迁移与交接入口：[HANDOFF.md](HANDOFF.md)、[MIGRATION.md](MIGRATION.md)。统一成绩表：`migration/final_scores.csv`；54项评测原始文件映射：`migration/evaluation_registry.json`；完整文件清单和校验和：`migration/file_manifest.json`、`migration/SHA256SUMS`。项目根目录的`TINA_evidence.zip`为轻量证据包，不含模型权重或优化器状态。

2026-10-21最终复核：机制队列已正常结束，两组各850步训练、24项评测、4280条输出全部通过核验。作者格式配方500/850步均值49.84%/46.30%，去除答案依赖后46.09%/44.35%，基座45.55%。方法主线得到本轮结果支持，纯格式机制解释未被充分验证。详见 [最终机制报告](MECHANISM.md)、[完成后复核](artifacts/mechanism/completion_reaudit.json)、[状态](artifacts/mechanism/status.json)。临时合并模型已清理，磁盘余量约6.1GiB；下述进度为历史阶段记录。

2026-10-19阶段更新：机制实验第一组已完成850步训练和12项评测，2140条输出核验通过。作者格式配方500步/850步六项均值49.84%/46.30%，基座45.55%。[第一阶段完整结果](MECHANISM_STAGE1.md)、[阶段审计](artifacts/mechanism/stage1_audit.json)。第二组去除答案依赖的训练已自动启动，实际进度见下面状态文件。

2026-10-18当前后台任务：推进核心机制消融，两组独立训练各850步，再对固定500/850步执行24项六基准评测。第一组沿用作者format-only配方；第二组只去掉length奖励中的答案依赖。详情：[MECHANISM.md](MECHANISM.md)、[状态](artifacts/mechanism/status.json)、[预先固定计划](artifacts/mechanism/plan.json)、[奖励核验](artifacts/mechanism/reward_semantics_audit.json)。screen：`tina-mechanism`；日志：`logs/mechanism_queue.log`。原观察命令自动显示当前机制实验：`bash /root/autodl-tmp/Tina/reproduction/watch_core.sh`；加`--suite core`查看此前完成的六项评测。

2026-10-18最终状态：核心队列全部完成，5个模型×6项基准、5350条输出核验通过，Open-RS2独立训练450步完成。两张GPU已释放，磁盘剩余约8.8GiB。总览：[CORE_EXPERIMENTS.md](CORE_EXPERIMENTS.md)、[结果数据](artifacts/core_suite/results.json)、[最终审计](artifacts/core_suite/final_audit.json)、[队列状态](artifacts/core_suite/status.json)。中文表格面板：`bash /root/autodl-tmp/Tina/reproduction/watch_core.sh`。下述阶段状态与磁盘余量均为历史快照。

本地Open-RS3 500步/850步、Open-RS2 450步的六项均值分别为49.46%/47.69%/47.53%，基座45.55%，作者500步权重46.57%。500步均值接近论文49.45%，但逐项结果与相对基座增益未完全复现；完整限制见总报告。

新增RS2权重与断点：[grpo_open_rs2_core](artifacts/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/grpo_open_rs2_core/)，滚动保留350/400/450步三份；[训练审计](artifacts/core_suite/rs2_training_audit.json)、[配置](reproduction/open_rs2_core.yaml)、[训练日志](logs/core_rs2_train.log)。原RS3的17份checkpoint全部保留。GPQA数据来源及字节一致性：[provenance](artifacts/core_suite/gpqa_public_provenance.json)。

早期训练轨迹校准已完成：[TRAJECTORY.md](TRAJECTORY.md)、[状态](artifacts/trajectory/status.json)。

现有850步官方协议补评已完成：[ENDPOINT850.md](ENDPOINT850.md)、[状态](artifacts/endpoint850/status.json)、[570条逐题审计](artifacts/endpoint850/final_audit.json)。MATH-50085.4%、AIME2440.0%、AMC2377.5%；GPU已空闲，数据盘余量约14GB。未启动长程重训或作者50步续训。

最新差异归因与外部复现证据：[ATTRIBUTION.md](ATTRIBUTION.md)；单变量控制数据位于 `artifacts/attribution/`，日志为 `logs/attribution_*_merged42.log`。

三 seed MATH-500 复核全部完成：[结果](artifacts/attribution/multiseed_results.json)、[4500条逐题核验](artifacts/attribution/multiseed_final_audit.json)。新增六个目录 `artifacts/attribution/multiseed/seed{42,2026}/{base,official500,local500}/` 均有可读 `predictions.jsonl`。GPU已释放，数据盘余量约20GB。

> 最新复核已完成：[RECHECK.md](RECHECK.md)、[9项对照汇总](artifacts/recheck/results_summary.json)、[1710题最终核验](artifacts/recheck/final_audit.json)。官方路径下，基座/本地500步的MATH-500为83.4%/83.2%，AIME24为33.33%/43.33%，AMC23为75.0%/80.0%。下述初轮分数属于已标记的历史简化判分。

工作目录：`/root/autodl-tmp/Tina`。主训练已完成 850 步，训练耗时 25.96 小时。
训练和第 450、850 步 MATH-500 评测均已完成。入口：[最终报告](REPORT.md)、[完整对照数据](artifacts/comparison.json)、[阶段状态](artifacts/pipeline_status.json)。

初轮历史简化判分记录：基线84.6%，第450步83.8%，第850步82.8%。这不是最终六项主表，后续已修正判分器和推理协议；最终结果以本页顶部的完成报告为准。

## 权重与断点

| 用途 | 位置 |
|---|---|
| 训练结束 LoRA（850步，非论文主表选择） | [grpo_open_rs3](artifacts/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/grpo_open_rs3/) |
| 第 450 步 LoRA 与完整续训状态 | [checkpoint-450](artifacts/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/grpo_open_rs3/checkpoint-450/) |
| 第 500 步 LoRA 与完整续训状态（官方Open-RS3主表检查点） | [checkpoint-500](artifacts/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/grpo_open_rs3/checkpoint-500/) |
| 第 850 步 LoRA 与完整续训状态 | [checkpoint-850](artifacts/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/grpo_open_rs3/checkpoint-850/) |
| 基础模型（只存一份） | [base](artifacts/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base/) |

正式断点每 50 步保存一次，共 17 份，全部保留，总占用约 9.55 GiB。
根目录中的 `adapter_model.safetensors` 是 LoRA，使用时需要基础模型；它不是独立的完整模型权重。

## 训练记录

- [训练曲线 PNG](artifacts/metrics/training_curves.png) / [SVG](artifacts/metrics/training_curves.svg)
- [优化指标 PNG](artifacts/metrics/optimization_curves.png) / [SVG](artifacts/metrics/optimization_curves.svg)
- [850 步原始指标 CSV](artifacts/metrics/training.csv)
- [前后 50 步均值摘要](artifacts/metrics/training_summary.json)
- [训练完整日志](logs/main.log)
- [离线 W&B 记录](logs/wandb/)

奖励和格式曲线属于训练指标，不能代替测试集评测。

## 审计与评测

- [新复核记录](RECHECK.md) / [复核队列状态](artifacts/recheck/queue_all.json)
- [官方与本地训练曲线叠图](artifacts/recheck/official_vs_local_training.png)
- [官方500步adapter](artifacts/recheck/official_rs3/checkpoint-500/) / [官方500步合并模型](artifacts/recheck/merged_official500/) / [本地500步合并模型](artifacts/recheck/merged_local500/)
- [训练与 17 个 checkpoint 的文件审计](artifacts/training_audit.json)
- [可重跑的训练核验脚本](reproduction/audit_training.py) / [本次核验执行日志](logs/audit_training.log)
- [最终 LoRA 参数审计](artifacts/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/grpo_open_rs3/adapter_audit.json)
- [基线每题输出](artifacts/eval/base.jsonl) / [摘要](artifacts/eval/base.summary.json) / [审计](artifacts/eval/base.audit.json)
- [评测目录](artifacts/eval/)：完整输出使用 `checkpoint450.jsonl`、`checkpoint850.jsonl` 命名。
- [第 450 步评测日志](logs/checkpoint450_math500.log)
- [第 850 步评测日志](logs/checkpoint850_math500.log)
- [数据重叠审计](artifacts/data_overlap_audit.json)：训练集与测试集有 6 道完全相同的问题，最终报告同时列出完整 500 题和排除重复题后的结果。

重跑文件核验：`source reproduction/env.sh && python reproduction/audit_training.py`。该核验检查训练记录、文件存在性和状态文件可读取性，不执行恢复后的训练步，也不代替推理准确率评测。

## 复现配置

- [运行说明、保存策略与本地改动](reproduction/README.md)
- [训练配置](reproduction/open_rs3.yaml)
- [上游提交版本](reproduction/upstream_commit.txt)
- [安装依赖版本](reproduction/installed.freeze.txt)
- [模型与数据版本](artifacts/download_manifest.json)

`smoke_open_rs3`、`smoke_full_length` 是训练前的验证产物，正式模型使用 `grpo_open_rs3`。当前保留这些产物和日志，便于追溯。
