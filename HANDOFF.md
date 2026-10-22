# TINA 实验交接入口

最后整理：2026-10-21。项目路径：`/root/autodl-tmp/Tina`。

## 接手时的状态

- 本轮计划的训练和评测均已完成，无需继续等待后台任务。
- 原RS3训练850步，RS2训练450步；两组格式机制对照各850步。
- 核心评测5个模型×6任务，共5350条输出；机制对照4个检查点×6任务，共4280条输出。二者合计54份不同的模型/任务评测、9630条输出，不把历史复用结果重复累计。
- 历史三推理seed MATH-500、动态LoRA/合并控制、早期训练轨迹核验另有独立记录，不包含在上述9630条中。
- 所有既有权重、优化器状态、逐题输出和日志保留；仅机制评测专用临时合并模型已按计划清理。
- 不要为了“继续观察”重新启动训练脚本。已完成的训练目录应作为冻结结果保留；新实验使用新目录。

## 先读什么

| 内容 | 入口 |
|---|---|
| 总结、主结果、最终状态 | [REPORT.md](REPORT.md) |
| 五个模型的六项基准、论文对应值 | [CORE_EXPERIMENTS.md](CORE_EXPERIMENTS.md) |
| 两组格式机制对照的全部结果与解释 | [MECHANISM.md](MECHANISM.md) |
| 与作者的差异、外部复现记录、控制实验 | [ATTRIBUTION.md](ATTRIBUTION.md) |
| 初轮评测错误及修正过程 | [RECHECK.md](RECHECK.md) |
| 早期训练轨迹对照 | [TRAJECTORY.md](TRAJECTORY.md) |
| 全部文件索引 | [ARTIFACTS.md](ARTIFACTS.md) |
| 搬迁、环境、校验和、恢复操作 | [MIGRATION.md](MIGRATION.md) |

## 最终成绩

以下均为同一套BF16合并、LightEval 0.8.1、双卡数据并行、引擎seed1234的六任务不加权均值。

| 模型 | 六项均值 | 相对本地基座（百分点） |
|---|---:|---:|
| 基座 | 45.55% | 0.00 |
| 本地RS3 500步 | 49.46% | +3.91 |
| 本地RS3 850步 | 47.69% | +2.14 |
| 本地RS2 450步 | 47.53% | +1.98 |
| 作者RS3 500步权重、本机评测 | 46.57% | +1.02 |
| 作者公开格式配方、本地500步 | 49.84% | +4.29 |
| 作者公开格式配方、本地850步 | 46.30% | +0.74 |
| 去除长度奖励答案依赖、本地500步 | 46.09% | +0.54 |
| 去除长度奖励答案依赖、本地850步 | 44.35% | −1.20 |

精确分数、各任务正确数和原始文件映射见 `migration/final_scores.csv`、`migration/evaluation_registry.json`；这里百分比仅保留两位。

## 实验过程与关键决策

| 阶段 | 做了什么 | 产物和结论 |
|---|---|---|
| 环境与预检 | 双卡48GB，固定代码/模型/数据版本；全长度反向预检 | 原vLLM预算0.4触发OOM，改0.25；保留batch/长度/奖励/损失 |
| 初轮RS3 | 基座独立LoRA-GRPO至850步，约25.96小时；每50步保存 | `grpo_open_rs3`，原17份断点保留；初轮简化判分不作为最终主结果 |
| 判分与协议复核 | 对初轮输出重判，补官方BF16合并+LightEval；补漏测的RS3 500步 | `RECHECK.md`，修正判分器不等价、检查点错位和推理协议差异 |
| 差异归因 | 公共作者日志、依赖代码差异、动态/合并单变量控制、三推理seed MATH-500 | `ATTRIBUTION.md`；只改变合并方式未解释原差距；MATH单次差距不稳定 |
| 训练早期对照 | 当前环境/作者核心版本环境各短程对照，核对数据和输入token | `TRAJECTORY.md`；没有找到单一依赖版本即可恢复作者轨迹的证据 |
| 六项主评测 | 补齐基座、本地500/850、作者500；独立训练RS2至450步并评测 | `CORE_EXPERIMENTS.md`，5350条输出审计通过 |
| 格式机制核验 | 发现官方format-only的length奖励仍验答案；先做固定文本改参考答案的反事实核验 | `artifacts/mechanism/reward_semantics_audit.json`、`group_advantage_counterfactual.json` |
| 格式机制训练 | 官方五奖励配方 vs 仅去除length答案依赖；各850步，固定500/850评测 | 各约26.5小时，24评测/4280输出完成；结果见 `MECHANISM.md` |
| 完成后复核 | 重读原始Parquet、核对题目/提示词/答案/顺序和分数；核验权重与代码哈希 | `artifacts/mechanism/completion_reaudit.json`；全部通过 |

结论口径：方法主线和本轮平均分收益得到支持。公开format-only配方不是真正无正确性监督；去除该依赖后均分下降，不能将原配方成绩直接解释为纯格式学习充分性的证据。实验不是对每项、每seed或整篇所有消融的穷尽验证。

## 训练配方

共同配置：1.5B基座，LoRA rank32/alpha128/dropout0.05，七个投影模块，BF16；学习率1e−6，cosine_with_min_lr/min_lr_rate0.1，调度总步数1500，warmup150步；每卡batch6，两卡，梯度累积4，每题6个生成，训练seed42，temperature0.7，prompt512/completion3584。

| 运行 | 奖励 | 停止步数 | 配置 |
|---|---|---:|---|
| RS3 | format + 2×cosine | 850 | `reproduction/open_rs3.yaml` |
| RS2 | format + 2×accuracy | 450 | `reproduction/open_rs2_core.yaml` |
| 官方格式配方 | format/tag_count/length/reasoning_steps/repetition_penalty，各权重1 | 850 | `reproduction/format_mechanism.yaml` |
| 无答案依赖格式对照 | 同上，仅length使用不读取答案的长度归一化 | 850 | 同上，加 `TINA_LABEL_FREE_LENGTH=1`，实现见 `reproduction/mechanism_train.py` |

本地共同设置：vLLM预算0.25；关闭固定题生成回调；原调度1500步保持不变，由停止回调在450/850步结束。训练vLLM默认引擎seed0，与训练seed42、评测seed1234是不同参数。

## 评测协议与数据

- 官方任务代码：`scripts/eval/run_eval_custom_tasks.py`，实际代码哈希写在每项manifest中。
- 六任务题数：AIME24 30，AIME25 30，AMC23 40，MATH500 500，GPQA Diamond 198，Minerva 272。每模型共1070题。
- BF16 `merge_and_unload(safe_merge=True)`，双卡data parallel；temperature0.6、top_p0.95、最多32768新tokens、上下文32768，零样本、每题一个输出。
- GPQA使用公开CSV，Git blob与原HF文件一致；创建题目之前固定prompt_seed1234，保证各模型选项顺序一致。来源、SHA和原文件oid见 `artifacts/core_suite/gpqa_public_provenance.json`。
- 原训练集7000题，保留官方配方；已记录6道与MATH500完全重合的问题。早期报告另有去重成绩，最终六任务主表使用完整任务集。
- 初轮 `artifacts/eval/` 的legacy分数不是最终六项主表。三推理seed结果也不能混进单seed六项表格。

## 权重位置

共同前缀：`artifacts/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/`。

| 权重 | 位置 | 能否直接续训 |
|---|---|---|
| 原基座 | `base/` | 基座权重 |
| RS3 500/850 | `grpo_open_rs3/checkpoint-500/`、`checkpoint-850/` | 含完整DeepSpeed/优化器状态 |
| RS2 450 | `grpo_open_rs2_core/checkpoint-450/` | 含完整状态；滚动保留350/400/450 |
| 格式配方850 | `grpo_official_format_core/checkpoint-850/` | 含完整状态；保留800/850 |
| 无答案依赖850 | `grpo_label_free_format_core/checkpoint-850/` | 含完整状态；保留800/850 |
| 格式配方500 | `artifacts/mechanism/official_format/adapter500/` | adapter快照，可推理；不含优化器状态 |
| 无答案依赖500 | `artifacts/mechanism/label_free_format/adapter500/` | adapter快照，可推理；不含优化器状态 |
| 作者RS3 500 | `artifacts/recheck/official_rs3/checkpoint-500/` | 作者adapter对照 |

目录根部的 `adapter_model.safetensors` 不是完整模型，使用时须加载对应基座。机制阶段两份临时合并模型已清理，可从上述adapter重建。

## 原始数据入口

每项评测目录保留 `summary.json`、`manifest.json`、`details/**/*.parquet`，以及相应可读输出。54项主表的具体目录无需猜测，直接读取 `migration/evaluation_registry.json` 的 `summary_path`、`parquet_paths`、`prediction_paths`。

训练：各模型目录 `trainer_state.json` 为逐步训练指标，`resources_rank*.jsonl` 为资源采样，`logs/*train*.log`/`logs/main.log` 为原始日志；机制两组还保留 `artifacts/mechanism/<组名>/metrics.jsonl`。曲线见 `artifacts/metrics/`、`artifacts/recheck/official_vs_local_training.*`、`artifacts/mechanism/training_comparison.*`。

## 本地文件的版本与完整性

上游提交：`106376c7c3388a30ae7769087276534c47cf0b0c`。本地未提交修改和新增脚本同样属于实验，已导出 `migration/local_changes.patch`、`git_status.txt`。不能只克隆上游来替代搬迁。

迁移工具导出全部实验文件的SHA-256清单、文件大小、环境快照、外部依赖和原始评测目录映射。具体覆盖范围、验证命令见 [MIGRATION.md](MIGRATION.md)。
