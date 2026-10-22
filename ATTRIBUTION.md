# 与作者的差异及第三方复现证据

2026-10-18最新：六项评测已全部完成，详见 [CORE_EXPERIMENTS.md](CORE_EXPERIMENTS.md)。本地RS3 500步均值49.46%，基座45.55%，作者500步权重46.57%；新增独立RS2 450步47.53%。RS3总均值接近论文，但本地基座更高、逐项结果不同，增益+3.91个百分点没有匹配论文约+8.27个百分点；RS2比论文低约3.07个百分点。这些单次采样结果不推翻下述MATH-500多seed限制，也没有确定唯一差异根因。下文为此前归因调查记录。

更新：2026-10-17。本报告区分“已观察到的差异”和“已证明的性能原因”。上一轮九项评测见 [RECHECK.md](RECHECK.md)。

**最新850步补评完成：** 保留原独立训练权重，在此前500步同一官方协议下，MATH-50085.4%、AIME2440.0%、AMC2377.5%；同协议基座为83.4%、33.33%、75.0%，论文850步为83.8%、26.67%、75.0%。三项本次均高于基座及论文对应分数，570条逐题核验通过。支持保留当前训练继续研究；单次采样、三项不代表稳定提升或六项全文复现。详见 [ENDPOINT850.md](ENDPOINT850.md)。下面三seed均值属于此前500步复核，不是850步结果。

**最新状态：三 seed 复核全部完成，4500 条逐题输出核验通过，其中新增 3000 条。** MATH-500 三 seed 均值：基座 83.73%，作者500步 84.33%，本地500步 83.47%。本地相对作者的差值分别为 −2.4、0、−0.2 pp，平均 −0.87 pp；原先单次 12 题差距未稳定重现。本地相对基座均值为 −0.27 pp，仍未观察到本任务的稳定提升。三个推理 seed、一个训练 seed，尚不能确定唯一训练根因。后台队列正常结束，两张 GPU 已释放。逐题核验：[multiseed_final_audit.json](artifacts/attribution/multiseed_final_audit.json)。

已锁定初轮评测的三个问题：判分器不等价、漏测 Open-RS3 最佳 500 步、推理协议不一致。它们使最初的“越训越差”结论不可靠，但还没有证明本地训练与作者之间剩余差距的唯一原因。在统一官方路径后，MATH-500 作者权重为 428/500，本地为 416/500；这是一次采样下的 12 题差距。

新增 1000 题单变量控制已完成：其他旧协议设置固定，仅动态 LoRA → BF16 合并，本地 426→425，作者 414→414。合并方式单独改变未恢复作者在完整官方路径下的 428 题成绩。因此，上轮分数/排名变化不能主要归因于合并方式；尚有 seed、批处理/并行、上下文协议及交互需要分离。固定官方协议的多 seed 复核结果见文末。

## 已确认的差异

| 项目 | 作者公开记录 / 官方代码 | 本地初轮 | 当前证据与解释 |
|---|---|---|---|
| checkpoint | Open-RS3 主结果选 500 步 | 仅评测 450/850 | 已补 500；不能要求 850 优于中间检查点 |
| 判分器 | LightEval 0.8.1 仓库原始任务 metric | 简化 math-verify first_match | 已用官方 metric 重判；存在误判，重判后旧协议下下降仍在 |
| 推理权重 | BF16 合并模型 | vLLM 动态 LoRA | 两份完整 500 题单变量对照已完成；准确率净变化 −1 / 0 题 |
| 推理采样 | 官方单次 CLI：引擎默认 seed 1234、未设逐请求 seed | 引擎和逐请求均 seed 42 | 同为随机采样，一次分数不能代表跨 seed 均值 |
| 推理并行 | 2 GPU data parallel，max_model_len 32768 | 1 GPU，max_num_seqs 32，max_model_len 33792 | 上轮官方路径已对齐；批处理变化也可能改变数值与生成轨迹 |
| 训练固定提示词回调 | 每 50 步 rank 0 generate | 关闭以避开额外显存/生成开销 | 会消耗随机数，影响后续轨迹；但第一步就已不同，不能作为最初分歧的唯一解释 |
| 训练 vLLM 显存比例 | 0.4 | 0.25 | 原值在本机长序列反向 OOM，降后完整跑通；缓存容量/调度可能变化，尚未证明影响精度 |
| transformers | 4.49.0 | 4.50.0 | 已比较相关 Qwen2 forward、Trainer backward，未发现相关训练语义变化 |
| PEFT / accelerate | 0.14.0 / 1.5.1 | 0.15.0 / 1.5.2 | 相关 merge/unmerge/delta、backward 相同，不能只凭版本号认定根因 |
| xgrammar | 0.1.15 | 0.1.17 | 无约束数学生成未启用 grammar；暂未发现关联证据 |
| 硬件 | 论文说明使用 L40S / 部分 RTX6000 Ada | 2× RTX4090 48GB | 本次作者 W&B run 的实际 GPU metadata 未能取到；不能断言该 run 的具体 GPU，更不能直接归因硬件 |

基座 revision、训练数据 Parquet、7000 条样本、LoRA r32/alpha128/目标模块、format+cosine 权重、batch6×2GPU/gradacc4/6 generations、学习率和调度器均已核对。这里以作者公开 run 的实际 config 为准，而非把论文汇总超参数表当成每个 run 的精确配置。来源：[作者 W&B](https://wandb.ai/upup-ashton-wang-usc/Tina/runs/3t1h5abx)、本地 `artifacts/recheck/3t1h5abx_wandb_config.json`、`official_requirements.txt`、`training_dataset_revision_parity.json`。

另外与最初公开 commit `e16606f` 比较 AST：format/cosine 奖励、make_conv_for_grpo、RepeatRandomSampler、固定提示词 callback 均相同。GRPOTrainer 仅 `_get_train_sampler` 接口包装变化；同一数据集和 seed 下主体相同，未发现核心损失/rollout/合并同步逻辑在发布后改变。最初公开提交晚于作者原训练日期，不能核验未公开的发布前代码。证据：[upstream_release_parity.json](artifacts/attribution/upstream_release_parity.json)。

作者权重在本机官方路径下 MATH-500 为 85.6%，接近论文的 85.2%。因此“本机任何路径都测不出作者成绩”已经排除；剩余问题是本地训练所得权重、采样方差及其交互。AIME24/AMC23 本地分别 43.33%/80.0%，本次基座 33.33%/75.0%，小样本的配对检验不显著，不能据此声称稳定增益。

## 分歧从什么时候开始

作者和本地的第 1 步已经不同：

| 第 1 步指标 | 作者 | 本地 |
|---|---:|---:|
| 平均 completion tokens | 2931.25 | 3201.94 |
| format reward | 0.3750 | 0.2708 |
| cosine reward | -0.17018 | -0.18063 |
| grad norm | 0.04999 | 0.04419 |
| learning rate | 6.6667e-9 | 6.6667e-9 |
| KL | 0 | 0 |

固定提示词回调首次在第 50 步末尾触发，所以不能解释第 1 步差异。该证据将调查范围推进到初始 rollout/随机数/数值计算阶段，但作者未公开逐条训练 rollout token、初始权重和完整 RNG 状态，不能仅由聚合日志判断是哪个环节先变。证据：[early_divergence.json](artifacts/attribution/early_divergence.json)。

## 单变量推理对照

固定原始直接 vLLM 协议：同 500 题顺序、同 prompt token IDs、temperature 0.6、top_p 0.95、引擎及逐请求 seed 42、单卡、max_num_seqs 32、max_model_len 33792、max_new_tokens 32768、官方 metric。仅将动态 LoRA 改成 BF16 合并模型。本地/作者各一份，总计 1000 条新生成。

所有 500 条 prompt 在基座及两份合并模型上的 token IDs 相同；generation_config 除 transformers_version 外相同。计划与输入哈希：[controlled_merge_plan.json](artifacts/attribution/controlled_merge_plan.json)。两个任务均以独立 detached screen 启动，SSH 断开继续运行。

<!-- CONTROLLED_RESULTS -->
两个完整 500 题控制已完成，输入顺序及官方判分核验通过。

| 权重 | 动态 LoRA（官方重判） | BF16 合并 | 差值 | 新增答对 / 新增答错 | 生成文本改变 |
|---|---:|---:|---:|---:|---:|
| 本地500 | 426/500 (85.2%) | 425/500 (85.0%) | -0.2 pp | 26 / 27 | 497/500 |
| 作者500 | 414/500 (82.8%) | 414/500 (82.8%) | +0.0 pp | 27 / 27 | 498/500 |

这里仅改变权重加载路径；结果可用于判断这一改变在固定条件下的影响。它没有恢复训练回调或训练依赖，也不能将官方路径下本地与作者的差距全部归因于合并方式。

完整统计、输出 SHA256 与逐题得失：[controlled_merge_results.json](artifacts/attribution/controlled_merge_results.json)。
<!-- /CONTROLLED_RESULTS -->

## 别人的实际复现反馈

本轮抓取官方仓库 44 个公开 fork 元数据，对有更新的候选比较上游差异，进一步检查相关文件树、提交日志、模型卡和官方全部 15 个 issue/PR 条目及可访问评论。未逐个运行所有 fork，也没有访问私人日志。检索范围和原始响应保存在 `artifacts/attribution/`、`artifacts/recheck/github_issues.json` 及 `issue_*_comments.json`。

| 来源 | 实际反馈 / 产物 | 能证明什么 |
|---|---|---|
| [官方 issue #6](https://github.com/shangshang-wang/Tina/issues/6) | 2×4090 用户调整 batch/accum 后，24 小时后 ETA 仍达 463 小时；作者建议固定 accum、调整 batch | 相同 GPU 系列确有训练耗时反馈；该用户跑 curated_deepscaler，不是本次 Open-RS3，也未给独立精度结果 |
| [官方 issue #2](https://github.com/shangshang-wang/Tina/issues/2) | 依赖/采样器接口不兼容，作者确认 TRL 0.15.2，讨论中使用 vLLM 0.7.2 | 支持固定原始依赖；本地 TRL/vLLM 已符合 |
| [官方 issue #10](https://github.com/shangshang-wang/Tina/issues/10) | 单 A6000、STILL recipe，固定 prompt callback 的 generate 中 NaN/assert | 公开的训练回调故障反馈；与本次双卡完整训练的情况不同，不能直接解释本地精度 |
| [cerisara/Tina 提交](https://github.com/cerisara/Tina/commit/13ba89e94895c9ac6d6e8bd7a1f94b71822bf91d) | 提到长生成、VRAM 限制、估算训练 1200 小时；另有 flash-attention 崩溃提交 | 真实实验记录；该 fork 加了 4bit/ladder 等修改，未找到独立 benchmark 成绩 |
| [第三方 Open-RS2 权重](https://huggingface.co/whalexdfsa/open-rs2-lora-grpo) / [代码](https://github.com/LYF22034/open-rs2-grpo-lora) | 声明 2×A100 80GB、format+accuracy、训练到 600 步，公开 12 份 adapter | 确有第三方训练产物；是 Open-RS2；卡中 50.60% 明确引用论文，不是其自身测得成绩。链接的 W&B API 当前未返回公开 run |
| [同作者 GPRA 权重](https://huggingface.co/whalexdfsa/open-rs2-GPRA) | 在 GRPO+LoRA 上加 PRIME 过程奖励，声明约 150s/step，训练至 500 | 修改方法的训练反馈；不能算原始 TINA 的性能复现 |
| [pum1rng/Tina](https://github.com/pum1rng/Tina)、[LinesHogan/Tina](https://github.com/LinesHogan/Tina) | 前者改 rank 等；后者探索采样器、PiSSA/SALT 等 | 有后续研究代码；在所检索公开文件中未找到可核验的原始 TINA 独立成绩 |

因此，可以看到别人的复现反馈，也确实应该用它们定位问题。但目前查到的材料不足以支持“第三方已经稳定复现 Open-RS3 的完整论文提升”，也没有发现与我们相同配置的明确降分问题及经过验证的修复。

## 上游 BF16 合并问题是否就是根因

最新 TRL 的确有 [issue #6688](https://github.com/huggingface/trl/issues/6688)、[issue #7423](https://github.com/huggingface/trl/issues/7423) 和 [PR #7427](https://github.com/huggingface/trl/pull/7427)，讨论向 vLLM 同步时原地 merge/unmerge 引起冻结基座漂移。这是重要的机制线索，但不是 TINA 的第三方 benchmark 复现。

现代报告使用的 PEFT 版本与本地 0.15 不同。本地 0.15 的 merge/unmerge 没有该报告中一加一减 dtype 不对称的代码。用真实本地 checkpoint500 的两层权重、实际 PEFT Linear、BF16 和 FP32 adapter 各重复 1/10/100/1000 次：存在极小的首次舍入漂移，但固定 adapter 下此后不继续增长。不能照搬“1000 次持续累积漂移”作为本次训练的已证实根因。

该 CPU 抽样诊断没有复现训练期间每步改变的 adapter，也没有原训练进程的内存基座快照。原始基座文件 SHA 不变只证明磁盘文件未改，不证明训练内存中的基座位级不变。证据：[merge_cycle_audit.json](artifacts/attribution/merge_cycle_audit.json)、`artifacts/recheck/merge_numeric_audit.json`。

## 仍需验证的内容

1. 多 seed 检查已完成：预先固定引擎 seed 1234、42、2026，1234 复用原始完整输出，新增六项，不挑高分 seed。计划：[multiseed_plan.json](artifacts/attribution/multiseed_plan.json)，完成状态：[multiseed_status.json](artifacts/attribution/multiseed_status.json)。此前 12 题差距未稳定重现，不能再把这个单次差距当成已证实的训练故障幅度。
2. 若继续定位训练机制，需要先做短程训练控制，恢复作者依赖和固定回调，并导出每步 prompt IDs、rollout token 哈希、奖励、RNG/采样配置。当前尚未开展该训练控制，也未启动新一轮 26 小时训练。不能因为三 seed 的均值差距变小，就反推所有训练实现完全等价。
3. 保留原始 checkpoint 和生成结果。作者未提供的初始 RNG/rollout 证据，不能用推测补齐；上述实验还不足以证明某个硬件或版本是唯一原因。


## 预先固定的多 seed MATH-500 复核

保持官方 BF16 合并、双卡 LightEval 协议，固定引擎 seed 1234、42、2026；逐请求 seed 均为 None。1234 复用此前完整输出，新增六项。

| 模型 | seed1234 | seed42 | seed2026 | 均值 ± 样本标准差 |
|---|---:|---:|---:|---:|
| base | 83.40% | 83.80% | 84.00% | 83.73% ± 0.31 pp |
| official500 | 85.60% | 83.20% | 84.20% | 84.33% ± 1.21 pp |
| local500 | 83.20% | 83.20% | 84.00% | 83.47% ± 0.46 pp |

这是三个推理 seed、一个训练 seed。它能检查推理采样波动，不能单独证明训练差异的原因，也不能把三个 seed 的重复题当成 1500 道独立题。

完整数据：[multiseed_results.json](artifacts/attribution/multiseed_results.json)。
