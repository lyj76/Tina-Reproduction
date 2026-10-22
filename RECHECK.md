# TINA 复核记录（三项官方路径对照完成）

2026-10-18：后续已补齐五个模型的六项基准并完成Open-RS2独立450步训练，5350条输出核验通过。最新全表、论文对照及限制见 [CORE_EXPERIMENTS.md](CORE_EXPERIMENTS.md)，本页保留初轮复核过程。

最新850步补评已完成：MATH-50085.4%、AIME2440.0%、AMC2377.5%，三项均高于本次同协议基座及论文850步分数。570条逐题核验通过；仍为单次采样、三项基准。详见 [ENDPOINT850.md](ENDPOINT850.md)。

后续归因调查见 [ATTRIBUTION.md](ATTRIBUTION.md)：包含第一步训练分歧、单变量合并控制和扩大后的第三方检索。本页的“GPU空闲”仅指下述九项评测结束时。

最新多 seed MATH-500 复核已完成：1234/42/2026 三次均值为基座83.73%、作者500步84.33%、本地500步83.47%。本地对作者的差值为−2.4/0/−0.2 pp，原单次12题差距未稳定重现；仍没有证明本地MATH-500稳定提升。4500条输出核验通过，详见 [最终审计](artifacts/attribution/multiseed_final_audit.json)。

2026-10-17：训练、判分、上游代码及公开复现记录审计完成；两份动态LoRA控制和9项官方路径评测全部完成。最终逐题核验通过：1710条官方路径输出，三个模型的题目、提示词、参考答案与顺序完全一致。

本次结论：初轮“训练后全面下降”的判断不成立。本地500步模型在官方路径下，MATH-500相对基座少1题，AIME24多3题，AMC23多2题。已经观察到两项竞赛题的改善，但MATH-500仍未复现作者权重的增益。所有结果仍属于单次采样，不能声称跨seed稳定提升或全文复现完成。

## 已确认的问题

1. 原 `reproduction/evaluate.py` 使用 math-verify `extraction_mode="first_match"` 的简化判分，与官方 LightEval 0.8.1 的任务判分器不等价。部分 Markdown 加粗的最终答案被误判。用仓库未经修改的 `latex_gold_metric` 对原始输出重新判分：

| 原输出 | 原判分正确数 | 官方判分正确数 | 官方判分准确率 |
|---|---:|---:|---:|
| base | 423/500 | 432/500 | 86.4% |
| checkpoint450 | 419/500 | 423/500 | 84.6% |
| checkpoint850 | 414/500 | 419/500 | 83.8% |

逐题变更见 `artifacts/recheck/*.regrade.json`。判分错误不能解释下降，官方重判后下降仍存在。

2. 此前只测 450/850 步，漏测 Open-RS3 官方最佳 checkpoint 500。450 是 Open-RS2 最佳 checkpoint。作者在 [issue #7](https://github.com/shangshang-wang/Tina/issues/7) 说明主表使用训练过程中的最佳 checkpoint；[官方 Open-RS3 模型卡](https://huggingface.co/Tina-Yi/R1-Distill-Qwen-1.5B-Open-RS3) 明确指定 500。

3. 原评测使用动态 LoRA 和显式采样 seed 42；官方使用 BF16 合并权重和 LightEval。官方单次 CLI 没有显式 sampling seed，LightEval 0.8.1 引擎默认 seed 1234。原分数只能作为该本地条件下的对照。

## 训练与官方日志对比

从作者公开 [W&B Open-RS3 run](https://wandb.ai/upup-ashton-wang-usc/Tina/runs/3t1h5abx) 下载 config、全部 1246 步训练指标及 requirements.txt。公开 config 与本地的 batch6、6 generations、grad accumulation4、lr1e-6、1500 步 scheduler 等一致。依赖有差异：官方 transformers4.49.0、PEFT0.14.0、accelerate1.5.1；本地分别4.50.0、0.15.0、1.5.2。Torch/vLLM/TRL/DeepSpeed/math-verify 主版本匹配。

| 800–850 步均值 | 官方 | 本地 |
|---|---:|---:|
| KL | 0.2957 | 0.2716 |
| grad norm | 0.6128 | 0.6117 |
| format reward | 0.6511 | 0.6234 |
| cosine reward | -0.0809 | -0.0824 |

官方与本地 checkpoint500 的 LoRA B L2 范数约0.2797、0.2771；均为392个BF16 LoRA张量、36,929,536个参数。完整数据见 `artifacts/recheck/training_comparison.json`。指标接近不证明实现逐项等价，但暂不支持“训练明显跑坏”的判断。

![官方与本地训练动态叠图](artifacts/recheck/official_vs_local_training.png)

FP64归约计算的权重方向余弦相似度：LoRA A为0.999992，LoRA B为0.912678，见 `artifacts/recheck/adapter_comparison.json`。训练Parquet与2025年3月24日版本LFS SHA完全相同，见 `training_dataset_revision_parity.json`。

对照相关依赖源码：transformers4.49/4.50的Qwen2ForCausalLM.forward仅文档变化；Trainer的梯度累积/backward路径一致；accelerate1.5.1/1.5.2的backward一致；PEFT0.14/0.15的merge、unmerge、get_delta_weight一致，普通LoRA前向的输入转换默认行为相同。原始文件与逐函数diff保存在 `artifacts/recheck/source_versions/`，暂未发现能解释结果差异的训练语义变更。

本地关闭了官方每50步的固定提示词生成callback。该callback在训练模式下调用generate，会消耗随机数；关闭它属于采样轨迹差异，即使损失公式不变，也不能声称与官方逐步位级复现。它不直接参与奖励或优化器更新。

数值诊断抽样407万个权重元素：BF16合并后约9.13%的元素相对基座改变，CPU下4个固定前缀的合并/未合并logits确有小差别。该检查不能证明准确率影响，故以完整推理对照为准。单次merge/unmerge也有约0.084%的抽样基座元素出现舍入差异；官方训练器同样使用此实现。因此之前“基础权重未变”的文件审计仅证明磁盘原始基座文件未改写，不能证明训练中内存的基座位级不变。

[ICLR 2026 正式版论文](https://proceedings.iclr.cc/paper_files/paper/2026/file/91be140e1524bb21531508c2a6a85c5e-Paper-Conference.pdf) 表12与初版一致：Open-RS3的500步MATH-500为85.2%，850步为83.8%，400步为82.8%。论文展示的性能并不随训练步数单调增长。

## 外部复现检查

已读取官方全部15个issues/PR及可用评论。讨论主要涉及依赖版本、采样器接口、显存、训练耗时和回调崩溃；没有找到已发布的“本配置训练下降”的确定修复。issue #15 的采样数与seed询问仍无回复。

找到第三方 [LYF22034/open-rs2-grpo-lora](https://github.com/LYF22034/open-rs2-grpo-lora) 和 [权重卡](https://huggingface.co/whalexdfsa/open-rs2-lora-grpo)：声明用2×A100、训练到600步，奖励是format+accuracy。它属于Open-RS2；未找到可核验的独立评测分数，其W&B项目API当前未返回公开run。不能把模型卡引用的论文分数当作第三方实际复现分数。

## 继承自上游的奖励细节

全量7000条训练数据审计见 `artifacts/recheck/reward_audit.json`：14条solution不可抽取，官方代码会无条件给cosine奖励1.0；469条从solution抽取的结果与answer列未被判为等价。后者混有完整答案集合与单项answer、选择题字母、末尾文字答案等不同情况，不能一律计作错标，也不应未经核验就把全部solution替换为answer。

cosine函数用未截断的字符长度，而非tokens；当max_len=3584时，正确输出的奖励在字符长度3584、7168、10752、14336处分别为0.5、1.0、0.5、1.0，确实不是跨整个长度范围单调递减。这些都是原官方实现的行为。当前复现保留它们，不能单凭代码细节认定它们导致本地与官方结果之差。

## 已完成的对照

- 同一旧推理配置：官方500步adapter、本地500步adapter，各500道MATH-500，显式seed42，随后官方判分。
- 官方路径：基础模型、官方500步、本地500步，采用BF16合并权重、LightEval0.8.1、仓库原始task、双卡data parallel、上下文32768、max_new_tokens32768、temperature0.6、top_p0.95；分别评测MATH-500、AIME24、AMC23。每个任务单独运行，保持论文命令的执行方式。
- 新评测使用隔离的 `artifacts/recheck/eval_env`，不修改原训练环境。

后台队列已正常结束，状态 `artifacts/recheck/queue_all.json` 为complete，两张GPU均0MiB占用。日志：`logs/recheck_*.log`。所有17份原checkpoint保留。

## 结论与剩余问题

确认了初轮判分器不等价、漏测Open-RS3最佳500步、推理路径/seed与官方不一致三项问题。训练动态和权重更新接近作者公开记录，未发现本地训练明显跑坏的证据。

同为旧推理配置、官方判分器：基座86.4%，本地500步85.2%，作者500步82.8%。切换到官方合并权重+双卡LightEval后，分别为83.4%、83.2%、85.6%。这说明旧配置下的相对排名不能直接用来判断论文复现；路径对照同时改变了合并方式、batch/并行和采样seed，尚不能把全部变化归因于单个参数。

在新的官方路径下，作者权重MATH-500为85.6%，接近论文主表85.2%；本地为83.2%，仍差2.4个百分点。AIME24本地43.33%、AMC23本地80.0%，高于本次基座33.33%、75.0%。排除6道训练集重合题后，基座与本地MATH-500均为414/494（83.81%）。

逐题配对的McNemar检验：本地对基座MATH-500、AIME24、AMC23的p值分别为1.0、0.375、0.7266；这些单次样本不足以证明稳定收益或稳定退化，也不证明模型等价。下一项有价值的性能核验是预先固定多个采样seed再比较均值；当前没有重训，也没有挑选新的高分checkpoint。

最终审计：[final_audit.json](artifacts/recheck/final_audit.json)。各评测目录新增 `predictions.jsonl`，可直接查看每题问题、生成答案、抽取结果和正确性。

<!-- OFFICIAL_RESULTS -->
## 完整官方路径对照结果

以下均为合并权重、官方任务判分、双卡LightEval的单次采样结果。checkpoint500在看到新评测结果前已预先指定。

| 模型 | MATH-500 | AIME24 | AMC23 |
|---|---:|---:|---:|
| base | 417/500 (83.40%) | 10/30 (33.33%) | 30/40 (75.00%) |
| official500 | 428/500 (85.60%) | 10/30 (33.33%) | 28/40 (70.00%) |
| local500 | 416/500 (83.20%) | 13/30 (43.33%) | 32/40 (80.00%) |

| 对照 | MATH-500差值 | AIME24差值 | AMC23差值 |
|---|---:|---:|---:|
| local500 − base | -0.20 pp | +10.00 pp | +5.00 pp |
| official500 − base | +2.20 pp | +0.00 pp | -5.00 pp |
| local500 − official500 | -2.40 pp | +10.00 pp | +10.00 pp |

原始逐题输出和抽取答案保存在 `artifacts/recheck/official_pipeline_*/<task>/details/`；配置、指标和时间见各目录 `manifest.json`、`summary.json`。

这些对照同时纠正了检查点选择和评测路径；不能把路径切换产生的全部变化归因于某一个推理参数。单次结果也不能证明跨seed稳定收益。
