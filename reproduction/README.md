# TINA 双卡本地复现

最终交接入口：[HANDOFF.md](../HANDOFF.md)。完整迁移、校验和、外部环境依赖、新实验目录与恢复说明见 [MIGRATION.md](../MIGRATION.md)。下文保留最初运行说明；本轮实验已经完成，无需重启原pipeline。

目录：`/root/autodl-tmp/Tina`。独立环境：`/root/autodl-tmp/envs/tina`。
上游版本见 `upstream_commit.txt`，硬件见 `hardware.csv`，模型/数据版本见运行下载后生成的 `artifacts/download_manifest.json`。

## 目标与配置

从 `DeepSeek-R1-Distill-Qwen-1.5B` 开始，在官方 `knoveleng/open-rs` 数据上执行 Open-RS3 的 LoRA + GRPO。
保留当前官方配方：rank 32、alpha 128、dropout 0.05、七类线性投影层、BF16、学习率 1e-6、每卡 batch 6、每题生成 6 条、梯度累积 4、prompt 512、completion 3584、seed 42、temperature 0.7。
奖励为 format + 2 × cosine，KL 系数为 0.04。850 步停止，但保留 max_steps=1500 和 warmup_ratio=0.1 的原始学习率调度。

当前代码配方与早期论文超参数表的采样数/batch 不一致。本次遵循锁定提交的代码配方。
原始 cosine 奖励以字符长度而非 token 数计算，仍保持原实现。不能将预算较短的 smoke 运行当作论文结果。

依赖例外：原锁定的 xgrammar 0.1.16 已无法从 PyPI 获取，改用 0.1.17，满足 vLLM 0.7.2 的 `xgrammar>=0.1.6` 要求。本实验不使用结构化输出/语法约束采样。

## 运行

```bash
cd /root/autodl-tmp/Tina
source reproduction/env.sh
python reproduction/download.py
bash reproduction/train.sh smoke
python reproduction/audit_adapter.py artifacts/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/smoke_open_rs3
bash reproduction/train.sh main
```

`train.sh main` 自动查找自己的最新 checkpoint 恢复。smoke 与正式训练输出互相独立。
smoke 为 2 步、512 completion tokens、梯度累积 1，仅验证双卡训练和参数更新。
原配方的正式 checkpoint 每 50 步保存，最多保留 20 个。

### 存储预算

实测双卡 smoke 的完整可续训 checkpoint 约 575 MiB：LoRA safetensors 约 70.5 MiB，其余主要是 DeepSpeed 优化器/参数状态及 tokenizer。
850 步按每 50 步保存，共 17 个正式 checkpoint，预计约 9.6 GiB，另有最终 adapter 输出约 82 MiB。
保留上限20高于这次运行的17个，因此第450、500、850步均保留。原始基础模型只存一份。
初轮评测直接加载LoRA；2026-10-17的官方路径复核额外保存了官方500步和本地500步两份BF16合并模型，共约6.62GiB。复核开始后的数据盘余量约20GB，所有原checkpoint保留。

## 本地改动

- `tina/post_train_hf/grpo.py`：支持本地数据、独立运行名称、指定步数停止；检查只有 LoRA 参数可训练；训练结束保存本地 adapter；仅在明确配置 push_to_hub 时上传。
- 通过环境变量关闭固定题目的额外 callback；每步 GRPO 训练指标仍记录在本地离线 W&B 和 trainer_state.json。
- 不执行原 set_vars.sh 中的全局 git 凭据修改和登录操作。
- 不执行原 set_env.sh 中对 base conda 环境的更新和全局缓存清理。
- 长序列预检在原 vLLM 40% 显存预算下反向传播 OOM；本地将 vLLM 预算改为 25%，并设置 `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`。训练 batch、生成数、序列长度与损失不变。
- `TINA_RESOURCE_LOG=1` 时，各训练进程将前 3 步及随后每 10 步的耗时、当前显存和峰值显存写到训练输出目录的 `resources_rank*.jsonl`，用于核验双卡资源占用。

## 评测

2026-10-17复核发现旧简化判分器与官方不等价，详见 `../RECHECK.md`。
`evaluate.py` 现默认直接使用仓库的官方LightEval任务判分器，须使用独立评测环境；`--scorer legacy-math-verify` 仅用于追溯旧流程。
原 `pipeline.py` 明确保留legacy判分以避免混用历史工件；新增复核使用 `recheck_official_eval.py`，完整执行合并权重+双卡LightEval。

```bash
source reproduction/env.sh
python -m venv --system-site-packages artifacts/recheck/eval_env
artifacts/recheck/eval_env/bin/python -m pip install 'lighteval==0.8.1' more-itertools langdetect
artifacts/recheck/eval_env/bin/python -m unittest reproduction.test_scoring
# 将 --model 指向已合并的模型目录；单任务方式对应论文命令
artifacts/recheck/eval_env/bin/python reproduction/recheck_official_eval.py \
  --model artifacts/recheck/merged_local500 --tasks math_500 \
  --output artifacts/recheck/official_pipeline_local500/math_500
```

`evaluate.py` 用官方 MATH_QUERY_TEMPLATE 和基础模型原始 chat template，temperature=0.6、top_p=0.95、seed=42、默认最多 32768 新 tokens，对固定 MATH-500 测试集做训练前后比较。

```bash
source reproduction/env.sh
CUDA_VISIBLE_DEVICES=0 artifacts/recheck/eval_env/bin/python reproduction/evaluate.py --output artifacts/recheck/base_dynamic_official_score.jsonl
CUDA_VISIBLE_DEVICES=0 artifacts/recheck/eval_env/bin/python reproduction/evaluate.py \
  --adapter artifacts/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/grpo_open_rs3/checkpoint-450 \
  --output artifacts/recheck/checkpoint450_dynamic_official_score.jsonl
```

保存每题输出、正确性、token 数、截断情况和 summary JSON。
`evaluate.py --adapter` 使用vLLM动态LoRA；`--model` 支持已有BF16合并模型，供固定旧协议的单变量对照。即使使用官方判分器，其单卡seed42协议仍属于本地条件对照。完整官方路径使用 `recheck_official_eval.py`。差异归因与第三方复现检索见 `../ATTRIBUTION.md`。
评测时应停止训练以避免显存争用。查看日志：`tail -f logs/main.log`。

自动流程由 `pipeline.py` 执行：smoke、adapter audit、完整 MATH-500 基线、正式训练、adapter audit、checkpoint 450/850 的 MATH-500 评测。
进度见 `artifacts/pipeline_status.json`；每阶段日志在 `logs/`。`screen` 会话 `tina-repro` 负责在控制进程意外退出后恢复流程，报告了错误的阶段不会自动无限重试。
恢复命令：`source reproduction/env.sh && python -u reproduction/pipeline.py`。文件锁会阻止重复启动。

数据审计发现官方 open-rs 训练集与 MATH-500 有 6 道完全相同的问题，见 `artifacts/data_overlap_audit.json`。保留训练配方，最终报告同时提供 500 题和排除这 6 题后的结果。
全部训练与评测完成后运行 `python reproduction/write_report.py`，核验工件后生成 `REPORT.md` 和 `artifacts/comparison.json`。该脚本会拒绝未完成的 850 步训练或不足 500 条的评测结果。

网络默认继承用户提供的代理。大模型权重可用 `parallel_download.py` 分块下载并检查完整 SHA-256；只有验证成功才写入正式权重路径。
