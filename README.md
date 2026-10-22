<div align="center">
  <h1>Tina: Tiny Reasoning Models via LoRA<br>（独立全流程复现与机制核验）</h1>

  <p>
    <b>探索极低成本下通过 LoRA + GRPO 激发小模型（1.5B）复杂推理能力的可行性与机制边界</b>
  </p>

  <div>
    <a href="https://arxiv.org/abs/2504.15777"><img src="https://img.shields.io/badge/Paper-arXiv%3A2504.15777-B31B1B.svg" alt="Paper"></a>
    <a href="https://github.com/shangshang-wang/Tina"><img src="https://img.shields.io/badge/Upstream-shangshang--wang%2FTina-blue.svg?logo=github" alt="Upstream"></a>
    <a href="https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"><img src="https://img.shields.io/badge/Base_Model-Qwen--1.5B-yellow.svg?logo=huggingface" alt="Base Model"></a>
    <img src="https://img.shields.io/badge/Hardware-2%C3%97_RTX_4090_(48GB)-76B900.svg?logo=nvidia&logoColor=white" alt="Hardware">
    <img src="https://img.shields.io/badge/Algorithm-LoRA_%2B_GRPO-green.svg" alt="Algorithm">
    <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache--2.0-lightgrey.svg" alt="License"></a>
  </div>

  <br>

  <p>
    <b>🌐 Language</b>: <b>🇨🇳 中文说明 (当前)</b> | <a href="README_en.md"><b>🇬🇧 English Upstream README</b></a>
  </p>
</div>

<!-- 顶栏核心实验大图 / Hero Figure -->
<div align="center">
  <img src="assets/reproduction_hero.png" alt="Tina Reproduction & Mechanistic Audit on 2x RTX 4090" width="100%" style="border-radius: 8px; box-shadow: 0 4px 16px rgba(0,0,0,0.08);">
  <p><sub><b>图 1：双卡 RTX 4090 平台 Tina 全流程复现主图</b> —— (A) 六大数学与推理基准评测对比；(B) 工业级集群 L40S 与本地双卡 4090 思维链长度扩展动态；(C) 消费级硬件与“纯格式奖励”机制核验要点。</sub></p>
</div>

---

## 📌 项目概述 (Overview)

本项目是对学术论文 [《Tina: Tiny Reasoning Models via LoRA》](https://arxiv.org/abs/2504.15777)（ICLR 2026）的**独立、端到端全流程复现与机制核验工程**。

原论文提出了一个核心问题：**“如何以高性价比的强化学习方案，在小参数量模型中高效注入强推理能力？”** 官方通过在 `DeepSeek-R1-Distill-Qwen-1.5B` 上使用 LoRA 进行 GRPO（Group Relative Policy Optimization）训练，声称仅耗费极少算力即可匹敌全参数强化学习的效果，并提出了一种无需参考答案标签的“纯格式奖励（Format-only）”机制。

<div align="center">
  <img src="assets/overall_comparison.png" alt="Official Paper Figure 1" width="95%">
  <p><sub><i>图 2：原论文 Figure 1 —— Tina 官方声称以超低算力成本（14小时 / $14）实现比肩全参数模型（DeepScaleR / STILL-3）的推理表现</i></sub></p>
</div>

为了验证其可重复性与底层机理，本项目在 **双卡消费级显卡 NVIDIA GeForce RTX 4090（24GB × 2 = 48GB）** 环境下，开展了从训练环境搭建、LoRA-GRPO 强化学习、全量基准对齐评测到机制消融的完整复现实验。

---

## 🏆 核心结论与主表对比 (Executive Summary)

本仓库所有模型均采用统一评测协议：**BF16 `merge_and_unload(safe_merge=True)` 合并权重、LightEval 0.8.1 评测框架、双卡数据并行、固定随机种子（评测 seed 1234）**，并在 6 项主流数学推理与逻辑评测基准（共 1,070 道无重叠题目/模型）上进行了严谨的逐题判分。

### 1. 六大综合基准对比总表

| 模型版本 | 训练步数 / 奖励机制 | 六项未加权均值 | 相对基座增益 | 核心基准亮点 (MATH-500 / AIME24 / AMC23) |
| :--- | :--- | :---: | :---: | :--- |
| **Qwen-1.5B 基座** | - | **45.55%** | `0.00` | 83.4% / 33.3% / 75.0% |
| **本地复现 RS3** | **500 步**（Cosine 奖励） | **49.46%** | **+3.91 pp** | 83.2% / **43.3%** / **80.0%** |
| **本地复现 RS3** | **850 步**（长程过拟合对照） | **47.69%** | **+2.14 pp** | **85.4%** / 40.0% / 77.5% |
| **本地复现 RS2** | **450 步**（Accuracy 奖励） | **47.53%** | **+1.98 pp** | **86.0%** / 40.0% / 72.5% |
| **作者官方权重** | 500 步（本机同一协议重测）| **46.57%** | **+1.02 pp** | 85.6% / 33.3% / 70.0% |
| **官方格式配方** | 500 步（本地复现） | **49.84%** | **+4.29 pp** | 85.2% / **43.3%** / 72.5% |
| **官方格式配方** | 850 步（本地复现） | **46.30%** | **+0.74 pp** | 85.8% / 33.3% / 67.5% |
| 🧪 **去答案依赖对照** | 500 步（真正无标签格式） | **46.09%** | **+0.54 pp** | 84.0% / 30.0% / 77.5% |
| 🧪 **去答案依赖对照** | 850 步（真正无标签格式） | **44.35%** | **-1.20 pp** | 85.4% / 26.7% / 70.0% |

*注：完整各任务正确题数与逐题判定详情见 [`migration/task_scores.csv`](migration/task_scores.csv) 和 [`migration/final_scores.csv`](migration/final_scores.csv)。*

---

## 🔬 关键科学发现与机制核验 (Key Findings)

通过本轮系统性的实验对比，我们提炼出以下三项关键结论：

### 1. LoRA + GRPO 提升小模型推理能力的有效性得到支持
* **小模型具备强可塑性**：仅训练 LoRA 参数（Rank 32, Alpha 128，训练参数量 < 1%），在 500 步内即可将六项综合基准提升接近 **4 个百分点**（45.55% → 49.46%），特别是在困难数学竞赛 AIME24 上准确率从 33.3% 跃升至 43.3%（提升超 20% 相对准确率）。
* 这强有力地证实了原作者核心主张：**LoRA 参数高效微调足以在 RL 阶段激发语言模型的长链思考（CoT）推理潜力**。

### 2. 长程训练收益递减与退化风险
* 无论是 RS3 还是格式对照组，当训练推进到 850 步时，综合均分普遍较 500 步阶段出现 **1.5 ~ 3.5 个百分点的回落**（如 RS3 从 49.46% 下滑至 47.69%）。
* 小模型在固定数据集上持续进行长步数强化学习时，容易出现过拟合、策略坍缩或泛化能力受损，**500 步左右是极佳的性能-算力平衡检查点**。

### 3. 核心机制揭秘：“纯格式奖励”并非真正无答案监督
* **原配方存在的隐式依赖**：原作者在论文中声称无需任何正确性标签即可仅靠“格式奖励”诱发推理能力。我们在代码审计中发现，官方公布的格式配方中 `length` 奖励实现**仍然读取并比对了真实参考答案**。
* **严格的反事实消融实验**：我们设计了无答案依赖对照组——保持全部 5 项格式奖励完全一致，仅将 `length` 奖励修改为纯粹基于文本长度的归一化（真正去除任何真实答案交互）。
* **实验结果表明**：一旦切断答案依赖，模型均分显著下降（500 步为 46.09%，850 步暴跌至 44.35%，低于基座基线）。**因此，不能将官方配方的优秀成绩直接解释为“纯格式学习充分性”的证据，底层仍存在隐式监督信号在起作用**。详细分析见 [`MECHANISM.md`](MECHANISM.md)。

---

## 📈 训练曲线与指标复查 (Training Dynamics)

项目完整记录了强化学习全过程的 Reward、Loss、KL 散度以及格式符合率演变：

<div align="center">
  <img src="assets/reproduction_training_curves.png" alt="Training Curves" width="90%">
  <p><i>图 1：本地 RS3 强化学习训练曲线（Reward 与 Completion 长度随步数平稳爬升）</i></p>
</div>

<div align="center">
  <img src="assets/reproduction_mechanism_comparison.png" alt="Mechanism Comparison" width="90%">
  <p><i>图 2：官方格式配方 vs 去除答案依赖配方的训练动态对比</i></p>
</div>

---

## 🛠️ 复现指南与避坑经验 (Practical Recipes)

在消费级硬件（双卡 NVIDIA GeForce RTX 4090 24GB × 2）环境下复现小模型强化学习时，原作者代码存在若干工程痛点，我们在本项目中提供了针对性的解决方案：

### 1. 消费级硬件与显存避坑：双卡 RTX 4090 调优
* **硬件配置**：2 × NVIDIA GeForce RTX 4090 (24GB × 2，共 48GB 显存)，消费级 GPU 平台。
* **痛点问题**：原仓库默认设置 `gpu_memory_utilization = 0.4`，在双卡 4090 上跑长文本推理（prompt 512 + completion 3584）反向传播阶段会频繁触发 PyTorch CUDA OOM。
* **调优方案**：在 [`reproduction/open_rs3.yaml`](reproduction/open_rs3.yaml) 中将 `gpu_memory_utilization` 精确调优至 `0.25`，成功在**消费级 4090 显卡**上完全保留 batch 6、梯度累积 4、每题生成 6 个候选项的全部原配方超参，实现零 OOM 稳定训练！

### 2. 评测严谨性：LightEval BF16 官方协议
* 初轮评测若使用简化的单题判分脚本，容易因正则表达式或题型映射差异出现系统误差（详见 [`RECHECK.md`](RECHECK.md)）。
* 本仓库在 [`reproduction/evaluate.py`](reproduction/evaluate.py) 中封装了基于 LightEval 0.8.1 的严格官方流水线，先安全合并 LoRA（BF16），再以统一 prompt 模版对全部 6 项基准执行 0-shot 生成与评估。

### 3. 快速启动训练

```bash
# 1. 激活训练环境
conda activate tina

# 2. 启动带防 OOM 配置与自动步数断点的 LoRA-GRPO 训练
bash reproduction/launch_core_train.sh
```

### 4. 快速复核基准评测

```bash
# 激活评估环境
conda activate tina_eval

# 执行 6 项标准基准评测（自动合并 LoRA 并调用 LightEval）
python reproduction/evaluate.py --model_path <MERGED_OR_BASE_PATH> --tasks math_500 aime24 aime25 amc23 gpqa_diamond minerva
```

---

## 📂 仓库结构导览 (Repository Structure)

```text
├── README.md               # 本中文复现报告与使用指南
├── README_en.md            # 作者原始英文 README
├── HANDOFF.md              # 实验交接入口与各阶段决策记录
├── REPORT.md               # 复现总报告与限制声明
├── CORE_EXPERIMENTS.md     # 6 大基准测试逐模型成绩汇总
├── MECHANISM.md            # 格式机制与去答案依赖消融深度报告
├── RECHECK.md              # 初轮评测偏差排查与修正记录
├── ATTRIBUTION.md          # 与作者原始报告的差异归因分析
│
├── migration/              # 归档数据与环境清单
│   ├── final_scores.csv    # 最终模型 6 项基准均分表
│   ├── task_scores.csv     # 54 组详细评测得分与正确题数
│   └── evaluation_registry.json # 9,630 条输出的溯源索引
│
├── reproduction/           # 复现专属实验工程
│   ├── open_rs3.yaml       # 修正后的 RS3 训练配置
│   ├── format_mechanism.yaml # 机制核验训练配置
│   └── evaluate.py         # 统一对齐的 LightEval 评测调度器
│
├── tina/                   # 核心算法与模型定义
│   ├── post_train_hf/      # GRPO Trainer、奖励函数与权重合并
│   └── utils/              # 提示词模版与 Chat Template
│
└── scripts/                # 原始训练与数据准备 Shell 脚本
```

---

## 📜 致谢与引用 (Citation & References)

感谢原作者团队开源的基础代码与工作成果：

```bibtex
@article{wang2025tina,
  title={Tina: Tiny Reasoning Models via LoRA},
  author={Wang, Shangshang and others},
  journal={arXiv preprint arXiv:2504.15777},
  year={2025}
}
```
