"""Build the final report only after all required training/evaluation artifacts exist."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
art = root / "artifacts"
model = art / "ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/grpo_open_rs3"
state = json.loads((model / "trainer_state.json").read_text())
assert state["global_step"] == 850, "Formal training has not reached step 850"
audit = json.loads((model / "adapter_audit.json").read_text())
assert audit["rank"] == 32 and audit["nonzero_lora_b"] > 0
overlap = json.loads((art / "data_overlap_audit.json").read_text())
exclude = set(overlap["normalized_overlap_ids"])
evaluations = {}
for name in ["base", "checkpoint450", "checkpoint850"]:
    rows = [json.loads(line) for line in (art / f"eval/{name}.jsonl").read_text().splitlines()]
    summary = json.loads((art / f"eval/{name}.summary.json").read_text())
    assert len(rows) == summary["samples"] == 500
    assert sum(row["correct"] for row in rows) == summary["correct"]
    assert summary["seed"] == 42 and summary["max_tokens"] == 32768
    assert summary["temperature"] == 0.6 and summary["top_p"] == 0.95
    assert len({row["unique_id"] for row in rows}) == 500
    evaluations[name] = {"rows": rows, "summary": summary}
base = evaluations["base"]["rows"]
results = {}
for name, data in evaluations.items():
    rows = data["rows"]
    assert [(r["unique_id"], r["problem"]) for r in rows] == [(r["unique_id"], r["problem"]) for r in base]
    clean = [r for r in rows if r["unique_id"] not in exclude]
    wins = sum(not b["correct"] and r["correct"] for b, r in zip(base, rows))
    losses = sum(b["correct"] and not r["correct"] for b, r in zip(base, rows))
    results[name] = dict(data["summary"], nonoverlap_samples=len(clean),
                         nonoverlap_correct=sum(r["correct"] for r in clean),
                         nonoverlap_accuracy=sum(r["correct"] for r in clean) / len(clean),
                         gains_over_base=wins, regressions_from_base=losses)
    results[name]["scorer"] = data["summary"].get("scorer", "legacy-math-verify")
resources = {}
for path in model.glob("resources_rank*.jsonl"):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    resources[path.stem] = {"peak_allocated_gib": max(r["peak_allocated_gib"] for r in rows),
                            "peak_reserved_gib": max(r["peak_reserved_gib"] for r in rows),
                            "last_recorded_step": rows[-1]["step"]}
hashes = {}
assert {"resources_rank0", "resources_rank1"} <= set(resources), "Both training ranks must have resource records"
for step in [450, 850]:
    path = model / f"checkpoint-{step}/adapter_model.safetensors"
    assert path.exists()
    hashes[str(step)] = hashlib.sha256(path.read_bytes()).hexdigest()
assert hashes["450"] != hashes["850"], "Checkpoints unexpectedly identical"
report = {"training_global_step": state["global_step"], "adapter_audit": audit,
          "checkpoint_sha256": hashes, "evaluations": results,
          "resource_peaks": resources, "data_overlap": overlap,
          "manifest": json.loads((art / "download_manifest.json").read_text())}
report["evaluation_scope"] = "Historical dynamic-LoRA evaluation; see scorer per evaluation and RECHECK.md."
recheck_path = art / "recheck/results_summary.json"
if recheck_path.exists():
    report["official_recheck"] = json.loads(recheck_path.read_text())
multiseed_path = art / "attribution/multiseed_results.json"
if multiseed_path.exists():
    report["official_multiseed"] = json.loads(multiseed_path.read_text())
endpoint_path = art / "endpoint850/results.json"
if endpoint_path.exists():
    report["official_endpoint850"] = json.loads(endpoint_path.read_text())
core_path = art / "core_suite/results.json"
if core_path.exists():
    report["core_experiments"] = json.loads(core_path.read_text())
(art / "comparison.json").write_text(json.dumps(report, indent=2))
lines = ["# TINA 核心复现实验报告", "",
         "与作者的逐项差异、单变量控制及第三方复现证据见 [ATTRIBUTION.md](ATTRIBUTION.md)。", "",
         "已完成 Open-RS3 的 850 步 LoRA-GRPO 训练及训练前、checkpoint 450/850 的同条件 MATH-500 评测。", "",
         f"初轮历史简化判分：基线 {results['base']['accuracy']:.1%}，第 450 步 {results['checkpoint450']['accuracy']:.1%}，第 850 步 {results['checkpoint850']['accuracy']:.1%}。",
         "训练与评测流程完成，不代表性能提升已复现；应以以下实际分数和限制判断结果。", "",
         "| 模型 | MATH-500 | 排除重复题后 | 截断数 | 平均生成 tokens | 相对基线改对/改错 |",
         "|---|---:|---:|---:|---:|---:|"]
for name, s in results.items():
    lines.append(f"| {name} | {s['correct']}/500 ({s['accuracy']:.2%}) | "
                 f"{s['nonoverlap_correct']}/{s['nonoverlap_samples']} ({s['nonoverlap_accuracy']:.2%}) | "
                 f"{s['truncated']} | {s['mean_tokens']:.1f} | {s['gains_over_base']}/{s['regressions_from_base']} |")
delta450 = (results["checkpoint450"]["accuracy"] - results["base"]["accuracy"]) * 100
delta850 = (results["checkpoint850"]["accuracy"] - results["base"]["accuracy"]) * 100
lines += ["", f"第 450、850 步相对基线分别变化 {delta450:+.1f}、{delta850:+.1f} 个百分点。",
          "在初轮配置、采样 seed 和历史判分下，没有观察到 MATH-500 准确率提升。排除 6 道重复题后方向相同。",
          "这不是严格对齐官方 LightEval 的全基准复现，也不能仅凭一个采样 seed 断言方法普遍无效。若继续验证性能结论，优先对齐官方合并权重与 LightEval 流程，再重复采样并补测 AIME24/25。", ""]
lines += ["", "## 证据与限制", "",
          f"- 训练完成步数：{state['global_step']}；adapter 参数量：{audit['adapter_parameters']:,}；LoRA B 非零元素：{audit['nonzero_lora_b']:,}。",
          f"- 原官方训练集含 {len(exclude)} 道与 MATH-500 完全相同的问题。保留原训练集以复现配方，同时报告排除这些题后的结果；这不排除其他改写重复或基础模型预训练暴露。",
          "- 评测保留官方数学提示词、基础模型 chat template、temperature=0.6、top_p=0.95、seed=42，最多生成 32768 tokens。",
          "- 本地评测使用 vLLM LoRA 加载和 math-verify；官方使用合并权重与 LightEval。上下文上限本地设为 33792 以容纳提示词加 32768 新 tokens，官方脚本为 32768。因此这是本地同条件对照，不能声称严格复现官方六项 benchmark 分数。",
          "- 只评估一个采样 seed；训练奖励上升或单次分数差异不能单独证明稳定推理能力提升。",
          "- 当前官方代码与早期论文超参数表存在差异，本次遵循锁定提交的代码配置。",
          "- 本地依赖将已不可获取的 xgrammar 0.1.16 替换为兼容的 0.1.17。",
          "- 原 vLLM 40% 显存预算触发长序列反向传播 OOM，改为 25% 并启用 expandable_segments；训练长度、batch、采样数、奖励及损失保持原实现。", "",
          "每题输出在 `artifacts/eval/*.jsonl`，完整统计和 checkpoint 校验和在 `artifacts/comparison.json`；版本见 `artifacts/download_manifest.json`、`reproduction/upstream_commit.txt` 和 `reproduction/installed.freeze.txt`。", "",
          "## 采样到的训练显存峰值", "", "```json", json.dumps(resources, indent=2), "```", ""]
training_audit = art / "training_audit.json"
if training_audit.exists():
    train = json.loads(training_audit.read_text())
    lines += ["## 训练审计", "",
              f"850 个训练步的指标均有限，梯度范数均大于零；训练耗时 {train['training_seconds'] / 3600:.2f} 小时。",
              f"17 个 checkpoint 共 {train['checkpoint_total_bytes'] / 2**30:.2f} GiB，磁盘原始基础权重 SHA-256 与训练前一致；这不证明训练中内存权重位级不变。", ""]
if (art / "metrics/training_curves.png").exists():
    lines += ["## 训练曲线", "", "奖励曲线属于训练分布，不等于测试准确率。", "",
              "![训练曲线](artifacts/metrics/training_curves.png)", "",
              "![优化指标](artifacts/metrics/optimization_curves.png)", "",
              "原始逐步数据：[training.csv](artifacts/metrics/training.csv)。", ""]
if (root / "RECHECK.md").exists():
    correction = ["> 2026-10-17复核：已确认初轮简化判分器与官方LightEval不等价，且漏测Open-RS3最佳checkpoint500。最新结论与官方权重对照见 [RECHECK.md](RECHECK.md)。下述初轮结果保留供追溯。", "",
                  "## 原输出的官方重判", "", "| 输出 | 官方正确数 | 官方准确率 |", "|---|---:|---:|"]
    for name in results:
        path = art / f"recheck/{name}.regrade.json"
        if path.exists():
            value = json.loads(path.read_text())
            correction.append(f"| {name} | {value['lighteval_correct']}/{value['samples']} | {value['lighteval_correct']/value['samples']:.1%} |")
    correction += ["", "## 初轮历史记录", ""]
    if report.get("official_recheck", {}).get("status") == "complete":
        latest = report["official_recheck"]["official_pipeline"]
        overview = ["## 三项基准的单次官方路径复核（9项已完成）", "",
                    "已使用BF16合并权重、双卡LightEval及官方任务，对基座、作者500步、本地500步执行三项基准。1710题逐题核验通过，详细证据见 [RECHECK.md](RECHECK.md)。", "",
                    "| 模型 | MATH-500 | AIME24 | AMC23 |", "|---|---:|---:|---:|"]
        for name in ["base", "official500", "local500"]:
            values = [f"{latest[name][task]['accuracy']:.2%}" for task in ["math_500", "aime24", "amc23"]]
            overview.append(f"| {name} | " + " | ".join(values) + " |")
        overview += ["", "本地500步在本次AIME24和AMC23采样中高于基座，MATH-500少1题；尚未复现作者权重的MATH-500增益。单次采样不证明跨seed稳定收益，也不等于论文全部基准复现。", ""]
        correction = overview + correction
    lines[2:2] = correction
if report.get("official_multiseed", {}).get("status") == "complete":
    multi = report["official_multiseed"]
    overview = ["## 500步：三 seed MATH-500 复核完成", "",
                "新增六项评测已结束，连同原seed1234共9项、4500条输出逐题核验通过。", "",
                "| 模型 | seed1234 | seed42 | seed2026 | 均值 |", "|---|---:|---:|---:|---:|"]
    for name in ["base", "official500", "local500"]:
        values = [f"{multi['records'][str(seed)][name]['accuracy']:.2%}" for seed in multi["seeds"]]
        overview.append(f"| {name} | " + " | ".join(values) + f" | {multi['statistics'][name]['mean_accuracy']:.2%} |")
    overview += ["", "本地对作者的差距为−2.4/0/−0.2个百分点，原单次12题差距未稳定重现；平均低0.87个百分点。本地相对基座平均低0.27个百分点，尚未证明稳定提升或确定唯一训练根因。这里只重复了推理seed，仍只有一个训练seed。", "",
                 "完整归因：[ATTRIBUTION.md](ATTRIBUTION.md)；逐题核验：[multiseed_final_audit.json](artifacts/attribution/multiseed_final_audit.json)。", ""]
    lines[2:2] = overview
if report.get("official_endpoint850", {}).get("status") == "complete":
    endpoint = report["official_endpoint850"]["results"]
    overview = ["## 历史阶段：850步三项官方协议评测", "",
        "原850步独立训练adapter未重训，在BF16合并、双卡LightEval、引擎seed1234协议下完成三项评测，570条逐题核验通过。", "",
        "| 任务 | 同协议基座 | 本地850步 | 论文850步 |", "|---|---:|---:|---:|"]
    for task, row in endpoint.items():
        overview.append(f"| {task} | {row['base_same_protocol_accuracy']:.2%} | {row['accuracy']:.2%} | {row['paper850_accuracy']:.2%} |")
    overview += ["", "本次三项均高于同协议基座和论文850步分数，支持保留现有训练并继续验证。此前旧协议下的下降不能作为推倒重训的依据；本轮仍是单次采样、三项基准，不能声称跨seed稳定提升或六项论文完整复现。", "",
        "结果：[ENDPOINT850.md](ENDPOINT850.md)；逐题与配对审计：[final_audit.json](artifacts/endpoint850/final_audit.json)。", ""]
    lines[2:2] = overview
if report.get("core_experiments", {}).get("status") == "complete":
    core = report["core_experiments"]
    overview = ["## 最新：六项评测和 Open-RS2 独立训练全部完成（2026-10-18）", "",
        "五个模型×六项基准，共5350条输出逐题核验通过。Open-RS3独立训练850步；新增Open-RS2独立训练450步、耗时约13.96小时，450步指标和最终adapter权重核验通过。", "",
        "| 模型 | 六项均值 | 相对本地基座（百分点） |", "|---|---:|---:|"]
    for name, data in core["models"].items():
        overview.append(f"| {name} | {data['six_benchmark_mean']:.2%} | {data['delta_to_local_base_pp']:+.2f} |")
    overview += ["", "Open-RS3 500步均值49.46%接近论文49.45%，但本地基座45.55%也高于论文41.18%；本地增益+3.91个百分点小于论文约+8.27个百分点。Open-RS2 450步为47.53%，低于论文50.60%约3.07个百分点。作者500步权重本地评测为46.57%。逐项分数并未完全匹配，不能据此声称稳定增益或精确复现论文。", "",
        "本轮每任务单次推理seed1234、训练seed42。GPQA公开数据与原文件字节一致，提示词创建前固定seed1234以统一选项顺序；作者原始选项排列不可核验。此前三seed MATH-500结果仍保留在下文。", "",
        "完整六项表格与论文对照：[CORE_EXPERIMENTS.md](CORE_EXPERIMENTS.md)；[结果数据](artifacts/core_suite/results.json)；[最终审计](artifacts/core_suite/final_audit.json)。下文各节为历史阶段记录。", ""]
    lines[2:2] = overview
if (art / "mechanism/plan.json").exists():
    overview = ["## 当前推进：核心机制消融", "",
        "此前核心训练与六项评测已经完成。当前新增两组独立850步训练及固定500/850步评测，检验作者format-only配方与去除答案依赖的对照。具体进度以 [MECHANISM.md](MECHANISM.md) 和 [状态文件](artifacts/mechanism/status.json) 为准。", "",
        "已确认公开format-only配方中的length奖励仍核对答案正误，因此不能把它解释成严格移除了正确性监督。固定文本、仅改变参考答案即可改变其奖励，见 [反事实奖励核验](artifacts/mechanism/reward_semantics_audit.json)。两组对照能否支持论文的机制解释，以实际评测结果为准。", ""]
    stage1_path = art / "mechanism/stage1_audit.json"
    if stage1_path.exists():
        stage1 = json.loads(stage1_path.read_text())
        overview += ["第一组训练和12项评测已完成：作者格式配方500步六项均值49.84%，850步46.30%，基座45.55%；2140条输出核验通过。500→850下降3.55个百分点。格式奖励和长度没有出现原RS3同样明显的后期转折，但公开配方仍保留正确性监督，不能据此证明纯格式机制。详情：[MECHANISM_STAGE1.md](MECHANISM_STAGE1.md)。第二组的完成情况以状态文件和机制结果表为准。", ""]
    if (art / "mechanism/completion_reaudit.json").exists():
        mechanism = json.loads((art / "mechanism/results.json").read_text())
        overview = ["## 最新：核心机制消融全部完成（2026-10-21复核）", "",
            "两组各850步独立训练，24项评测、4280条输出已重新核验。题目、提示词、参考答案与顺序一致；原始分数与汇总匹配；四份关键adapter权重有限且校验和一致。", "",
            "| 固定检查点 | 作者格式配方 | 去除答案依赖 |", "|---|---:|---:|"]
        for step in (500,850):
            a = mechanism["models"][f"official_format_{step}"]["six_benchmark_mean"]
            b = mechanism["models"][f"label_free_format_{step}"]["six_benchmark_mean"]
            overview.append(f"| {step}步 | {a:.2%} | {b:.2%} |")
        overview += ["", "基座六项均值45.55%。方法主线得到本次结果支持：LoRA+GRPO训练的RS3 500步49.46%、RS2 450步47.53%及作者格式配方500步49.84%，均有本轮平均分收益。", "",
            "机制解释尚未完整验证：公开format-only配方仍含正确性监督，去除该依赖后500/850步均值为46.09%/44.35%，相对基座+0.54/−1.20个百分点。不能把作者格式配方成绩当作纯格式学习充分性的证据，也没有通过知识保持测试证明只学格式、不改知识。结果仍为单训练seed、单推理seed，不能声称跨seed稳定提升或全部论文结论成立。", "",
            "完整结果与解释：[MECHANISM.md](MECHANISM.md)；[最终审计](artifacts/mechanism/final_audit.json)；[完成后原始输出与权重复核](artifacts/mechanism/completion_reaudit.json)。下文为此前各阶段记录。", ""]
    lines[2:2] = overview
(root / "REPORT.md").write_text("\n".join(lines))
print(json.dumps({"training_step": state["global_step"], "report": str(root / 'REPORT.md'), "results": results}, indent=2))
