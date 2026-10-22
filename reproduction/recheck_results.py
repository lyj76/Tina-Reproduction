"""Collect the predeclared control experiments without selecting a lucky checkpoint."""
import json
from pathlib import Path


root = Path("artifacts/recheck")
counts = {"math_500": 500, "aime24": 30, "amc23": 40}
models = ["base", "official500", "local500"]
results, pending = {}, []
for model in models:
    results[model] = {}
    for task, count in counts.items():
        path = root / f"official_pipeline_{model}" / task / "summary.json"
        if not path.exists():
            pending.append(f"{model}/{task}")
            continue
        result = json.loads(path.read_text())
        task_key = next(key for key in result["results"] if task in key and key != "all")
        metrics = result["results"][task_key]
        score = metrics["extractive_match"]
        assert abs(score * count - round(score * count)) < 1e-5
        assert result["manifest"]["data_parallel_size"] == 2
        assert result["manifest"]["max_samples"] is None
        assert result["manifest"]["requests"] == count
        results[model][task] = {"accuracy": score, "correct": round(score * count), "samples": count,
                                "seconds": result["seconds"], "source": str(path)}

regraded = {}
for path in root.glob("*.regrade.json"):
    value = json.loads(path.read_text())
    regraded[path.stem.removesuffix(".regrade")] = {k: v for k, v in value.items() if k != "rows"}
summary = {"status": "complete" if not pending else "in_progress", "pending": pending,
           "official_pipeline": results, "saved_outputs_regraded": regraded,
           "caveat": "Single sampled run, predeclared checkpoint500; three benchmarks, not the paper's entire six-benchmark suite."}
if not pending:
    summary["input_parity"] = {}
    for task in counts:
        manifests = [json.loads((root / f"official_pipeline_{model}" / task / "manifest.json").read_text()) for model in models]
        assert len({x["input_set_sha256"] for x in manifests}) == 1
        assert len({x["tasks_source_sha256"] for x in manifests}) == 1
        summary["input_parity"][task] = {"same_questions_and_prompts": True,
                                        "same_order": len({x["input_order_sha256"] for x in manifests}) == 1}
(root / "results_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)

if not pending:
    lines = ["## 完整官方路径对照结果", "",
             "以下均为合并权重、官方任务判分、双卡LightEval的单次采样结果。checkpoint500在看到新评测结果前已预先指定。", "",
             "| 模型 | MATH-500 | AIME24 | AMC23 |", "|---|---:|---:|---:|"]
    for model in models:
        values = [f"{results[model][task]['correct']}/{counts[task]} ({100*results[model][task]['accuracy']:.2f}%)" for task in counts]
        lines.append(f"| {model} | " + " | ".join(values) + " |")
    lines += ["", "| 对照 | MATH-500差值 | AIME24差值 | AMC23差值 |", "|---|---:|---:|---:|"]
    for a, b in [("local500", "base"), ("official500", "base"), ("local500", "official500")]:
        values = [f"{100*(results[a][task]['accuracy']-results[b][task]['accuracy']):+.2f} pp" for task in counts]
        lines.append(f"| {a} − {b} | " + " | ".join(values) + " |")
    lines += ["", "原始逐题输出和抽取答案保存在 `artifacts/recheck/official_pipeline_*/<task>/details/`；配置、指标和时间见各目录 `manifest.json`、`summary.json`。", "",
              "这些对照同时纠正了检查点选择和评测路径；不能把路径切换产生的全部变化归因于某一个推理参数。单次结果也不能证明跨seed稳定收益。"]
    path = Path("RECHECK.md")
    content = path.read_text().split("\n<!-- OFFICIAL_RESULTS -->")[0]
    content = content.replace("# TINA 复核记录（运行中）", "# TINA 复核记录（三项官方路径对照完成）")
    content += "\n<!-- OFFICIAL_RESULTS -->\n" + "\n".join(lines) + "\n"
    path.write_text(content)
