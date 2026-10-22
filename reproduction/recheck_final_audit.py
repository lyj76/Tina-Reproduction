"""Verify all saved official-path predictions and export readable JSONL files."""
import ast
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

from datasets import load_from_disk
import pyarrow.parquet as pq
from scipy.stats import binomtest


root = Path("artifacts/recheck")
models = ["base", "official500", "local500"]
counts = {"math_500": 500, "aime24": 30, "amc23": 40}
assert json.loads((root / "queue_all.json").read_text())["status"] == "complete"
overlap_ids = set(json.loads(Path("artifacts/data_overlap_audit.json").read_text())["normalized_overlap_ids"])
math_ids = {x["problem"]: x["unique_id"] for x in load_from_disk("artifacts/data/math500")}
all_rows, audits = {}, {}


def value(item):
    return ast.literal_eval(item) if isinstance(item, str) else item


for model in models:
    all_rows[model], audits[model] = {}, {}
    for task, expected in counts.items():
        folder = root / f"official_pipeline_{model}" / task
        files = list((folder / "details").glob("**/*.parquet"))
        assert len(files) == 1
        rows = pq.read_table(files[0], columns=["example", "full_prompt", "gold", "predictions", "metrics",
                                               "specifics", "num_asked_few_shots", "num_effective_few_shots"]).to_pylist()
        assert len(rows) == expected
        assert len({x["full_prompt"] for x in rows}) == expected
        output = []
        for index, row in enumerate(rows):
            assert row["num_asked_few_shots"] == row["num_effective_few_shots"] == 0
            predictions = value(row["predictions"])
            metrics = value(row["metrics"])
            assert len(predictions) == 1 and predictions[0]
            assert metrics["extractive_match"] in (0.0, 1.0)
            problem = row["example"].split("\n\n", 1)[1]
            output.append({"index": index, "problem": problem, "gold": value(row["gold"]),
                           "prediction": predictions[0], "correct": bool(metrics["extractive_match"]),
                           "extracted": value(row["specifics"]), "full_prompt": row["full_prompt"]})
            if task == "math_500":
                output[-1]["unique_id"] = math_ids[problem]
        summary = json.loads((folder / "summary.json").read_text())
        task_key = next(k for k in summary["results"] if task in k)
        correct = sum(x["correct"] for x in output)
        assert abs(correct / expected - summary["results"][task_key]["extractive_match"]) < 1e-12
        (folder / "predictions.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in output))
        audits[model][task] = {"samples": expected, "correct": correct, "zero_shot_verified": True}
        if task == "math_500":
            clean = [x for x in output if x["unique_id"] not in overlap_ids]
            assert len(clean) == 494
            audits[model][task]["nonoverlap"] = {"samples": 494, "correct": sum(x["correct"] for x in clean),
                                                  "accuracy": sum(x["correct"] for x in clean) / 494}
        all_rows[model][task] = output

paired = {}
for model in ["official500", "local500"]:
    paired[model] = {}
    for task in counts:
        base, rows = all_rows["base"][task], all_rows[model][task]
        assert [(x["full_prompt"], x["gold"]) for x in base] == [(x["full_prompt"], x["gold"]) for x in rows]
        gains = sum(not a["correct"] and b["correct"] for a, b in zip(base, rows))
        losses = sum(a["correct"] and not b["correct"] for a, b in zip(base, rows))
        paired[model][task] = {"gains": gains, "losses": losses,
                               "mcnemar_exact_p": binomtest(gains, gains+losses).pvalue if gains+losses else 1.0}

gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu", "--format=csv,noheader,nounits"], text=True)
report = {"completed_utc": datetime.now(timezone.utc).isoformat(), "status": "passed",
          "evaluations_verified": 9, "prediction_rows_verified": sum(counts.values()) * len(models),
          "same_questions_prompts_golds_and_order_verified": True, "counts": audits,
          "paired_vs_base": paired, "gpu_snapshot": gpu.strip(),
          "checkpoint_count": len(list(Path("artifacts/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/grpo_open_rs3").glob("checkpoint-*"))),
          "scope": "Artifact, input parity, and score-count audit; single sampling run, not proof of robust multi-seed improvements."}
assert report["checkpoint_count"] == 17
(root / "final_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
print(json.dumps(report, indent=2, ensure_ascii=False))
