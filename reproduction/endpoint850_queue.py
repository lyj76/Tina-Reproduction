"""Evaluate the existing850-step adapter with the same official protocol as500."""
import ast
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time

import pyarrow.parquet as pq


ROOT = Path("artifacts/endpoint850")
ROOT.mkdir(exist_ok=True)
PYTHON = str(Path("artifacts/recheck/eval_env").resolve() / "bin/python")
MODEL = ROOT / "merged_local850"
TASKS = {"math_500": {"samples": 500, "paper": .838},
         "aime24": {"samples": 30, "paper": 8/30},
         "amc23": {"samples": 40, "paper": .75}}


def status(state, **extra):
    record = {"status": state, "pid": os.getpid(), "updated_utc": datetime.now(timezone.utc).isoformat(), **extra}
    temp = ROOT / "status.tmp"
    temp.write_text(json.dumps(record, indent=2)); temp.replace(ROOT / "status.json")
    print(json.dumps(record), flush=True)


def decode(value):
    return ast.literal_eval(value) if isinstance(value, str) else value


def aggregate():
    audits, results = {}, {}
    for task, spec in TASKS.items():
        folder = ROOT / task
        files = list((folder / "details").glob("**/*.parquet"))
        assert len(files) == 1
        raw = pq.read_table(files[0]).to_pylist()
        assert len(raw) == spec["samples"] and len({row["full_prompt"] for row in raw}) == spec["samples"]
        base = [json.loads(line) for line in Path(f"artifacts/recheck/official_pipeline_base/{task}/predictions.jsonl").read_text().splitlines()]
        rows = []
        for index, (row, baseline) in enumerate(zip(raw, base)):
            assert row["num_asked_few_shots"] == row["num_effective_few_shots"] == 0
            gold, predictions, metrics = decode(row["gold"]), decode(row["predictions"]), decode(row["metrics"])
            assert row["full_prompt"] == baseline["full_prompt"] and gold == baseline["gold"]
            assert len(predictions) == 1 and predictions[0] and metrics["extractive_match"] in (0., 1.)
            rows.append({"index": index, "problem": baseline["problem"], "full_prompt": row["full_prompt"],
                "gold": gold, "prediction": predictions[0], "correct": bool(metrics["extractive_match"]),
                "extracted": decode(row["specifics"])})
        count = sum(row["correct"] for row in rows)
        summary = json.loads((folder / "summary.json").read_text())
        score = summary["results"][f"custom:{task}:0"]["extractive_match"]
        assert abs(count / len(rows) - score) < 1e-12
        m = summary["manifest"]
        assert m["engine_seed"] == 1234 and m["sampling_seed"] is None and m["data_parallel_size"] == 2
        (folder / "predictions.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        baseline_score = sum(row["correct"] for row in base) / len(base)
        results[task] = {"samples": len(rows), "correct": count, "accuracy": score,
            "paper850_accuracy": spec["paper"], "base_same_protocol_accuracy": baseline_score,
            "vs_paper_percentage_points": 100*(score-spec["paper"]),
            "vs_base_percentage_points": 100*(score-baseline_score)}
        audits[task] = {"input_parity_with_base": True, "zero_shot": True, "score_count_matches": True}
    result = {"status": "complete", "results": results, "audit": audits, "predictions_verified": 570,
        "scope": "Existing independent850-step adapter, one sampled run per task,3 of6 paper benchmarks. Not a retraining or a complete six-benchmark reproduction."}
    (ROOT / "results.json").write_text(json.dumps(result, indent=2))
    lines = ["# 现有850步模型的官方协议评测", "", "保留原850步adapter，在BF16合并、双卡LightEval0.8.1、官方任务、引擎seed1234/逐请求seedNone下评测。", "",
        "| 任务 | 论文850步 | 本地850步 | 同协议基座 |", "|---|---:|---:|---:|"]
    for task, row in results.items():
        lines.append(f"| {task} | {row['paper850_accuracy']:.2%} | {row['accuracy']:.2%} | {row['base_same_protocol_accuracy']:.2%} |")
    lines += ["", "570条输出逐题核验通过，题目/提示词/参考答案与此前基座完全一致。三项、单次采样，不能与论文六项均值直接比较。", "",
        "结果：[results.json](artifacts/endpoint850/results.json)。原850步输出官方重判为MATH-50083.8%，属于旧单卡动态LoRA协议，保留供追溯。"]
    Path("ENDPOINT850.md").write_text("\n".join(lines) + "\n")


def main():
    plan = {"declared_utc": datetime.now(timezone.utc).isoformat(), "checkpoint": 850, "tasks": TASKS,
        "engine_seed": 1234, "sampling_seed": None, "training_runs": 0,
        "purpose": "Assess the existing independent run before deciding whether any long retraining is justified."}
    if not (ROOT / "plan.json").exists():
        (ROOT / "plan.json").write_text(json.dumps(plan, indent=2))
    if not (MODEL / "merge_audit.json").exists():
        status("merging_on_cpu")
        with open("logs/endpoint850_merge.log", "w") as handle:
            subprocess.run([PYTHON, "reproduction/recheck_merge.py", "--adapter",
                "artifacts/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/grpo_open_rs3/checkpoint-850", "--output", str(MODEL)],
                stdout=handle, stderr=subprocess.STDOUT, check=True)
    status("waiting_for_trajectory_controls")
    deadline = time.monotonic() + 3600
    while True:
        state = json.loads(Path("artifacts/trajectory/status.json").read_text())["status"]
        if state == "failed":
            raise RuntimeError("Early-trajectory controls failed; inspect before proceeding")
        if state == "complete":
            used = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
            if all(int(row.strip()) < 1024 for row in used.splitlines()):
                break
        if time.monotonic() > deadline:
            raise TimeoutError("Waiting for training controls/GPU release timed out")
        time.sleep(15)
    for task in TASKS:
        out = ROOT / task
        if (out / "summary.json").exists():
            continue
        status("evaluating", task=task)
        with open(f"logs/endpoint850_{task}.log", "w") as handle:
            subprocess.run([PYTHON, "-u", "reproduction/recheck_official_eval.py", "--model", str(MODEL),
                "--tasks", task, "--output", str(out), "--seed", "1234"], stdout=handle, stderr=subprocess.STDOUT, check=True)
    aggregate()
    status("complete", results=str(ROOT / "results.json"))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        status("failed", error=repr(error))
        raise
