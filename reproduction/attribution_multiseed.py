"""Predeclared MATH-500 seed controls, using the unchanged official pipeline."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


ROOT = Path("artifacts/attribution")
PYTHON = str(Path("artifacts/recheck/eval_env").resolve() / "bin/python")
MODELS = {
    "base": "artifacts/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base",
    "official500": "artifacts/recheck/merged_official500",
    "local500": "artifacts/recheck/merged_local500",
}
SEEDS = [1234, 42, 2026]


def status(state, **extra):
    result = {"status": state, "pid": os.getpid(),
              "updated_utc": datetime.now(timezone.utc).isoformat(), **extra}
    temp = ROOT / "multiseed_status.tmp"
    temp.write_text(json.dumps(result, indent=2))
    temp.replace(ROOT / "multiseed_status.json")
    print(json.dumps(result), flush=True)


def output_dir(name, seed):
    if seed == 1234:
        return Path(f"artifacts/recheck/official_pipeline_{name}/math_500")
    return ROOT / "multiseed" / f"seed{seed}" / name


def aggregate():
    records, hashes = {}, set()
    for seed in SEEDS:
        records[str(seed)] = {}
        for name in MODELS:
            path = output_dir(name, seed)
            summary = json.loads((path / "summary.json").read_text())
            m = summary["manifest"]
            assert m["engine_seed"] == seed and m["sampling_seed"] is None
            assert m["requests"] == 500 and m["data_parallel_size"] == 2
            hashes.add(m["input_set_sha256"])
            score = summary["results"]["custom:math_500:0"]["extractive_match"]
            assert abs(score * 500 - round(score * 500)) < 1e-6
            records[str(seed)][name] = {"accuracy": score, "correct": round(score * 500),
                                        "source": str(path / "summary.json")}
    assert len(hashes) == 1, "Questions/prompts differ across controls"
    stats = {}
    for name in MODELS:
        values = [records[str(seed)][name]["accuracy"] for seed in SEEDS]
        stats[name] = {"mean_accuracy": statistics.mean(values),
                       "sample_sd": statistics.stdev(values), "min": min(values), "max": max(values)}
    deltas = {f"local500_minus_{other}": {
        "per_seed_percentage_points": {str(seed): 100 * (
            records[str(seed)]["local500"]["accuracy"] - records[str(seed)][other]["accuracy"])
            for seed in SEEDS},
        "mean_percentage_points": 100 * (stats["local500"]["mean_accuracy"] - stats[other]["mean_accuracy"])}
        for other in ("base", "official500")}
    result = {"status": "complete", "seeds": SEEDS, "records": records, "statistics": stats,
              "deltas": deltas, "input_parity": True,
              "caveat": "Three inference seeds, one training seed. This estimates inference variation only; it does not identify the training root cause or prove cross-training-seed generalization."}
    (ROOT / "multiseed_results.json").write_text(json.dumps(result, indent=2))
    lines = ["\n## 预先固定的多 seed MATH-500 复核\n",
             "保持官方 BF16 合并、双卡 LightEval 协议，固定引擎 seed 1234、42、2026；逐请求 seed 均为 None。1234 复用此前完整输出，新增六项。\n",
             "| 模型 | seed1234 | seed42 | seed2026 | 均值 ± 样本标准差 |",
             "|---|---:|---:|---:|---:|"]
    for name in MODELS:
        values = [f"{records[str(seed)][name]['accuracy']:.2%}" for seed in SEEDS]
        s = stats[name]
        lines.append(f"| {name} | {' | '.join(values)} | {100*s['mean_accuracy']:.2f}% ± {100*s['sample_sd']:.2f} pp |")
    lines += ["", "这是三个推理 seed、一个训练 seed。它能检查推理采样波动，不能单独证明训练差异的原因，也不能把三个 seed 的重复题当成 1500 道独立题。",
              "", "完整数据：[multiseed_results.json](artifacts/attribution/multiseed_results.json)。"]
    path = Path("ATTRIBUTION.md")
    content = path.read_text().split("\n## 预先固定的多 seed MATH-500 复核")[0]
    path.write_text(content + "\n" + "\n".join(lines) + "\n")


def main():
    plan = ROOT / "multiseed_plan.json"
    if not plan.exists():
        plan.write_text(json.dumps({"declared_utc": datetime.now(timezone.utc).isoformat(),
            "seeds": SEEDS, "models": MODELS, "task": "math_500", "reused_seed": 1234,
            "new_runs": 6, "training_runs": 0, "sampling_seed": None,
            "purpose": "Estimate whether local/author 12-question gap persists across inference seeds before any long retraining."}, indent=2))
    status("waiting_for_controlled_merge")
    deadline = time.monotonic() + 7200
    while not (ROOT / "controlled_merge_results.json").exists():
        if time.monotonic() > deadline:
            raise TimeoutError("Controlled merge evaluation did not complete")
        time.sleep(15)
    assert json.loads((ROOT / "controlled_merge_results.json").read_text())["status"] == "complete"
    # The inference processes finish saving/scoring before releasing their vLLM workers.
    while True:
        used = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
        if all(int(x.strip()) < 1024 for x in used.splitlines()):
            break
        if time.monotonic() > deadline:
            raise TimeoutError("GPUs were not released after controls")
        time.sleep(15)
    for seed in SEEDS:
        for name, model in MODELS.items():
            out = output_dir(name, seed)
            if (out / "summary.json").exists():
                continue
            status("running", seed=seed, model=name)
            command = [PYTHON, "-u", "reproduction/recheck_official_eval.py", "--model", model,
                       "--output", str(out), "--tasks", "math_500", "--seed", str(seed)]
            with open(f"logs/attribution_seed{seed}_{name}.log", "w") as logfile:
                subprocess.run(command, stdout=logfile, stderr=subprocess.STDOUT, check=True)
    aggregate()
    status("complete", results=str(ROOT / "multiseed_results.json"))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        status("failed", error=repr(error))
        raise
