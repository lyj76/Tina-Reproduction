"""Produce traceable diagnostics comparing our run with the author's public run."""
import csv
import hashlib
import json
from pathlib import Path
import statistics

from safetensors.torch import load_file
from scipy.stats import binomtest


ROOT = Path("artifacts/recheck")
MAIN = Path("artifacts/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/grpo_open_rs3")


def average(rows, keys):
    return {key: statistics.mean(float(x[key]) for x in rows if x.get(key) is not None)
            for key in keys if any(x.get(key) is not None for x in rows)}


official = json.loads((ROOT / "3t1h5abx_history.json").read_text())
official = [{key.removeprefix("train/"): value for key, value in row.items()}
            for row in official if isinstance(row.get("train/reward"), (int, float))]
local = list(csv.DictReader(open("artifacts/metrics/training.csv")))
keys = ["reward", "completion_length", "kl", "grad_norm", "rewards/format_reward", "rewards/cosine_scaled_reward"]
windows = []
for lo, hi in [(1, 50), (400, 500), (800, 850)]:
    windows.append({"steps": [lo, hi],
                    "official": average([x for x in official if lo <= x["global_step"] <= hi], keys),
                    "local": average([x for x in local if lo <= int(x["step"]) <= hi], keys)})

adapters = {}
for name, path in {"official500": ROOT / "official_rs3/checkpoint-500", "local500": MAIN / "checkpoint-500"}.items():
    tensors = load_file(path / "adapter_model.safetensors")
    adapters[name] = {"sha256": hashlib.sha256((path / "adapter_model.safetensors").read_bytes()).hexdigest(),
                      "parameters": sum(v.numel() for v in tensors.values()),
                      "dtypes": sorted({str(v.dtype) for v in tensors.values()}),
                      "B_l2": sum(v.float().square().sum().item() for k, v in tensors.items() if "lora_B" in k) ** 0.5}

paired = {}
base = [json.loads(x) for x in Path("artifacts/eval/base.jsonl").read_text().splitlines()]
for name in ["checkpoint450", "checkpoint850"]:
    rows = [json.loads(x) for x in Path(f"artifacts/eval/{name}.jsonl").read_text().splitlines()]
    assert [x["unique_id"] for x in base] == [x["unique_id"] for x in rows]
    gains = sum(not a["correct"] and b["correct"] for a, b in zip(base, rows))
    losses = sum(a["correct"] and not b["correct"] for a, b in zip(base, rows))
    paired[name] = {"gains": gains, "losses": losses,
                    "mcnemar_exact_p": binomtest(gains, gains + losses).pvalue,
                    "caveat": "Single sampled generation per problem; no proof of equivalence or improvement."}

report = {"official_run": "https://wandb.ai/upup-ashton-wang-usc/Tina/runs/3t1h5abx",
          "windows": windows, "adapters": adapters, "legacy_scoring_paired_tests": paired,
          "checkpoint_selection": {
              "source": "https://huggingface.co/Tina-Yi/R1-Distill-Qwen-1.5B-Open-RS3/blob/main/README.md",
              "official_best": 500, "paper_math500_best_checkpoint_score": 85.2,
              "paper_math500_final850_score": 83.8,
              "note": "Selected by six-benchmark mean, not monotonically increasing training steps."}}
(ROOT / "training_comparison.json").write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
