"""Overlay the original author's public training log and the local run."""
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


root = Path("artifacts/recheck")
official = json.loads((root / "3t1h5abx_history.json").read_text())
official = [{k.removeprefix("train/"): v for k, v in x.items()}
            for x in official if isinstance(x.get("train/reward"), (int, float))
            and x["train/global_step"] <= 850]
local = [{k: float(v) for k, v in x.items()} for x in csv.DictReader(open("artifacts/metrics/training.csv"))]


def smooth(rows, key, window=25):
    return [statistics.mean(row[key] for row in rows[max(0, i-window+1):i+1]) for i in range(len(rows))]


fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
for ax, key, title in zip(axes.flat,
                         ["completion_length", "rewards/format_reward", "kl", "reward"],
                         ["Mean completion tokens", "Format reward", "KL to reference", "Weighted training reward"]):
    ax.plot([x["global_step"] for x in official], smooth(official, key), label="Author: L40S", color="#2563eb")
    ax.plot([x["step"] for x in local], smooth(local, key), label="Local: 4090", color="#dc6b14")
    ax.axvline(500, color="#6b7280", linestyle=":", linewidth=1)
    ax.set(title=title, xlabel="Optimizer step")
    ax.grid(alpha=0.2)
axes[0, 0].legend(frameon=False)
fig.suptitle("TINA Open-RS3: public vs local training dynamics (25-step moving mean)", fontsize=12)
fig.savefig(root / "official_vs_local_training.png", dpi=170)
fig.savefig(root / "official_vs_local_training.svg")
print(root / "official_vs_local_training.png")
