"""Export the completed training history as CSV, JSON, PNG, and SVG."""
import csv
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).resolve().parents[1]
model = root / "artifacts/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/grpo_open_rs3"
state = json.loads((model / "trainer_state.json").read_text())
rows = [r for r in state["log_history"] if "grad_norm" in r]
out = root / "artifacts/metrics"
out.mkdir(parents=True, exist_ok=True)
fields = ["step"] + sorted(set().union(*(r.keys() for r in rows)) - {"step"})
with (out / "training.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    writer.writerows(rows)
window = 25
steps = np.array([r["step"] for r in rows])

def smooth(values):
    return np.convolve(values, np.ones(window), "full")[:len(values)] / np.minimum(np.arange(1, len(values) + 1), window)

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 180})
fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, constrained_layout=True)
panels = [("reward", "Weighted training reward"), ("rewards/format_reward", "Format reward"),
          ("completion_length", "Mean completion length (tokens)"), ("kl", "KL divergence")]
for ax, (key, title) in zip(axes.flat, panels):
    values = np.array([r[key] for r in rows])
    ax.plot(steps, values, color="#9AA8B5", alpha=.4, linewidth=.65, label="Per step")
    ax.plot(steps, smooth(values), color="#176D9C", linewidth=1.8, label="25-step trailing mean")
    ax.axvline(450, color="#B95F36", linestyle="--", linewidth=.9, alpha=.75)
    ax.set_title(title, loc="left")
    ax.grid(alpha=.15)
    ax.set_xlim(1, 850)
axes[0, 0].legend(frameon=False, fontsize=8)
for ax in axes[1]: ax.set_xlabel("Optimizer step")
fig.suptitle("TINA Open-RS3: LoRA-GRPO training, 850 steps", fontsize=14)
for ext in ["png", "svg"]: fig.savefig(out / f"training_curves.{ext}")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
grad = np.array([r["grad_norm"] for r in rows])
axes[0].plot(steps, grad, color="#9AA8B5", alpha=.4, linewidth=.65)
axes[0].plot(steps, smooth(grad), color="#176D9C", linewidth=1.8)
axes[0].set_title("Gradient norm", loc="left")
axes[1].plot(steps, [r["learning_rate"] for r in rows], color="#7757A8", linewidth=1.8)
axes[1].axvline(150, color="#B95F36", linestyle="--", linewidth=.9, label="Warmup ends")
axes[1].set_title("Learning rate (1500-step schedule, stopped at 850)", loc="left", fontsize=10)
axes[1].legend(frameon=False, fontsize=8)
for ax in axes:
    ax.set_xlabel("Optimizer step")
    ax.set_xlim(1, 850)
    ax.grid(alpha=.15)
for ext in ["png", "svg"]: fig.savefig(out / f"optimization_curves.{ext}")
plt.close(fig)
summary = {"global_step": state["global_step"], "training_seconds": state["log_history"][-1]["train_runtime"]}
for label, chunk in [("first50", rows[:50]), ("last50", rows[-50:])]:
    summary[label] = {key: float(np.mean([r[key] for r in chunk])) for key, _ in panels}
(out / "training_summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
