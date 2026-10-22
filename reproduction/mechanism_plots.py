"""Export like-for-like training curves for the mechanism comparison on CPU."""
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    root=Path("artifacts/mechanism"); model_root=Path("artifacts/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B")
    sources={"RS3: format + cosine":model_root / "grpo_open_rs3",
        "Published format recipe":model_root / "grpo_official_format_core",
        "Label-free format control":model_root / "grpo_label_free_format_core"}
    fig, axes=plt.subplots(2,2,figsize=(12,7),constrained_layout=True)
    keys=[("rewards/format_reward","Format reward"),("completion_length","Completion length (tokens)"),
        ("rewards/len_reward","Length reward (different correctness dependence)"),("kl","KL to base")]
    for label, path in sources.items():
        state=path / "trainer_state.json"
        if not state.exists(): continue
        rows=[r for r in json.loads(state.read_text())["log_history"] if "loss" in r]
        if not rows: continue
        out=root / (path.name+"_training.csv")
        fields=["step"]+sorted(set().union(*(r.keys() for r in rows)) - {"step"})
        with out.open("w",newline="") as handle:
            writer=csv.DictWriter(handle,fieldnames=fields);writer.writeheader();writer.writerows(rows)
        for axis,(key,title) in zip(axes.flat,keys):
            available=[r for r in rows if key in r]
            if not available: continue
            smoothed=[]; current=None
            for row in available:
                current=row[key] if current is None else 0.1*row[key]+0.9*current
                smoothed.append(current)
            line,=axis.plot([r["step"] for r in available],smoothed,label=label,linewidth=1.4)
            axis.plot([r["step"] for r in available],[r[key] for r in available],color=line.get_color(),alpha=.12,linewidth=.5)
    for axis,(_,title) in zip(axes.flat,keys):
        axis.set_title(title);axis.set_xlabel("Optimizer step");axis.grid(alpha=.15)
        axis.axvline(500,color="gray",linestyle="--",alpha=.5);axis.axvline(850,color="gray",linestyle=":",alpha=.5)
        if axis.get_legend_handles_labels()[0]: axis.legend(fontsize=8)
    fig.suptitle("TINA mechanism: fixed checkpoints; EMA=0.1; no causal turning point assumed")
    for extension in ("png","svg"):fig.savefig(root / f"training_comparison.{extension}",dpi=180)
    plt.close(fig)


if __name__=="__main__":main()
