#!/usr/bin/env python3
"""Generate a high-impact, publication-grade Hero Figure for Tina Reproduction README."""

import json
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# Set professional font and styling
plt.rcParams.update({
    "font.sans-serif": ["DejaVu Sans", "Helvetica", "Arial"],
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 8.5,
    "figure.titlesize": 13,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#CBD5E1",
    "axes.linewidth": 1.0,
    "grid.color": "#E2E8F0",
    "grid.linestyle": "--",
    "grid.alpha": 0.7,
})

repo_root = Path("/mnt/d/实验数据/autodl-tmp/复现")
assets_dir = repo_root / "assets"
assets_dir.mkdir(parents=True, exist_ok=True)

# 1. Data Preparation for Benchmark Comparison
benchmarks = ["AIME24", "AIME25", "AMC23", "MATH500", "GPQA-D", "Minerva", "Average"]

# Data points: [AIME24, AIME25, AMC23, MATH500, GPQA, Minerva, Average]
base_scores = [33.33, 16.67, 75.00, 83.40, 35.86, 29.04, 45.55]
author_scores = [33.33, 23.33, 70.00, 85.60, 37.37, 29.78, 46.57]
local_4090_scores = [43.33, 20.00, 80.00, 83.20, 37.88, 32.35, 49.46]
ablation_scores = [26.67, 30.00, 67.50, 83.40, 32.83, 25.74, 44.35]

x = np.arange(len(benchmarks))
width = 0.20

# Create figure layout
fig = plt.figure(figsize=(15, 6.2), constrained_layout=True)
gs = fig.add_gridspec(2, 2, width_ratios=[1.35, 1.0], height_ratios=[1.0, 1.0])

# ==========================================
# Panel A: 6-Benchmark + Average Bar Chart
# ==========================================
ax_bars = fig.add_subplot(gs[:, 0])

c_base = "#94A3B8"        # Slate grey
c_author = "#3B82F6"      # Royal blue
c_4090 = "#10B981"        # Emerald green (NVIDIA / Success)
c_ablation = "#F43F5E"    # Rose red (Warning / Collapse)

rects1 = ax_bars.bar(x - 1.5*width, base_scores, width, label="Base (Qwen-1.5B)", color=c_base, alpha=0.9, edgecolor="none", zorder=3)
rects2 = ax_bars.bar(x - 0.5*width, author_scores, width, label="Author Official 500-step (L40S)", color=c_author, alpha=0.9, edgecolor="none", zorder=3)
rects3 = ax_bars.bar(x + 0.5*width, local_4090_scores, width, label="Reproduction 500-step (2× RTX 4090)", color=c_4090, alpha=0.95, edgecolor="#047857", linewidth=1.2, zorder=4)
rects4 = ax_bars.bar(x + 1.5*width, ablation_scores, width, label="True Label-Free Control (Ablation 850s)", color=c_ablation, alpha=0.85, edgecolor="none", zorder=3)

# Add highlight separator line before Average
ax_bars.axvline(5.5, color="#94A3B8", linestyle="--", linewidth=1.2, alpha=0.6, zorder=2)
ax_bars.text(5.55, 92, "Overall Mean", color="#475569", fontsize=9, fontweight="bold")

# Annotate AIME24 jump
ax_bars.annotate('+10.0%', 
                 xy=(x[0] + 0.5*width, local_4090_scores[0]), 
                 xytext=(x[0] + 0.5*width, local_4090_scores[0] + 4.5),
                 ha='center', va='bottom', fontsize=9, fontweight='bold', color='#047857',
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="#D1FAE5", edgecolor="#10B981", alpha=0.9),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", color="#047857", lw=1.2))

# Annotate Average gain
ax_bars.annotate('+3.91 pp\n(49.46%)', 
                 xy=(x[6] + 0.5*width, local_4090_scores[6]), 
                 xytext=(x[6] + 0.5*width, local_4090_scores[6] + 6.0),
                 ha='center', va='bottom', fontsize=8.5, fontweight='bold', color='#047857',
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="#D1FAE5", edgecolor="#10B981", alpha=0.9),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", color="#047857", lw=1.2))

# Annotate Ablation collapse on Average
ax_bars.annotate('-1.20 pp\nCollapse', 
                 xy=(x[6] + 1.5*width, ablation_scores[6]), 
                 xytext=(x[6] + 1.5*width, ablation_scores[6] - 12.0),
                 ha='center', va='top', fontsize=8, fontweight='bold', color='#BE123C',
                 bbox=dict(boxstyle="round,pad=0.2", facecolor="#FFE4E6", edgecolor="#F43F5E", alpha=0.9),
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0", color="#BE123C", lw=1.0))

ax_bars.set_ylabel("Pass@1 Accuracy (%)", fontweight="bold")
ax_bars.set_title("A. Reasoning Performance Across 6 Standard Benchmarks (LightEval BF16 Protocol)", fontweight="bold", loc="left", pad=10)
ax_bars.set_xticks(x)
ax_bars.set_xticklabels(benchmarks, fontweight="semibold")
ax_bars.set_ylim(0, 100)
ax_bars.grid(axis="y", zorder=0)
ax_bars.legend(loc="upper left", framealpha=0.95, edgecolor="#E2E8F0")

# ==========================================
# Panel B: Training Dynamics (Length Growth)
# ==========================================
ax_len = fig.add_subplot(gs[0, 1])

# Load training curves
official_file = repo_root / "artifacts/recheck/3t1h5abx_history.json"
local_file = repo_root / "artifacts/metrics/training.csv"

def smooth_data(values, window=25):
    return np.convolve(values, np.ones(window)/window, mode='valid')

if official_file.exists() and local_file.exists():
    official_data = json.loads(official_file.read_text())
    clean_official = [
        (x["train/global_step"], float(x["train/completion_length"]))
        for x in official_data
        if isinstance(x.get("train/completion_length"), (int, float)) and x.get("train/global_step", 9999) <= 850
    ]
    off_steps = [item[0] for item in clean_official]
    off_lens = [item[1] for item in clean_official]
    
    local_rows = list(csv.DictReader(local_file.open()))
    loc_steps = [int(r["step"]) for r in local_rows if r.get("completion_length")]
    loc_lens = [float(r["completion_length"]) for r in local_rows if r.get("completion_length")]
    
    # Smooth
    w = 20
    if len(off_lens) >= w:
        ax_len.plot(off_steps, off_lens, color=c_author, alpha=0.15, lw=0.6)
        ax_len.plot(off_steps[w-1:], smooth_data(off_lens, w), color=c_author, lw=2.0, label="Author (L40S Cluster)")
    
    if len(loc_lens) >= w:
        ax_len.plot(loc_steps, loc_lens, color="#EA580C", alpha=0.15, lw=0.6)
        ax_len.plot(loc_steps[w-1:], smooth_data(loc_lens, w), color="#EA580C", lw=2.0, label="Local Reproduction (2× RTX 4090)")
    
    ax_len.axvline(500, color="#64748B", linestyle=":", lw=1.2)
    ax_len.text(508, 1200, "Optimal Step 500", color="#475569", fontsize=8, fontstyle="italic")

ax_len.set_title("B. Thinking Length Expansion: Author L40S vs. Local 2× RTX 4090", fontweight="bold", loc="left", pad=8)
ax_len.set_ylabel("Completion Tokens", fontweight="bold")
ax_len.set_xlim(0, 850)
ax_len.set_ylim(0, 3600)
ax_len.grid(True, zorder=0)
ax_len.legend(loc="lower right", framealpha=0.9, edgecolor="#E2E8F0")

# ==========================================
# Panel C: Hardware & Mechanism Insights Card
# ==========================================
ax_card = fig.add_subplot(gs[1, 1])
ax_card.axis("off")

# Draw modern card containers
card_bbox1 = patches.FancyBboxPatch((0.02, 0.52), 0.96, 0.44, boxstyle="round,pad=0.03", 
                                   facecolor="#F0FDF4", edgecolor="#86EFAC", linewidth=1.2)
ax_card.add_patch(card_bbox1)

ax_card.text(0.06, 0.86, "Hardware: Consumer Feasibility (2× RTX 4090 48GB)", 
             fontweight="bold", fontsize=10.5, color="#166534")
ax_card.text(0.06, 0.60, 
             "• Zero OOM via vLLM memory tuning: gpu_memory_utilization = 0.25\n"
             "• Full recipe fidelity: batch size 6, grad-acc 4, 6 rollouts per prompt\n"
             "• Total post-training compute budget: ~$8.5 USD on commodity hardware", 
             fontsize=9, color="#14532D", linespacing=1.4)

card_bbox2 = patches.FancyBboxPatch((0.02, 0.02), 0.96, 0.44, boxstyle="round,pad=0.03", 
                                   facecolor="#FFF1F2", edgecolor="#FECDD3", linewidth=1.2)
ax_card.add_patch(card_bbox2)

ax_card.text(0.06, 0.36, "Mechanism Audit: The 'Format-Only' Reality", 
             fontweight="bold", fontsize=10.5, color="#9F1239")
ax_card.text(0.06, 0.10, 
             "• Code Audit: Upstream length reward secretly referenced ground-truth labels!\n"
             "• Counterfactual Test: Removing label dependency collapsed score to 44.35% (-1.2 pp)\n"
             "• Conclusion: Structural formatting alone is insufficient without answer grounding", 
             fontsize=9, color="#881337", linespacing=1.4)

fig.suptitle("Tina (DeepSeek-R1-Distill-Qwen-1.5B via LoRA+GRPO): Full Reproduction & Mechanistic Audit on 2× RTX 4090", 
             fontweight="bold", fontsize=13.5, y=1.01)

# Save high-res PNG and vector SVG
out_png = assets_dir / "reproduction_hero.png"
out_svg = assets_dir / "reproduction_hero.svg"
plt.savefig(out_png, dpi=220, bbox_inches="tight")
plt.savefig(out_svg, bbox_inches="tight")
plt.close(fig)

print(f"Generated hero figure: {out_png} ({out_png.stat().st_size} bytes)")
