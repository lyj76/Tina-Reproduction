"""Compare predeclared first-three-step controls, not benchmark-selected seeds."""
import json
from pathlib import Path
import statistics

from safetensors.torch import load_file
import torch


ROOT = Path("artifacts/trajectory")
MODEL_ROOT = Path("artifacts/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B")
KEYS = ["completion_length", "rewards/format_reward", "rewards/cosine_scaled_reward", "grad_norm", "kl", "learning_rate"]


def load_history(path):
    data = json.loads(path.read_text())["log_history"]
    return {int(row["step"]): row for row in data if "reward" in row and row["step"] <= 3}


def main():
    history = json.loads(Path("artifacts/recheck/3t1h5abx_history.json").read_text())
    official = {int(row["train/global_step"]): {key.removeprefix("train/"): value for key, value in row.items()}
                for row in history if isinstance(row.get("train/reward"), (int, float)) and row["train/global_step"] <= 3}
    original = load_history(MODEL_ROOT / "grpo_open_rs3/trainer_state.json")
    current_inputs = json.loads((ROOT / "current_inputs.json").read_text())
    author_inputs = json.loads((ROOT / "author_inputs.json").read_text())
    assert current_inputs["source_sha256"] == author_inputs["source_sha256"]
    assert current_inputs["token_ids_sha256"] == author_inputs["token_ids_sha256"]
    assert current_inputs["first24_indices"] == author_inputs["first24_indices"]
    conditions, traces = {}, {}
    for name in ["current42", "author42"]:
        rows = load_history(MODEL_ROOT / f"trajectory_{name}/trainer_state.json")
        assert set(rows) == {1, 2, 3}
        conditions[name] = {
            "steps": rows,
            "mean_absolute_error_vs_author": {key: statistics.mean(abs(rows[s][key] - official[s][key]) for s in [1, 2, 3]) for key in KEYS},
            "mean_absolute_error_vs_original_local": {key: statistics.mean(abs(rows[s][key] - original[s][key]) for s in [1, 2, 3]) for key in KEYS},
            "environment": json.loads((ROOT / name / "environment.json").read_text())}
        traces[name] = {}
        for rank in [0, 1]:
            path = ROOT / name / f"rollouts_rank{rank}.jsonl"
            data = [json.loads(line) for line in path.read_text().splitlines()]
            assert len(data) == 12 and all(len(row["samples"]) == 6 for row in data)
            assert [row["global_step_before_update"] for row in data] == [0]*4 + [1]*4 + [2]*4
            for i, row in enumerate(data):
                expected = current_inputs["first24"][2*i+rank]
                assert all(sample["problem"] == expected["problem"] and sample["vllm_prompt_ids"] == expected["tokens"] for sample in row["samples"])
            traces[name][rank] = data
    comparisons = []
    for rank in [0, 1]:
        for i, (a, b) in enumerate(zip(traces["current42"][rank], traces["author42"][rank])):
            assert [row["problem"] for row in a["samples"]] == [row["problem"] for row in b["samples"]]
            comparisons.append({"rank": rank, "microbatch": i,
                "step_before_update": a["global_step_before_update"],
                "prompt_tokens_identical": all(x["vllm_prompt_ids"] == y["vllm_prompt_ids"] for x, y in zip(a["samples"], b["samples"])),
                "completion_tokens_identical_count": sum(x["completion_ids"] == y["completion_ids"] for x, y in zip(a["samples"], b["samples"])),
                "rng_before_identical": a["rng_before_prepare"] == b["rng_before_prepare"],
                "rng_after_identical": a["rng_after_prepare"] == b["rng_after_prepare"]})
    current_a = load_file(ROOT / "current42/initial_lora_A.safetensors")
    author_a = load_file(ROOT / "author42/initial_lora_A.safetensors")
    initial_equal = all(torch.equal(current_a[key], author_a[key]) for key in current_a)
    result = {"status": "complete", "official_steps": official, "original_local_steps": original,
        "all7000_prompt_tokens_identical_across_environments": True,
        "actual_first24_training_inputs_match_reconstructed_seed42_order": True,
        "conditions": conditions, "initial_A_bitwise_equal_across_environments": initial_equal,
        "rollout_comparisons": comparisons,
        "official50_vs_original_local50": json.loads((ROOT / "official_local50_comparison.json").read_text()),
        "scope": "Early-trajectory calibration at fixed seed42, not a search for a high benchmark seed. Same4090 hardware, memory0.25, xgrammar0.1.17. No claim that matching3 aggregate log rows recovers the author's RNG or final accuracy."}
    (ROOT / "results.json").write_text(json.dumps(result, indent=2))
    lines = ["# 早期训练轨迹校准", "", "两组完整长度、原1500步调度器的三步训练已完成。训练seed42、vLLM默认seed0；两组均保留本机显存预算0.25，未对齐作者硬件和已下架xgrammar0.1.15。", "",
        "| 指标：前3步对作者日志的平均绝对偏差 | 当前依赖 | 作者依赖 |", "|---|---:|---:|"]
    for key in KEYS:
        a = conditions["current42"]["mean_absolute_error_vs_author"][key]
        b = conditions["author42"]["mean_absolute_error_vs_author"][key]
        lines.append(f"| {key} | {a:.8g} | {b:.8g} |")
    lines += ["", f"两组初始LoRA A位级一致：{initial_equal}。7000条训练输入token一致；实际前24题及双卡顺序与seed42重建一致。", "",
        "作者W&B训练seed42；公开训练代码的vLLM默认seed0。公开completion table是50步起的固定题输出，不是最初batch逐条rollout。", "",
        "原50步LoRA与作者50步：A方向余弦0.99999999964，B方向余弦0.668484。A接近不代表B更新轨迹一致。", "",
        "这些结果用于定位环境/轨迹差异；三个聚合日志的接近程度不等于最终性能。完整统计和逐microbatch输入、生成token、RNG哈希见 `artifacts/trajectory/`。", "",
        "当前决策：保留原有独立训练结果，先补850步官方协议评测（见ENDPOINT850.md）。尚未启动长程重训或作者50步起点续训；后者仅作明确标注的辅助备选。"]
    Path("TRAJECTORY.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"initial_A_equal": initial_equal, "errors": {name: data["mean_absolute_error_vs_author"] for name, data in conditions.items()}}, indent=2))


if __name__ == "__main__":
    main()
