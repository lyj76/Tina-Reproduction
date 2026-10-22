"""Audit the two predeclared dynamic-versus-merged inference controls."""
import hashlib
import json
from math import comb
from pathlib import Path


ROOT = Path("artifacts/attribution")


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def exact_p(gains, losses):
    n = gains + losses
    return min(1.0, 2 * sum(comb(n, i) for i in range(min(gains, losses) + 1)) / 2**n) if n else 1.0


def main():
    pending = [name for name in ("local500", "official500")
               if not (ROOT / f"{name}_merged_seed42.summary.json").exists()]
    if pending:
        print(json.dumps({"status": "running", "pending": pending}))
        return
    results = {}
    for name in ("local500", "official500"):
        old_path = Path(f"artifacts/recheck/{name}_dynamic_seed42.jsonl")
        new_path = ROOT / f"{name}_merged_seed42.jsonl"
        old, new = read_rows(old_path), read_rows(new_path)
        regrade = json.loads(old_path.with_suffix(".regrade.json").read_text())
        summary = json.loads(new_path.with_suffix(".summary.json").read_text())
        assert len(old) == len(new) == len(regrade["rows"]) == summary["samples"] == 500
        assert summary["scorer"] == "lighteval" and summary["adapter"] is None
        for key, expected in {"seed": 42, "max_tokens": 32768, "temperature": 0.6,
                              "top_p": 0.95, "tensor_parallel_size": 1,
                              "max_num_seqs": 32, "max_model_len": 33792}.items():
            assert summary[key] == expected, (key, summary[key])
        gains, losses, changed_text, rows = [], [], [], []
        for a, b, grade in zip(old, new, regrade["rows"]):
            for key in ("index", "unique_id", "problem", "gold"):
                assert a[key] == b[key], (name, key, a["index"])
            assert grade["index"] == a["index"]
            before, after = bool(grade["lighteval_correct"]), bool(b["correct"])
            if not before and after:
                gains.append(a["index"])
            if before and not after:
                losses.append(a["index"])
            if a["prediction"] != b["prediction"]:
                changed_text.append(a["index"])
            rows.append({"index": a["index"], "unique_id": a["unique_id"],
                         "dynamic_official_metric_correct": before,
                         "merged_official_metric_correct": after,
                         "prediction_identical": a["prediction"] == b["prediction"]})
        old_count = sum(row["lighteval_correct"] for row in regrade["rows"])
        new_count = sum(row["correct"] for row in new)
        assert old_count == regrade["lighteval_correct"]
        assert new_count == summary["correct"]
        assert new_count - old_count == len(gains) - len(losses)
        results[name] = {
            "samples": 500, "dynamic_correct": old_count, "merged_correct": new_count,
            "delta_percentage_points": (new_count - old_count) / 5,
            "gains": gains, "losses": losses, "changed_predictions": changed_text,
            "mcnemar_exact_p": exact_p(len(gains), len(losses)),
            "dynamic_truncated": sum(x["finish_reason"] == "length" for x in old),
            "merged_truncated": summary["truncated"],
            "dynamic_mean_tokens": sum(x["tokens"] for x in old) / 500,
            "merged_mean_tokens": summary["mean_tokens"],
            "input_parity": True,
            "dynamic_sha256": hashlib.sha256(old_path.read_bytes()).hexdigest(),
            "merged_sha256": hashlib.sha256(new_path.read_bytes()).hexdigest(),
            "rows": rows,
        }
    output = {"status": "complete", "results": results,
              "scope": "One sampled run per model, fixed old direct-vLLM protocol. Intervention changes dynamic LoRA to BF16 merged weights, including execution kernels. It does not isolate training differences or establish multi-seed mean effects."}
    temp = ROOT / "controlled_merge_results.tmp"
    temp.write_text(json.dumps(output, indent=2))
    temp.replace(ROOT / "controlled_merge_results.json")
    lines = ["两个完整 500 题控制已完成，输入顺序及官方判分核验通过。", "",
             "| 权重 | 动态 LoRA（官方重判） | BF16 合并 | 差值 | 新增答对 / 新增答错 | 生成文本改变 |",
             "|---|---:|---:|---:|---:|---:|"]
    for name, r in results.items():
        label = "本地500" if name == "local500" else "作者500"
        lines.append(f"| {label} | {r['dynamic_correct']}/500 ({r['dynamic_correct']/5:.1f}%) | "
                     f"{r['merged_correct']}/500 ({r['merged_correct']/5:.1f}%) | "
                     f"{r['delta_percentage_points']:+.1f} pp | {len(r['gains'])} / {len(r['losses'])} | "
                     f"{len(r['changed_predictions'])}/500 |")
    lines += ["", "这里仅改变权重加载路径；结果可用于判断这一改变在固定条件下的影响。它没有恢复训练回调或训练依赖，也不能将官方路径下本地与作者的差距全部归因于合并方式。",
              "", "完整统计、输出 SHA256 与逐题得失：[controlled_merge_results.json](artifacts/attribution/controlled_merge_results.json)。"]
    path = Path("ATTRIBUTION.md")
    report = path.read_text()
    start, end = "<!-- CONTROLLED_RESULTS -->", "<!-- /CONTROLLED_RESULTS -->"
    prefix, rest = report.split(start, 1)
    _, suffix = rest.split(end, 1)
    path.write_text(prefix + start + "\n" + "\n".join(lines) + "\n" + end + suffix)
    print(json.dumps({"status": "complete", "results": {
        name: {k: v for k, v in r.items() if k not in ("rows", "changed_predictions", "gains", "losses")}
        | {"gains": len(r["gains"]), "losses": len(r["losses"]),
           "changed_predictions": len(r["changed_predictions"])} for name, r in results.items()}}, indent=2))


if __name__ == "__main__":
    main()
