"""Verify saved multi-seed predictions against summary scores and input parity."""
import ast
from datetime import datetime, timezone
import hashlib
import json
from math import comb
from pathlib import Path

from datasets import load_from_disk
import pyarrow.parquet as pq

from reproduction.attribution_multiseed import MODELS, SEEDS, output_dir


def value(item):
    return ast.literal_eval(item) if isinstance(item, str) else item


def paired(left, right):
    gains = sum(not a["correct"] and b["correct"] for a, b in zip(left, right))
    losses = sum(a["correct"] and not b["correct"] for a, b in zip(left, right))
    n = gains + losses
    p = min(1.0, 2 * sum(comb(n, i) for i in range(min(gains, losses) + 1)) / 2**n) if n else 1.0
    return {"gains": gains, "losses": losses, "mcnemar_exact_p": p}


def main():
    root = Path("artifacts/attribution")
    results = json.loads((root / "multiseed_results.json").read_text())
    assert results["status"] == "complete"
    math_ids = {x["problem"]: x["unique_id"] for x in load_from_disk("artifacts/data/math500")}
    overlap = set(json.loads(Path("artifacts/data_overlap_audit.json").read_text())["normalized_overlap_ids"])
    reference, audits, comparisons = None, {}, {}
    for seed in SEEDS:
        outputs = {}
        audits[str(seed)] = {}
        for name in MODELS:
            folder = output_dir(name, seed)
            files = list((folder / "details").glob("**/*.parquet"))
            assert len(files) == 1, folder
            raw = pq.read_table(files[0], columns=["example", "full_prompt", "gold", "predictions", "metrics",
                "specifics", "num_asked_few_shots", "num_effective_few_shots"]).to_pylist()
            assert len(raw) == 500 and len({x["full_prompt"] for x in raw}) == 500
            rows = []
            for i, row in enumerate(raw):
                assert row["num_asked_few_shots"] == row["num_effective_few_shots"] == 0
                predictions, metrics = value(row["predictions"]), value(row["metrics"])
                assert len(predictions) == 1 and predictions[0]
                assert metrics["extractive_match"] in (0.0, 1.0)
                problem = row["example"].split("\n\n", 1)[1]
                rows.append({"index": i, "unique_id": math_ids[problem], "problem": problem,
                    "gold": value(row["gold"]), "prediction": predictions[0],
                    "correct": bool(metrics["extractive_match"]), "extracted": value(row["specifics"]),
                    "full_prompt": row["full_prompt"]})
            inputs = [(x["unique_id"], x["problem"], x["full_prompt"], x["gold"]) for x in rows]
            if reference is None:
                reference = inputs
            assert inputs == reference, (seed, name, "input/order mismatch")
            correct = sum(x["correct"] for x in rows)
            assert correct == results["records"][str(seed)][name]["correct"]
            summary = json.loads((folder / "summary.json").read_text())
            assert abs(correct / 500 - summary["results"]["custom:math_500:0"]["extractive_match"]) < 1e-12
            # Preserve previously audited seed1234 exports; export the six new runs.
            if seed != 1234:
                (folder / "predictions.jsonl").write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in rows))
            clean = [x for x in rows if x["unique_id"] not in overlap]
            assert len(clean) == 494
            audits[str(seed)][name] = {"samples": 500, "correct": correct, "zero_shot_verified": True,
                "nonoverlap_samples": 494, "nonoverlap_correct": sum(x["correct"] for x in clean),
                "parquet_sha256": hashlib.sha256(files[0].read_bytes()).hexdigest(),
                "source": str(files[0])}
            outputs[name] = rows
        comparisons[str(seed)] = {
            "local_vs_base": paired(outputs["base"], outputs["local500"]),
            "local_vs_official": paired(outputs["official500"], outputs["local500"]),
            "official_vs_base": paired(outputs["base"], outputs["official500"])}
    output = {"status": "passed", "completed_utc": datetime.now(timezone.utc).isoformat(),
        "evaluations_verified": 9, "prediction_rows_verified": 4500, "new_prediction_rows_verified": 3000,
        "same_questions_prompts_golds_and_order": True, "counts": audits, "paired_per_seed": comparisons,
        "scope": "Three inference seeds, one training seed. Repeated questions are not independent new questions; paired tests are per seed, not pooled."}
    (root / "multiseed_final_audit.json").write_text(json.dumps(output, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in output.items() if k != "counts"}, indent=2))


if __name__ == "__main__":
    main()
