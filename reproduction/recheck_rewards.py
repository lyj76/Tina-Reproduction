"""Audit all official training labels and characterize the inherited reward code."""
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path

from datasets import load_from_disk
from math_verify import LatexExtractionConfig, parse, verify


def check(row):
    index, solution, answer = row
    gold = parse(solution, extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
    direct = parse(f"${answer}$", extraction_mode="first_match", extraction_config=[LatexExtractionConfig()])
    consistent = bool(verify(gold, direct)) if gold and direct else None
    return {"index": index, "gold_parseable": bool(gold), "answer_parseable": bool(direct),
            "solution_answer_match": consistent,
            "solution_extracted": [str(x) for x in gold], "answer_extracted": [str(x) for x in direct]}


if __name__ == "__main__":
    data = load_from_disk("artifacts/data/open-rs")
    inputs = [(i, row["solution"], row["answer"]) for i, row in enumerate(data)]
    with ProcessPoolExecutor(max_workers=8) as pool:
        rows = list(pool.map(check, inputs, chunksize=8))
    from tina.post_train_hf.rewards import get_cosine_scaled_reward
    reward = get_cosine_scaled_reward(min_value_wrong=0.0, max_value_wrong=-0.5,
                                     min_value_correct=0.5, max_value_correct=1.0, max_len=3584)
    probes = []
    for length in [1792, 3584, 7168, 10752, 14336]:
        tail = r"</think><answer>$\boxed{1}$</answer>"
        content = "<think>" + "x" * (length - len("<think>") - len(tail)) + tail
        assert len(content) == length
        value = reward(completions=[[{"content": content}]], solution=["$1$"])[0]
        probes.append({"characters": length, "correct_answer_reward": value})
    report = {"samples": len(rows), "unparseable_solution_count": sum(not r["gold_parseable"] for r in rows),
              "solution_answer_mismatch_count": sum(r["solution_answer_match"] is False for r in rows),
              "rows": rows, "cosine_probes": probes,
              "scope": "Official unchanged recipe: training uses solution, not answer, when both columns exist. Parse failures receive unconditional reward1.0. Cosine uses unclamped character length. These behaviors are shared with the author's implementation and do not establish a local reproduction bug."}
    Path("artifacts/recheck/reward_audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2), flush=True)
