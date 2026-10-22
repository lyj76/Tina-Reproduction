"""Regrade saved generations with the unmodified official LightEval metric."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import importlib.util
import json
from pathlib import Path


def initialize():
    global tasks
    spec = importlib.util.spec_from_file_location("tina_tasks", "scripts/eval/run_eval_custom_tasks.py")
    tasks = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tasks)


def grade(row):
    doc = tasks.math_prompt_fn({"problem": row["problem"], "solution": row["gold"]})
    score = tasks.latex_gold_metric.sample_level_fn(
        golds=[row["gold"]], predictions=[row["prediction"]], formatted_doc=doc)
    return {"index": row["index"], "original_correct": row["correct"],
            "lighteval_correct": bool(score), "extracted": doc.specific}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+")
    args = parser.parse_args()
    for path in map(Path, args.paths):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        with ProcessPoolExecutor(max_workers=8, initializer=initialize) as pool:
            scores = list(pool.map(grade, rows, chunksize=8))
        summary = {"source": str(path), "metric": "Official task latex_gold_metric / LightEval 0.8.1",
                   "samples": len(rows), "original_correct": sum(x["correct"] for x in rows),
                   "lighteval_correct": sum(x["lighteval_correct"] for x in scores),
                   "changed_indices": [x["index"] for x in scores if x["original_correct"] != x["lighteval_correct"]],
                   "rows": scores}
        output = Path("artifacts/recheck") / (path.stem + ".regrade.json")
        output.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        print(json.dumps({k: v for k, v in summary.items() if k != "rows"}), flush=True)
