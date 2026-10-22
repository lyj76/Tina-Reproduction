"""Use the actual repository task metric rather than a similar-looking parser."""
from functools import lru_cache
import importlib.util
from pathlib import Path


@lru_cache(maxsize=1)
def official_tasks():
    source = Path(__file__).resolve().parents[1] / "scripts/eval/run_eval_custom_tasks.py"
    spec = importlib.util.spec_from_file_location("tina_official_scoring_tasks", source)
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "Official scoring requires the evaluation environment. Run with "
            "artifacts/recheck/eval_env/bin/python; see reproduction/README.md."
        ) from exc
    return module


def grade_math500(problem, solution, prediction):
    tasks = official_tasks()
    doc = tasks.math_prompt_fn({"problem": problem, "solution": solution})
    return bool(tasks.latex_gold_metric.sample_level_fn(
        golds=[solution], predictions=[prediction], formatted_doc=doc))
