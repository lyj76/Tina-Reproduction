"""Detached, auditable evaluation queue; wait for an existing GPU evaluation to finish."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time


parser = argparse.ArgumentParser()
parser.add_argument("--lane", choices=["all"], default="all")
parser.add_argument("--wait-pid", type=int, nargs="+", required=True)
args = parser.parse_args()
root = Path("artifacts/recheck")
python = str(root.resolve() / "eval_env/bin/python")
status_path = root / f"queue_{args.lane}.json"


def status(state, **kwargs):
    record = {"lane": args.lane, "status": state, "updated_utc": datetime.now(timezone.utc).isoformat(),
              "pid": os.getpid(), **kwargs}
    temp = status_path.with_suffix(".tmp")
    temp.write_text(json.dumps(record, indent=2))
    temp.replace(status_path)
    print(json.dumps(record), flush=True)


status("waiting", wait_pids=args.wait_pid)
for pid in args.wait_pid:
    while Path(f"/proc/{pid}/cmdline").exists():
        cmd = Path(f"/proc/{pid}/cmdline").read_bytes()
        if b"reproduction/evaluate.py" not in cmd:
            break
        time.sleep(20)

# Regrade the two dynamic-LoRA controls before running the merged-weight controls.
status("regrading")
sources = [root / "local500_dynamic_seed42.jsonl", root / "official500_dynamic_seed42.jsonl"]
for source in sources:
    if not source.exists() or len(source.read_text().splitlines()) != 500:
        status("failed", reason=f"Dynamic control incomplete: {source}")
        sys.exit(1)
with open("logs/recheck_dynamic_regrade.log", "w") as logfile:
    result = subprocess.run([python, "-u", "reproduction/recheck_grade.py", *map(str, sources)],
                            stdout=logfile, stderr=subprocess.STDOUT)
if result.returncode:
    status("failed", reason="Dynamic control regrading failed")
    sys.exit(result.returncode)

models = [
    ("base", "artifacts/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base"),
    ("official500", "artifacts/recheck/merged_official500"),
    ("local500", "artifacts/recheck/merged_local500")]
for name, model in models:
    for task in ["math_500", "aime24", "amc23"]:
        output = root / f"official_pipeline_{name}" / task
        if (output / "summary.json").exists():
            continue
        status("running", model=name, task=task)
        command = [python, "-u", "reproduction/recheck_official_eval.py", "--model", model,
                   "--output", str(output), "--tasks", task]
        with open(f"logs/recheck_pipeline_{name}_{task}.log", "w") as logfile:
            result = subprocess.run(command, stdout=logfile, stderr=subprocess.STDOUT)
        if result.returncode:
            status("failed", model=name, task=task, returncode=result.returncode)
            sys.exit(result.returncode)
with open("logs/recheck_results.log", "w") as logfile:
    result = subprocess.run([python, "-u", "reproduction/recheck_results.py"], stdout=logfile, stderr=subprocess.STDOUT)
if result.returncode:
    status("failed", reason="Result aggregation failed")
    sys.exit(result.returncode)
status("complete")
