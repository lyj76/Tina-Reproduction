"""Keep a long reproduction run attached to a detached screen session.

Adopt an existing run; restart once if its controlling process disappears.
Do not retry reported training/evaluation errors automatically.
"""
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import psutil

root = Path(__file__).resolve().parents[1]
os.chdir(root)
scripts = {"reproduction/pipeline.py", "reproduction/evaluate.py", "tina/post_train_hf/grpo.py"}
while True:
    active = False
    for proc in psutil.process_iter(["cmdline", "status"]):
        try:
            args = proc.info["cmdline"] or []
            if proc.info["status"] != psutil.STATUS_ZOMBIE and any(arg in scripts for arg in args):
                active = True
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    if not active:
        break
    time.sleep(15)
status = root / "artifacts/pipeline_status.json"
state = json.loads(status.read_text()) if status.exists() else {}
if state.get("status") in {"failed", "complete"} and state.get("stage") == "all":
    raise SystemExit(0)
if state.get("status") == "failed":
    raise SystemExit("Pipeline reported an error; inspect stage logs before resuming")
with (root / "logs/pipeline.log").open("a") as log:
    result = subprocess.run([sys.executable, "-u", "reproduction/pipeline.py"], stdout=log, stderr=subprocess.STDOUT)
raise SystemExit(result.returncode)
