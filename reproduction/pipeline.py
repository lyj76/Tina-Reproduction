"""Resumeable local training/evaluation pipeline with durable stage status."""
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
os.chdir(root)
lock = (root / "logs/pipeline.lock").open("w")
try:
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    raise SystemExit("A reproduction pipeline is already running")
status_file = root / "artifacts/pipeline_status.json"
status_file.parent.mkdir(parents=True, exist_ok=True)
state = json.loads(status_file.read_text()) if status_file.exists() else {"completed": []}
state["pid"] = os.getpid()
(root / "logs/pipeline.pid").write_text(str(os.getpid()) + "\n")
model = root / "artifacts/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B"

def status(stage, result, **extra):
    state.update(stage=stage, status=result, updated_utc=datetime.now(timezone.utc).isoformat(), **extra)
    temp = status_file.with_suffix(".tmp")
    temp.write_text(json.dumps(state, indent=2))
    temp.replace(status_file)
    print(f"{state['updated_utc']} {stage}: {result}", flush=True)

stages = [
    ("smoke", ["bash", "reproduction/train.sh", "smoke"]),
    ("audit_smoke", [sys.executable, "reproduction/audit_adapter.py", str(model / "smoke_open_rs3")]),
    ("baseline_math500", [sys.executable, "reproduction/evaluate.py", "--output", "artifacts/eval/base.jsonl"]),
    ("main", ["bash", "reproduction/train.sh", "main"]),
    ("audit_main", [sys.executable, "reproduction/audit_adapter.py", str(model / "grpo_open_rs3")]),
    ("checkpoint450_math500", [sys.executable, "reproduction/evaluate.py", "--adapter", str(model / "grpo_open_rs3/checkpoint-450"), "--output", "artifacts/eval/checkpoint450.jsonl"]),
    ("checkpoint850_math500", [sys.executable, "reproduction/evaluate.py", "--adapter", str(model / "grpo_open_rs3/checkpoint-850"), "--output", "artifacts/eval/checkpoint850.jsonl"]),
]
for name, command in stages:
    # This archived pipeline preserves its historical evaluation definition.
    # For official scoring/merged-weight replication, use recheck_queue.py.
    if "math500" in name:
        command += ["--scorer", "legacy-math-verify"]
    if name in state["completed"]:
        continue
    # Recover completed work if a controller exits between child exit and status write.
    if name == "main":
        saved_state = model / "grpo_open_rs3/trainer_state.json"
        if saved_state.exists() and (model / "grpo_open_rs3/adapter_model.safetensors").exists():
            if json.loads(saved_state.read_text()).get("global_step", 0) >= 850:
                state["completed"].append(name)
                status(name, "complete")
                continue
    if "math500" in name:
        summary_path = Path(command[command.index("--output") + 1]).with_suffix(".summary.json")
        expected_adapter = command[command.index("--adapter") + 1] if "--adapter" in command else None
        if summary_path.exists():
            summary = json.loads(summary_path.read_text())
            if (summary.get("samples") == 500 and summary.get("seed") == 42
                    and summary.get("max_tokens") == 32768 and summary.get("adapter") == expected_adapter):
                state["completed"].append(name)
                status(name, "complete")
                continue
    status(name, "running")
    env = os.environ.copy()
    if "math500" in name:
        env["CUDA_VISIBLE_DEVICES"] = "0"
    with (root / f"logs/{name}.log").open("a") as output:
        result = subprocess.run(command, env=env, stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        status(name, "failed", returncode=result.returncode)
        raise SystemExit(result.returncode)
    state["completed"].append(name)
    status(name, "complete")
status("all", "complete")
