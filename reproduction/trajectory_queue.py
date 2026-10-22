"""Run two full-length three-step training controls in detached execution."""
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import time


ROOT = Path("artifacts/trajectory")


def status(state, **extra):
    data = {"status": state, "pid": os.getpid(), "updated_utc": datetime.now(timezone.utc).isoformat(), **extra}
    temp = ROOT / "status.tmp"
    temp.write_text(json.dumps(data, indent=2))
    temp.replace(ROOT / "status.json")
    print(json.dumps(data), flush=True)


def main():
    models = Path("artifacts/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B")
    conditions = [("current42", "/root/autodl-tmp/envs/tina/bin/python", "1"),
                  ("author42", str(ROOT.resolve() / "author_env/bin/python"), "0")]
    plan = {"declared_utc": datetime.now(timezone.utc).isoformat(), "training_seed": 42,
        "vllm_seed": 0, "sampling_seed": None, "conditions": [x[0] for x in conditions],
        "steps_each": 3, "scheduler_steps": 1500, "max_completion_length": 3584,
        "batch_per_device": 6, "world_size": 2, "gradient_accumulation": 4, "generations": 6,
        "vllm_memory_fraction": 0.25,
        "limitations": ["Same local4090 hardware, not author's hardware", "xgrammar0.1.15 unavailable;0.1.17 retained",
            "Only first3 steps, before callback first trigger50", "Matching early scalar metrics does not establish benchmark performance"]}
    if not (ROOT / "plan.json").exists():
        (ROOT / "plan.json").write_text(json.dumps(plan, indent=2))
    for name, python, disable in conditions:
        checkpoint = models / f"trajectory_{name}"
        if (checkpoint / "trainer_state.json").exists():
            state = json.loads((checkpoint / "trainer_state.json").read_text())
            if state["global_step"] == 3:
                continue
        status("running", condition=name, output=str(checkpoint))
        env = os.environ.copy()
        env.update({"TINA_RUN_NAME": f"trajectory_{name}", "TINA_STOP_STEP": "3",
            "TINA_TRACE_DIR": str(ROOT / name), "TINA_DISABLE_FIXED_EVAL": disable})
        command = [python, "-m", "accelerate.commands.launch", "--config_file", "recipes/accelerate_ds_cfgs/ds_zero2.yaml",
            "--num_processes", "2", "--main_process_port", "29511", "reproduction/trajectory_train.py",
            "--config", "reproduction/open_rs3.yaml", "--cosine_max_len", "3584", "--save_total_limit", "1"]
        with open(f"logs/trajectory_{name}.log", "w") as handle:
            subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)
        deadline = time.monotonic() + 120
        while time.monotonic() < deadline:
            used = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
            if all(int(value.strip()) < 1024 for value in used.splitlines()):
                break
            time.sleep(5)
        else:
            raise RuntimeError("Previous training workers did not release GPUs")
    status("aggregating")
    subprocess.run([conditions[0][1], "-m", "reproduction.trajectory_results"], check=True)
    status("complete", results=str(ROOT / "results.json"))


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        status("failed", error=repr(error))
        raise
