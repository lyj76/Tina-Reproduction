"""Detached, disk-bounded queue for the two format-reward mechanism experiments."""
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import time

from reproduction.mechanism_results import ROOT, EXPERIMENTS, TASKS, folder, aggregate

TRAIN = "/root/autodl-tmp/envs/tina/bin/python"
EVAL = str(Path("artifacts/recheck/eval_env").resolve() / "bin/python")
MODEL_ROOT = Path("artifacts/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B")
MERGED = ROOT / "temporary_merged"
CONFIG = Path("reproduction/format_mechanism.yaml")
CURRENT = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def heartbeat():
    row = {**CURRENT, "pid":os.getpid(), "updated_utc":datetime.now(timezone.utc).isoformat(),
        "disk_free_gib":round(shutil.disk_usage(ROOT).free/2**30,2)}
    path = Path(CURRENT.get("training_dir", "missing")) / "resources_rank0.jsonl"
    if path.exists():
        lines = path.read_text().splitlines()
        if lines:
            try: row["latest_training_resource"] = json.loads(lines[-1])
            except json.JSONDecodeError: pass
    temp = ROOT / "status.tmp"; temp.write_text(json.dumps(row,indent=2)); temp.replace(ROOT / "status.json")


def status(stage, **extra):
    CURRENT.clear(); CURRENT.update(status=stage, **extra); heartbeat()
    print(json.dumps(CURRENT), flush=True)


def run(command, logfile, env=None):
    with Path(logfile).open("a") as output:
        output.write("\nCOMMAND " + json.dumps(command)+"\n"); output.flush()
        process = subprocess.Popen(command, env=env, stdout=output, stderr=subprocess.STDOUT)
        CURRENT.update(child_pid=process.pid, logfile=str(logfile))
        while True:
            heartbeat()
            try: code=process.wait(timeout=20); break
            except subprocess.TimeoutExpired: pass
    CURRENT.pop("child_pid", None)
    if code: raise RuntimeError(f"Command exited {code}; see {logfile}")


def wait_gpu():
    deadline = time.monotonic()+180
    while time.monotonic()<deadline:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"], text=True)
        if all(int(v.strip())<1024 for v in output.splitlines()): return
        heartbeat(); time.sleep(5)
    raise RuntimeError("GPUs occupied; refusing to overlap training/inference jobs")


def discard_merged():
    if MERGED.exists():
        # This exact, newly created cache is the only directory this queue removes.
        assert MERGED == Path("artifacts/mechanism/temporary_merged")
        assert (ROOT / "plan.json").exists()
        shutil.rmtree(MERGED)


def main():
    ROOT.mkdir(exist_ok=True)
    lock=(ROOT / "queue.lock").open("w"); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    plan_path=ROOT / "plan.json"
    files=[CONFIG, Path("reproduction/mechanism_train.py"), Path("tina/post_train_hf/grpo.py"),
        Path("tina/post_train_hf/grpo_trainer.py"), Path("tina/post_train_hf/rewards.py")]
    hashes={str(p):sha(p) for p in files}
    if plan_path.exists():
        plan=json.loads(plan_path.read_text()); assert plan["files_sha256"]==hashes, "Training semantics changed after declaration"
    else:
        plan={"declared_utc":datetime.now(timezone.utc).isoformat(), "experiments":list(EXPERIMENTS),
            "train_stop_step":850, "scheduler_steps":1500, "training_seed":42,
            "fixed_evaluation_steps":[500,850], "tasks":TASKS, "inference_seed":1234,
            "checkpoint_limit_per_run":2, "pin_adapter_at_step":500,
            "temporary_merge_cache":str(MERGED), "files_sha256":hashes,
            "reason":"Official format-only length reward depends on correctness; compare against label-free normalization with all other rewards fixed."}
        plan_path.write_text(json.dumps(plan,indent=2)+"\n")
    status("preflight")
    run([TRAIN,"-m","reproduction.mechanism_reward_audit"],"logs/mechanism_reward_audit.log")
    aggregate()
    for experiment in EXPERIMENTS:
        model=MODEL_ROOT / f"grpo_{experiment}_core"
        context={"experiment":experiment, "training_dir":str(model), "target_step":850}
        state_path=model / "trainer_state.json"
        done=state_path.exists() and json.loads(state_path.read_text())["global_step"]==850
        if not done:
            wait_gpu()
            if shutil.disk_usage(ROOT).free<3*2**30: raise RuntimeError("Need at least3GiB for bounded training checkpoints")
            status("training", **context)
            env=os.environ.copy();env.update(TINA_RUN_NAME=model.name, TINA_STOP_STEP="850",
                TINA_DISABLE_FIXED_EVAL="1", TINA_MECHANISM_DIR=str(ROOT / experiment),
                TINA_LABEL_FREE_LENGTH="1" if experiment=="label_free_format" else "0")
            run([TRAIN,"-m","accelerate.commands.launch","--config_file","recipes/accelerate_ds_cfgs/ds_zero2.yaml",
                "--num_processes","2","--main_process_port","29531","reproduction/mechanism_train.py",
                "--config",str(CONFIG),"--cosine_max_len","3584"], f"logs/mechanism_{experiment}_train.log",env)
        status("auditing_training", **context)
        state=json.loads(state_path.read_text());assert state["global_step"]==850
        metrics=[r for r in state["log_history"] if "loss" in r]
        assert len(metrics)==850 and {r["step"] for r in metrics}==set(range(1,851))
        assert all(math.isfinite(v) for r in metrics for v in r.values() if isinstance(v,(int,float)))
        assert len(list(model.glob("checkpoint-*")))<=2
        semantics=json.loads((ROOT / experiment / "training_semantics.json").read_text())
        assert semantics["label_free_length"] == (experiment=="label_free_format")
        assert semantics["rewards"]==["format_reward","tag_count_reward","len_reward","reasoning_steps_reward","repetition_penalty_reward"]
        assert semantics["reward_weights"]==[1.0]*5 and semantics["scale_rewards"] is True
        for adapter in (model, ROOT / experiment / "adapter500"):
            run([TRAIN,"reproduction/audit_adapter.py",str(adapter)], f"logs/mechanism_{experiment}_adapter_audit.log")
        (ROOT / experiment / "training_audit.json").write_text(json.dumps({"status":"passed", "steps":850,
            "finite_metric_steps":len(metrics), "runtime_hours":json.loads((model / "train_results.json").read_text())["train_runtime"]/3600,
            "adapter500_sha256":sha(ROOT / experiment / "adapter500/adapter_model.safetensors"),
            "adapter850_sha256":sha(model / "adapter_model.safetensors"), "semantics":semantics},indent=2)+"\n")
        run(["/root/miniconda3/bin/python","-m","reproduction.mechanism_plots"],"logs/mechanism_plots.log")
        for step in (500,850):
            if all((folder(experiment,step,t) / "summary.json").exists() for t in TASKS): continue
            wait_gpu(); discard_merged()
            if shutil.disk_usage(ROOT).free<4*2**30: raise RuntimeError("Need4GiB for temporary BF16 merge")
            adapter=ROOT / experiment / "adapter500" if step==500 else model
            status("merging", **context, checkpoint=step)
            run([EVAL,"reproduction/recheck_merge.py","--adapter",str(adapter),"--output",str(MERGED)],
                f"logs/mechanism_{experiment}_{step}_merge.log")
            merge=json.loads((MERGED / "merge_audit.json").read_text())
            (ROOT / experiment / f"merge{step}_audit.json").write_text(json.dumps(merge,indent=2)+"\n")
            for task in TASKS:
                out=folder(experiment,step,task)
                if (out / "summary.json").exists(): continue
                wait_gpu(); status("evaluating", **context, checkpoint=step, task=task)
                command=[EVAL,"-u","reproduction/recheck_official_eval.py","--model",str(MERGED),"--tasks",task,
                    "--output",str(out),"--seed","1234"]
                if task=="gpqa:diamond": command += ["--gpqa-csv","artifacts/core_suite/gpqa_diamond_public.csv","--prompt-seed","1234"]
                run(command, f"logs/mechanism_{experiment}_{step}_{task.replace(':','_')}.log")
                aggregate()
            wait_gpu(); aggregate(); discard_merged()
    status("final_audit")
    results=aggregate(); assert results["status"]=="complete" and results["predictions_verified"]==4280
    (ROOT / "final_audit.json").write_text(json.dumps({"status":"passed", "evaluations":24,
        "predictions_verified":4280, "input_parity_verified":True,
        "results_sha256":sha(ROOT / "results.json"), "plan":plan,
        "training_audits":[json.loads((ROOT / exp / "training_audit.json").read_text()) for exp in EXPERIMENTS]},indent=2)+"\n")
    status("complete", results=str(ROOT / "results.json"))


if __name__=="__main__":
    try: main()
    except Exception as error:
        status("failed", error=repr(error)); raise
