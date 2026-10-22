"""Detached six-benchmark evaluations, bounded independentRS2 training, then evaluation."""
import argparse
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

from reproduction.core_results import ROOT, MODELS, TASKS, folder, aggregate


EVAL = str(Path("artifacts/recheck/eval_env").resolve()/"bin/python")
TRAIN = "/root/autodl-tmp/envs/tina/bin/python"
RS2 = Path("artifacts/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/grpo_open_rs2_core")
CURRENT = {}


def status(state, **extra):
    CURRENT.clear(); CURRENT.update(status=state,**extra)
    heartbeat()
    print(json.dumps(CURRENT),flush=True)


def heartbeat():
    record={**CURRENT,"pid":os.getpid(),"updated_utc":datetime.now(timezone.utc).isoformat()}
    try:
        record["gpu_snapshot"]=subprocess.check_output(["nvidia-smi","--query-gpu=index,utilization.gpu,memory.used","--format=csv,noheader,nounits"],text=True,timeout=10).strip()
    except subprocess.SubprocessError: pass
    record["disk_free_gib"]=round(shutil.disk_usage(ROOT).free/2**30,2)
    resources=RS2/"resources_rank0.jsonl"
    if resources.exists():
        lines=resources.read_text().splitlines()
        if lines:
            try: record["latest_training_resource"]=json.loads(lines[-1])
            except json.JSONDecodeError: pass
    temp=ROOT/"status.tmp";temp.write_text(json.dumps(record,indent=2));temp.replace(ROOT/"status.json")


def run(command, logfile, env=None):
    with open(logfile,"w") as handle:
        process=subprocess.Popen(command,env=env,stdout=handle,stderr=subprocess.STDOUT)
        CURRENT["child_pid"]=process.pid;CURRENT["logfile"]=str(logfile)
        while True:
            heartbeat()
            try: code=process.wait(timeout=30);break
            except subprocess.TimeoutExpired: pass
    CURRENT.pop("child_pid",None)
    if code: raise RuntimeError(f"Command failed with exit{code}; see {logfile}")


def wait_gpu():
    deadline=time.monotonic()+180
    while time.monotonic()<deadline:
        used=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used","--format=csv,noheader,nounits"],text=True)
        if all(int(x.strip())<1024 for x in used.splitlines()):return
        time.sleep(5)
    raise RuntimeError("Previous GPU processes did not release memory")


def evaluate(name, task):
    out=folder(name,task)
    if (out/"summary.json").exists():return
    wait_gpu();status("evaluating",model=name,task=task)
    command=[EVAL,"-u","reproduction/recheck_official_eval.py","--model",MODELS[name],"--tasks",task,"--output",str(out),"--seed","1234"]
    if task=="gpqa:diamond":
        command += ["--gpqa-csv","artifacts/core_suite/gpqa_diamond_public.csv","--prompt-seed","1234"]
    run(command,f"logs/core_{name}_{task.replace(':','_')}.log")


def main():
    parser=argparse.ArgumentParser();parser.add_argument("--wait-pid",type=int);args=parser.parse_args()
    ROOT.mkdir(exist_ok=True)
    lock=(ROOT/"queue.lock").open("w");fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    plan={"declared_utc":datetime.now(timezone.utc).isoformat(),"phases":["six_benchmark_existing_models","independent_open_rs2_450","six_benchmark_rs2"],
        "models":MODELS,"tasks":TASKS,"engine_seed":1234,"gpqa_prompt_seed":1234,"training_seed":42,
        "rs2_stop_step":450,"rs2_scheduler_steps":1500,"rs2_checkpoint_limit":3,
        "rs2_config_sha256":hashlib.sha256(Path("reproduction/open_rs2_core.yaml").read_bytes()).hexdigest()}
    if not (ROOT/"plan.json").exists():(ROOT/"plan.json").write_text(json.dumps(plan,indent=2))
    if args.wait_pid:
        status("waiting_for_existing_aime25",external_pid=args.wait_pid)
        while True:
            cmd=Path(f"/proc/{args.wait_pid}/cmdline")
            try: active=b"core_suite/base/aime25" in cmd.read_bytes()
            except FileNotFoundError:active=False
            if not active:break
            heartbeat();time.sleep(15)
    for name in ["base","local850","local500","official500"]:
        for task in TASKS:evaluate(name,task)
    status("auditing_existing_six_benchmark_results")
    aggregate()
    wait_gpu()
    if shutil.disk_usage(ROOT).free<6*2**30:raise RuntimeError("Need6GiB free before RS2 training")
    state=RS2/"trainer_state.json"
    completed=state.exists() and json.loads(state.read_text())["global_step"]==450
    if not completed:
        status("training_rs2",target_step=450)
        env=os.environ.copy();env.update(TINA_RUN_NAME="grpo_open_rs2_core",TINA_STOP_STEP="450",TINA_DISABLE_FIXED_EVAL="1")
        run([TRAIN,"-m","accelerate.commands.launch","--config_file","recipes/accelerate_ds_cfgs/ds_zero2.yaml",
            "--num_processes","2","--main_process_port","29521","tina/post_train_hf/grpo.py",
            "--config","reproduction/open_rs2_core.yaml","--cosine_max_len","3584"],"logs/core_rs2_train.log",env)
    state=json.loads((RS2/"trainer_state.json").read_text());assert state["global_step"]==450
    rows=[r for r in state["log_history"] if "reward" in r]
    assert len(rows)==450 and all(math.isfinite(v) for r in rows for v in r.values() if isinstance(v,(float,int)))
    assert len(list(RS2.glob("checkpoint-*")))<=3
    status("auditing_rs2")
    run([TRAIN,"reproduction/audit_adapter.py",str(RS2)],"logs/core_rs2_adapter_audit.log")
    (ROOT/"rs2_training_audit.json").write_text(json.dumps({"status":"passed","steps":450,"finite_metric_steps":len(rows),
        "checkpoints_retained":len(list(RS2.glob('checkpoint-*'))),"training_config":plan["rs2_config_sha256"]},indent=2))
    wait_gpu()
    merged=Path(MODELS["rs2_450"])
    if not (merged/"merge_audit.json").exists():
        if shutil.disk_usage(ROOT).free<4*2**30:raise RuntimeError("Need4GiB free for RS2 merge")
        status("merging_rs2_on_cpu")
        run([EVAL,"reproduction/recheck_merge.py","--adapter",str(RS2/"checkpoint-450"),"--output",str(merged)],"logs/core_rs2_merge.log")
    for task in TASKS:evaluate("rs2_450",task)
    status("final_audit");result=aggregate();assert result["status"]=="complete"
    status("complete",results=str(ROOT/"results.json"))


if __name__=="__main__":
    try: main()
    except Exception as error:
        status("failed",error=repr(error));raise
