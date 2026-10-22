"""Recheck the completed core suite and save a CPU-only final audit."""
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from safetensors import safe_open
import torch

from core_results import aggregate, folder, MODELS, TASKS


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    results = aggregate()  # Re-reads all 30 Parquets and checks every prediction and input.
    assert results["status"] == "complete" and results["predictions_verified"] == 5350
    status = json.loads(Path("artifacts/core_suite/status.json").read_text())
    assert status["status"] == "complete"
    model_root = Path("artifacts/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B")
    rs2 = model_root / "grpo_open_rs2_core"
    state = json.loads((rs2 / "trainer_state.json").read_text())
    assert state["global_step"] == 450
    metrics = [r for r in state["log_history"] if "loss" in r]
    assert len(metrics) == 450 and {r["step"] for r in metrics} == set(range(1, 451))
    assert all(math.isfinite(v) for r in metrics for v in r.values() if isinstance(v, (int, float)))
    checkpoints = sorted(int(p.name.split("-")[1]) for p in rs2.glob("checkpoint-*") if p.is_dir())
    assert checkpoints == [350, 400, 450]
    original = sorted(int(p.name.split("-")[1]) for p in (model_root / "grpo_open_rs3").glob("checkpoint-*") if p.is_dir())
    assert original == list(range(50, 851, 50))
    adapter = rs2 / "adapter_model.safetensors"
    parameters = 0
    with safe_open(adapter, framework="pt", device="cpu") as weights:
        keys = list(weights.keys())
        assert len(keys) == 392
        for key in keys:
            tensor = weights.get_tensor(key)
            assert torch.isfinite(tensor).all().item(), key
            parameters += tensor.numel()
    assert parameters == 36929536
    assert digest(adapter) == digest(rs2 / "checkpoint-450/adapter_model.safetensors")
    provenance = json.loads(Path("artifacts/core_suite/gpqa_public_provenance.json").read_text())
    csv = Path("artifacts/core_suite/gpqa_diamond_public.csv").read_bytes()
    oid = hashlib.sha1(b"blob " + str(len(csv)).encode() + b"\0" + csv).hexdigest()
    assert hashlib.sha256(csv).hexdigest() == provenance["sha256"]
    assert oid == provenance["original_hf_file_metadata"]["oid"]
    hashes = {}
    for name in MODELS:
        for task in TASKS:
            path = folder(name, task)
            for artifact in [path / "summary.json", *path.glob("details/**/*.parquet")]:
                hashes[str(artifact)] = digest(artifact)
    training = json.loads((rs2 / "train_results.json").read_text())
    audit = {"status":"passed", "audited_utc":datetime.now(timezone.utc).isoformat(),
        "evaluations":30, "predictions_verified":5350, "input_prompt_gold_order_parity":True,
        "results_sha256":digest("artifacts/core_suite/results.json"),
        "rs2_training_steps":450, "rs2_finite_metric_steps":len(metrics),
        "rs2_training_hours":training["train_runtime"]/3600,
        "rs2_checkpoints_retained":checkpoints, "original_rs3_checkpoints_retained":original,
        "rs2_adapter_finite":True, "rs2_adapter_parameters":parameters,
        "rs2_adapter_sha256":digest(adapter), "gpqa_original_blob_oid":oid,
        "artifact_sha256":hashes,
        "limitations":"One training seed; one inference seed per task in this suite. Does not establish stable gains or exact paper replication."}
    Path("artifacts/core_suite/final_audit.json").write_text(json.dumps(audit, indent=2)+"\n")
    print(json.dumps({k:v for k,v in audit.items() if k != "artifact_sha256"}, indent=2))


if __name__ == "__main__":
    main()
