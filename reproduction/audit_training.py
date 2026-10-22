"""Re-run the completed 850-step training and checkpoint file audit on CPU."""
import gc
import hashlib
import json
import math
from pathlib import Path
import torch

root = Path(__file__).resolve().parents[1]
model = root / "artifacts/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/grpo_open_rs3"
state = json.loads((model / "trainer_state.json").read_text())
assert state["global_step"] == 850
rows = [row for row in state["log_history"] if "grad_norm" in row]
assert [row["step"] for row in rows] == list(range(1, 851))
assert all(math.isfinite(value) for row in rows for value in row.values() if isinstance(value, (float, int)))
assert all(row["grad_norm"] > 0 for row in rows)

checkpoints = []
required = ["adapter_model.safetensors", "adapter_config.json", "trainer_state.json",
            "rng_state_0.pth", "rng_state_1.pth", "scheduler.pt"]
for step in range(50, 851, 50):
    directory = model / f"checkpoint-{step}"
    for name in required:
        assert (directory / name).is_file(), str(directory / name)
    assert json.loads((directory / "trainer_state.json").read_text())["global_step"] == step
    assert len(list((directory / f"global_step{step}").glob("*optim_states.pt"))) == 2
    checkpoints.append({"step": step, "bytes": sum(f.stat().st_size for f in directory.rglob("*") if f.is_file())})

serialized = []
for step in [450, 850]:
    for path in sorted((model / f"checkpoint-{step}/global_step{step}").glob("*.pt")):
        data = torch.load(path, map_location="cpu", weights_only=False)
        assert isinstance(data, dict) and data
        if path.name == "mp_rank_00_model_states.pt":
            assert data["global_steps"] == step and data["dp_world_size"] == 2
        else:
            assert data.get("optimizer_state_dict")
        serialized.append({"step": step, "file": path.name, "keys": list(data)})
        del data
        gc.collect()
assert len(serialized) == 6

resources = {}
for path in model.glob("resources_rank*.jsonl"):
    entries = [json.loads(line) for line in path.read_text().splitlines()]
    assert entries[-1]["step"] == 850
    resources[path.stem] = {"last_step": entries[-1]["step"],
                            "peak_allocated_gib": max(r["peak_allocated_gib"] for r in entries),
                            "peak_reserved_gib": max(r["peak_reserved_gib"] for r in entries)}
assert {"resources_rank0", "resources_rank1"} <= set(resources)
digest = hashlib.sha256()
with (model.parent / "base/model.safetensors").open("rb") as source:
    while chunk := source.read(8 * 1024 * 1024):
        digest.update(chunk)
assert digest.hexdigest() == "58858233513d76b8703e72eed6ce16807b523328188e13329257fb9594462945"
result = {"global_step": 850, "step_metrics_count": len(rows), "all_metrics_finite": True,
          "all_gradient_norms_positive": True, "training_seconds": state["log_history"][-1]["train_runtime"],
          "checkpoint_count": len(checkpoints), "checkpoint_total_bytes": sum(c["bytes"] for c in checkpoints),
          "checkpoints": checkpoints, "resources": resources,
          "unchanged_base_sha256": digest.hexdigest(), "checkpoint_serialization_verified": serialized,
          "scope": "Training metrics and checkpoint files were checked; this audit does not run a resumed training step or evaluate reasoning accuracy."}
target = root / "artifacts/training_audit.json"
target.write_text(json.dumps(result, indent=2))
print(f"PASS: 850 steps, 17 checkpoints, 6 serialized state files, both GPU resource logs, unchanged base SHA-256.\n{target}")
