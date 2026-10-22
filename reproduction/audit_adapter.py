import argparse
import json
from pathlib import Path
from safetensors.torch import load_file

parser = argparse.ArgumentParser()
parser.add_argument("adapter")
args = parser.parse_args()
path = Path(args.adapter)
config = json.loads((path / "adapter_config.json").read_text())
tensors = load_file(path / "adapter_model.safetensors")
assert config["r"] == 32
assert tensors and all("lora_" in key for key in tensors)
assert all(t.isfinite().all().item() for t in tensors.values()), "Adapter contains NaN or infinite weights"
b_tensors = [t.float() for key, t in tensors.items() if "lora_B" in key]
nonzero_b = sum(int(t.count_nonzero()) for t in b_tensors)
assert nonzero_b > 0, "LoRA B remains zero: training did not update the adapter"
summary = {"path": str(path), "rank": config["r"], "tensor_count": len(tensors), "all_weights_finite": True,
           "adapter_parameters": sum(t.numel() for t in tensors.values()),
           "nonzero_lora_b": nonzero_b,
           "lora_b_norm": sum(float(t.square().sum()) for t in b_tensors) ** 0.5}
(path / "adapter_audit.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
