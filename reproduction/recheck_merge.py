"""Save official-style BF16 merged models without using an occupied GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    base = "artifacts/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base"
    out = Path(args.output)
    if out.exists():
        raise RuntimeError(f"Refusing to overwrite {out}")
    torch.set_num_threads(8)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="cpu")
    model = PeftModel.from_pretrained(model, args.adapter)
    model = model.merge_and_unload(safe_merge=True)
    model.save_pretrained(out)
    AutoTokenizer.from_pretrained(base).save_pretrained(out)
    hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.glob("*.safetensors")}
    report = {"base": base, "adapter": args.adapter, "dtype": "bfloat16",
              "merge": "PeftModel.merge_and_unload(safe_merge=True)", "hashes": hashes}
    (out / "merge_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
