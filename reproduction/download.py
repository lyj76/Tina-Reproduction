"""Fetch the one base model and training/evaluation datasets needed for reproduction."""
import json
from pathlib import Path
from huggingface_hub import HfApi, snapshot_download
import os

api = HfApi()
manifest_path = Path("artifacts/download_manifest.json")
previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
repo = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
revision = previous.get("base_model", {}).get("revision") or api.model_info(repo).sha
target = Path(os.environ["CKPT_DIR"]) / "models/DeepSeek-R1-Distill-Qwen-1.5B/base"
snapshot_download(repo, revision=revision, local_dir=target,
                  allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model"], max_workers=4)
manifest = {"base_model": {"repo": repo, "revision": revision, "path": str(target)}}
from datasets import load_dataset
for repo, split in [("knoveleng/open-rs", "train"), ("HuggingFaceH4/MATH-500", "test")]:
    revision = previous.get(repo, {}).get("revision") or api.dataset_info(repo).sha
    data = load_dataset(repo, split=split, revision=revision)
    manifest[repo] = {"revision": revision, "rows": len(data), "columns": data.column_names}
    if "open-rs" in repo:
        data.save_to_disk("artifacts/data/open-rs")
    else:
        data.save_to_disk("artifacts/data/math500")
manifest_path.write_text(json.dumps(manifest, indent=2))
print(json.dumps(manifest, indent=2), flush=True)
