"""CPU audit of all training prompt token IDs under each dependency environment."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

from datasets import load_from_disk
import torch
from transformers import AutoTokenizer

from tina.post_train_hf.preprocess import make_conv_for_grpo
from tina.utils.chat_template import REASON_CHAT_TEMPLATE
from tina.utils.prompt import OPEN_RS_SYSTEM_PROMPT


parser = argparse.ArgumentParser()
parser.add_argument("--name", required=True)
args = parser.parse_args()
data = load_from_disk("artifacts/data/open-rs")
tokenizer = AutoTokenizer.from_pretrained("artifacts/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base")
tokenizer.pad_token = "<|fim_pad|>"
tokenizer.chat_template = REASON_CHAT_TEMPLATE
indices = torch.randperm(len(data), generator=torch.Generator().manual_seed(42)).tolist()[:24]
source_hash, tokens_hash = hashlib.sha256(), hashlib.sha256()
selected = {}
for index, row in enumerate(data):
    source_hash.update(json.dumps({k: row[k] for k in ["problem", "solution", "answer"]}, sort_keys=True, ensure_ascii=False).encode())
    prompt = make_conv_for_grpo(row, OPEN_RS_SYSTEM_PROMPT)["prompt"]
    text = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
    tokens = tokenizer.encode(text, add_special_tokens=False)
    tokens_hash.update(json.dumps(tokens, separators=(",", ":")).encode())
    if index in indices:
        selected[index] = {"dataset_index": index, "problem": row["problem"], "prompt": text, "tokens": tokens}
output = {"samples": len(data), "source_sha256": source_hash.hexdigest(), "token_ids_sha256": tokens_hash.hexdigest(),
    "first24_indices": indices, "first24": [selected[i] for i in indices],
    "versions": {k: importlib.metadata.version(k) for k in ["datasets", "tokenizers", "transformers"]}}
path = Path(f"artifacts/trajectory/{args.name}_inputs.json")
path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
print(json.dumps({k: v for k, v in output.items() if k != "first24"}, indent=2))
