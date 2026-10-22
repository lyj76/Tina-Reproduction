"""Measure BF16 merging effects; these are diagnostics, not accuracy estimates."""
import gc
import json
from pathlib import Path

from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


torch.set_num_threads(8)
root = Path("artifacts/recheck")
base = "artifacts/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base"
adapter = "artifacts/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/grpo_open_rs3/checkpoint-500"
tokenizer = AutoTokenizer.from_pretrained(base)
model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16,
                                            device_map="cpu", attn_implementation="sdpa")
model = PeftModel.from_pretrained(model, adapter).eval()
rows = [json.loads(x) for x in Path("artifacts/eval/base.jsonl").read_text().splitlines()]
inputs = []
for index in [0, 5, 25, 60]:
    text = tokenizer.apply_chat_template([{"role": "user", "content": rows[index]["problem"]}],
                                         tokenize=False, add_generation_prompt=True)
    ids = tokenizer.encode(text, add_special_tokens=False)
    ids += tokenizer.encode(rows[index]["prediction"], add_special_tokens=False)[:64]
    inputs.append(torch.tensor([ids]))


@torch.inference_mode()
def logits():
    return [model(input_ids=ids, logits_to_keep=1, use_cache=False).logits[0, -1].float() for ids in inputs]


unmerged = logits()
samples = []
for name, module in model.named_modules():
    if hasattr(module, "lora_A") and "default" in module.lora_A:
        # Evaluate the first eight complete rows of each adapted matrix, in FP32.
        a = module.lora_A["default"].weight.detach().float()
        b = module.lora_B["default"].weight.detach()[:8].float()
        delta = b @ a * module.scaling["default"]
        weight = module.base_layer.weight.detach()[:8].float()
        rounded = (weight + delta).to(torch.bfloat16).float()
        roundtrip = (rounded - delta).to(torch.bfloat16).float()
        samples.append({"module": name, "elements": weight.numel(),
                        "merge_changes": (rounded != weight).sum().item(),
                        "merge_unmerge_changes": (roundtrip != weight).sum().item(),
                        "delta_l2": delta.square().sum().sqrt().item()})
model = model.merge_and_unload(safe_merge=True).eval()
merged = logits()
comparisons = []
for index, x, y in zip([0, 5, 25, 60], unmerged, merged):
    p, q = x.softmax(-1), y.softmax(-1)
    comparisons.append({"index": index, "prompt_plus_prefix_tokens": inputs[len(comparisons)].numel(),
                        "mean_absolute_logit_difference": (x - y).abs().mean().item(),
                        "max_absolute_logit_difference": (x - y).abs().max().item(),
                        "next_token_argmax_same": x.argmax().item() == y.argmax().item(),
                        "next_token_distribution_l1": (p - q).abs().sum().item()})
total = sum(x["elements"] for x in samples)
report = {"adapter": adapter, "device": "CPU", "attention": "sdpa", "rows_per_module": 8,
          "sampled_weight_elements": total,
          "fraction_changed_by_bf16_merge": sum(x["merge_changes"] for x in samples) / total,
          "fraction_changed_after_single_merge_unmerge": sum(x["merge_unmerge_changes"] for x in samples) / total,
          "logit_comparisons": comparisons, "sampled_modules": samples,
          "caveat": "Teacher-forced CPU diagnostic; not a vLLM equivalence test or evidence of an accuracy effect. The official trainer also merges/unmerges in BF16."}
(root / "merge_numeric_audit.json").write_text(json.dumps(report, indent=2))
print(json.dumps({k: v for k, v in report.items() if k != "sampled_modules"}, indent=2), flush=True)
