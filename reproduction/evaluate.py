"""Matched MATH-500 evaluation; prompt and sampling follow the official scripts.

Uses vLLM directly, with the actual official LightEval task metric by default.
The legacy parser is opt-in for reproducing historical outputs. Dynamic LoRA
inference is a local comparison, not exact parity with BF16 merged evaluation.
"""
import argparse
import ast
import json
import os
from pathlib import Path
import time

from datasets import load_from_disk
from math_verify import ExprExtractionConfig, LatexExtractionConfig, parse, verify
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

parser = argparse.ArgumentParser()
parser.add_argument("--adapter")
parser.add_argument("--model", help="Full merged model path; prompt tokenizer remains the pinned base tokenizer")
parser.add_argument("--output", required=True)
parser.add_argument("--limit", type=int, default=500)
parser.add_argument("--max-tokens", type=int, default=32768)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--scorer", choices=["lighteval", "legacy-math-verify"], default="lighteval")
args = parser.parse_args()
if args.scorer == "lighteval":
    from reproduction.scoring import grade_math500, official_tasks
    official_tasks()  # Fail before allocating a GPU if the evaluation dependency is missing.
base = str(Path(os.environ["CKPT_DIR"]) / "models/DeepSeek-R1-Distill-Qwen-1.5B/base")

# Read the exact string constant without importing LightEval.
tree = ast.parse(Path("scripts/eval/run_eval_custom_tasks.py").read_text())
for node in tree.body:
    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "MATH_QUERY_TEMPLATE" for t in node.targets):
        template = ast.literal_eval(node.value.func.value).strip()
        break
else:
    raise RuntimeError("Official MATH prompt not found")

data = load_from_disk("artifacts/data/math500")
data = data.select(range(min(args.limit, len(data))))
tokenizer = AutoTokenizer.from_pretrained(base)
prompts = [tokenizer.apply_chat_template(
    [{"role": "user", "content": template.format(Question=row["problem"])}],
    tokenize=False, add_generation_prompt=True) for row in data]
out_path = Path(args.output)
out_path.parent.mkdir(parents=True, exist_ok=True)
model = LLM(model=args.model or base, dtype="bfloat16", tensor_parallel_size=1,
            gpu_memory_utilization=0.85, max_model_len=33792,
            enable_lora=bool(args.adapter), max_lora_rank=32,
            seed=args.seed, max_num_seqs=32)
params = SamplingParams(temperature=0.6, top_p=0.95, max_tokens=args.max_tokens, seed=args.seed)
request = LoRARequest("tina", 1, str(Path(args.adapter).resolve())) if args.adapter else None
started = time.time()
outputs = model.generate(prompts, params, lora_request=request)
rows = []
for i, (row, output) in enumerate(zip(data, outputs)):
    response = output.outputs[0]
    if args.scorer == "lighteval":
        correct = grade_math500(row["problem"], row["solution"], response.text)
    else:
        gold = parse(row["solution"], extraction_config=[LatexExtractionConfig()], extraction_mode="first_match")
        pred = parse(response.text, extraction_config=[ExprExtractionConfig(), LatexExtractionConfig(boxed_match_priority=0)], extraction_mode="first_match")
        correct = bool(verify(gold, pred))
    rows.append({"index": i, "unique_id": row.get("unique_id"), "problem": row["problem"],
                 "gold": row["solution"], "prediction": response.text, "correct": correct,
                 "tokens": len(response.token_ids), "finish_reason": response.finish_reason})
with out_path.open("w") as f:
    for row in rows:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")
summary = {"dataset": "HuggingFaceH4/MATH-500", "samples": len(rows),
           "correct": sum(r["correct"] for r in rows),
           "accuracy": sum(r["correct"] for r in rows) / len(rows),
           "truncated": sum(r["finish_reason"] == "length" for r in rows),
           "mean_tokens": sum(r["tokens"] for r in rows) / len(rows),
           "seconds": time.time() - started, "adapter": args.adapter,
           "max_tokens": args.max_tokens, "seed": args.seed,
           "temperature": 0.6, "top_p": 0.95, "backend": "vllm-direct",
           "scorer": args.scorer, "model": args.model or base,
           "tensor_parallel_size": 1, "max_num_seqs": 32, "max_model_len": 33792}
out_path.with_suffix(".summary.json").write_text(json.dumps(summary, indent=2))
print(json.dumps(summary, indent=2))
