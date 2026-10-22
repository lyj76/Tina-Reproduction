"""Run the original trainer with passive first-step rollout/RNG tracing."""
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import runpy

import peft
from safetensors.torch import save_file
import torch

from tina.post_train_hf.grpo_trainer import GRPOTrainer
from trl.data_utils import maybe_apply_chat_template


ROOT = Path(os.environ["TINA_TRACE_DIR"])
ROOT.mkdir(parents=True, exist_ok=True)
RANK = int(os.environ.get("RANK", "0"))


def sha(tensor):
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def rng_state(device=None):
    state = {"cpu": sha(torch.get_rng_state())}
    if device is not None:
        state["cuda"] = sha(torch.cuda.get_rng_state(device))
    return state


original_peft = peft.get_peft_model


def traced_peft(*args, **kwargs):
    before = rng_state()
    model = original_peft(*args, **kwargs)
    if RANK == 0:
        initial_a = {name: param.detach().cpu().contiguous() for name, param in model.named_parameters()
                     if "lora_A" in name}
        save_file(initial_a, ROOT / "initial_lora_A.safetensors")
        metadata = {"before_initialization_rng": before, "after_initialization_rng": rng_state(),
            "lora_B_all_zero": all(not torch.count_nonzero(param).item() for name, param in model.named_parameters() if "lora_B" in name),
            "initial_A_parameters": sum(x.numel() for x in initial_a.values()),
            "initial_A_dtype": sorted({str(x.dtype) for x in initial_a.values()})}
        (ROOT / "initialization.json").write_text(json.dumps(metadata, indent=2))
    return model


peft.get_peft_model = traced_peft
original_prepare = GRPOTrainer._prepare_inputs


def traced_prepare(self, inputs):
    before = rng_state(self.accelerator.device)
    prompts = [maybe_apply_chat_template(row, self.processing_class)["prompt"] for row in inputs]
    result = original_prepare(self, inputs)
    after = rng_state(self.accelerator.device)
    row = {"global_step_before_update": self.state.global_step, "rank": RANK,
        "rng_before_prepare": before, "rng_after_prepare": after,
        "metrics": {name: values[-1] for name, values in self._metrics.items() if values},
        "samples": []}
    for i, (source, prompt) in enumerate(zip(inputs, prompts)):
        keep = result["completion_mask"][i].bool()
        tokens = result["completion_ids"][i][keep].tolist()
        row["samples"].append({"problem": source["problem"], "solution": source["solution"],
            "vllm_prompt": prompt, "vllm_prompt_ids": self.processing_class.encode(prompt, add_special_tokens=False),
            "forward_prompt_ids": result["prompt_ids"][i][result["prompt_mask"][i].bool()].tolist(),
            "completion_ids": tokens, "completion": self.processing_class.decode(tokens, skip_special_tokens=True),
            "advantage": result["advantages"][i].item()})
    with (ROOT / f"rollouts_rank{RANK}.jsonl").open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return result


GRPOTrainer._prepare_inputs = traced_prepare
if RANK == 0:
    packages = ["torch", "transformers", "peft", "accelerate", "vllm", "trl", "xgrammar",
                "tokenizers", "datasets", "huggingface-hub", "deepspeed", "math-verify"]
    (ROOT / "environment.json").write_text(json.dumps({
        "packages": {name: importlib.metadata.version(name) for name in packages},
        "seed": 42, "vllm_engine_seed": 0, "vllm_sampling_seed": None,
        "stop_step": int(os.environ["TINA_STOP_STEP"]), "max_steps_scheduler": 1500,
        "fixed_callback_disabled": os.environ.get("TINA_DISABLE_FIXED_EVAL") == "1",
        "scope": "Passive trace: hashes, serialization, and token decoding; RNG states are never set or restored by tracing."}, indent=2))

runpy.run_module("tina.post_train_hf.grpo", run_name="__main__")
