"""Run the original trainer, optionally removing correctness from its length reward.

The clean variant changes only len_reward; the upstream experiment stays intact.
Local callbacks persist metrics and an adapter-only step-500 snapshot, without RNG use.
"""
import json
import os
from pathlib import Path
import runpy
import shutil


def label_free_length_reward(completions, **kwargs):
    """Upstream length normalization with the correctness-dependent clamp removed."""
    lengths = [len(c[0]["content"]) for c in completions]
    low, high = min(lengths), max(lengths)
    return [0.0] * len(lengths) if low == high else [0.5-(n-low)/(high-low) for n in lengths]


def main():
    from transformers import TrainerCallback
    from tina.post_train_hf import rewards
    from tina.post_train_hf.grpo_trainer import GRPOTrainer

    root = Path(os.environ["TINA_MECHANISM_DIR"])
    root.mkdir(parents=True, exist_ok=True)
    clean = os.environ.get("TINA_LABEL_FREE_LENGTH") == "1"
    if clean:
        # grpo.py imports this function only after the replacement below.
        label_free_length_reward.__name__ = "len_reward"
        rewards.len_reward = label_free_length_reward

    class Snapshot500(TrainerCallback):
        def on_save(self, args, state, control, **kwargs):
            if not state.is_world_process_zero or state.global_step != 500:
                return control
            source = Path(args.output_dir) / "checkpoint-500"
            target = root / "adapter500"
            target.mkdir(exist_ok=True)
            for name in ("adapter_model.safetensors", "adapter_config.json"):
                shutil.copy2(source / name, target / name)
            (target / "snapshot.json").write_text(json.dumps({"step":500, "source":str(source),
                "scope":"Adapter only; original tokenizer is supplied by the pinned base."}, indent=2))
            return control

    original_init = GRPOTrainer.__init__
    original_log = GRPOTrainer.log

    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self.add_callback(Snapshot500())
        if self.is_world_process_zero():
            (root / "training_semantics.json").write_text(json.dumps({
                "label_free_length":clean, "training_seed":self.args.seed,
                "rewards":[f.__name__ for f in self.reward_funcs],
                "reward_weights":self.reward_weights.detach().cpu().tolist(),
                "scheduler_steps":self.args.max_steps, "stop_step":850,
                "scale_rewards":self.args.scale_rewards,
                "original_length_uses_solution":not clean}, indent=2))

    def log(self, logs, *args, **kwargs):
        # Save the same metrics original_log will log, before it clears _metrics.
        if self.is_world_process_zero():
            row = {**logs, **{k:sum(v)/len(v) for k,v in self._metrics.items() if v},
                "step":self.state.global_step}
            temp = root / "latest_metrics.tmp"
            temp.write_text(json.dumps(row)); temp.replace(root / "latest_metrics.json")
            with (root / "metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(row)+"\n")
        return original_log(self, logs, *args, **kwargs)

    GRPOTrainer.__init__ = initialize
    GRPOTrainer.log = log
    runpy.run_module("tina.post_train_hf.grpo", run_name="__main__")


if __name__ == "__main__":
    main()
