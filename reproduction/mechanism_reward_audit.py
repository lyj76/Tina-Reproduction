"""Counterfactual check: changing gold labels must not change format-only rewards."""
import hashlib
import inspect
import json
from pathlib import Path

from tina.post_train_hf import rewards
from reproduction.mechanism_train import label_free_length_reward


def main():
    root = Path("artifacts/mechanism"); root.mkdir(exist_ok=True)
    texts = ["First, compute.\n</think>\n\\boxed{2}",
        "First, " + "reason carefully. "*20 + "\n</think>\n\\boxed{2}"]
    completions = [[{"role":"assistant", "content":s}] for s in texts]
    right, wrong = ["$2$"]*2, ["$3$"]*2
    common = [rewards.format_reward, rewards.tag_count_reward, rewards.reasoning_steps_reward,
        rewards.get_repetition_penalty_reward(ngram_size=3, max_penalty=-1.0)]
    original = [rewards.len_reward(completions=completions, solution=s) for s in (right, wrong)]
    clean = [label_free_length_reward(completions=completions, solution=s) for s in (right, wrong)]
    assert original == [[0.5, -0.5], [0.0, -0.5]], original
    assert clean == [[0.5, -0.5], [0.5, -0.5]], clean
    assert label_free_length_reward([completions[0], completions[0]]) == [0.0, 0.0]
    for function in common:
        assert function(completions=completions, solution=right) == function(completions=completions, solution=wrong)
        assert "solution" not in inspect.getsource(function)
    proof = {"status":"passed", "fixed_completion_texts":texts,
        "gold_original":right, "gold_counterfactual":wrong,
        "official_length_rewards":original, "label_free_length_rewards":clean,
        "official_is_label_free":False, "clean_reward_set_label_invariant":True,
        "conclusion":"The published format-only recipe includes correctness supervision through len_reward. This contradicts a strict interpretation of removing all accuracy-based reward; it does not establish that the whole method is invalid.",
        "files_sha256":{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            Path("tina/post_train_hf/rewards.py"),
            Path("recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/train_model_open_rs3_format_ablation.yaml"),
            Path("reproduction/mechanism_train.py")]}}
    (root / "reward_semantics_audit.json").write_text(json.dumps(proof, indent=2)+"\n")
    print(json.dumps(proof, indent=2))


if __name__ == "__main__":
    main()
