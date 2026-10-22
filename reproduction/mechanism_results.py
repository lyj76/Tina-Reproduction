"""Audit format-reward ablations against the existing, fixed six-task inputs."""
import ast
from datetime import datetime, timezone
import json
from pathlib import Path

import pyarrow.parquet as pq

from reproduction.core_results import TASKS, folder as core_folder

ROOT = Path("artifacts/mechanism")
EXPERIMENTS = {"official_format":"作者格式配方", "label_free_format":"去除答案依赖"}


def folder(experiment, step, task):
    return ROOT / experiment / f"eval{step}" / task.replace(":", "_")


def value(data):
    return ast.literal_eval(data) if isinstance(data, str) else data


def rows(path):
    files = list((path / "details").glob("**/*.parquet"))
    assert len(files) == 1, path
    return pq.read_table(files[0], columns=["example", "full_prompt", "gold", "predictions", "metrics",
        "specifics", "num_asked_few_shots", "num_effective_few_shots"]).to_pylist()


def inputs(data):
    return [(r["example"], r["full_prompt"], value(r["gold"])) for r in data]


def aggregate(write=True):
    references = {task:inputs(rows(core_folder("base", task))) for task in TASKS}
    models, pending, verified = {}, [], 0
    for experiment in EXPERIMENTS:
        for step in (500, 850):
            tasks = {}
            for task, expected in TASKS.items():
                path = folder(experiment, step, task)
                if not (path / "summary.json").exists():
                    pending.append(f"{experiment}/{step}/{task}"); continue
                summary = json.loads((path / "summary.json").read_text())
                manifest = summary["manifest"]
                assert manifest["engine_seed"] == 1234 and manifest["sampling_seed"] is None
                assert manifest["requests"] == expected and manifest["data_parallel_size"] == 2
                assert manifest["temperature"] == 0.6 and manifest["top_p"] == 0.95
                assert manifest["max_new_tokens"] == manifest["max_model_length"] == 32768
                assert manifest["dtype"] == "bfloat16" and manifest["use_chat_template"] is True
                if task == "gpqa:diamond": assert manifest["prompt_seed"] == 1234
                data = rows(path)
                assert len(data) == expected and inputs(data) == references[task]
                assert len({r["full_prompt"] for r in data}) == expected
                exported = []
                for index, row in enumerate(data):
                    assert row["num_asked_few_shots"] == row["num_effective_few_shots"] == 0
                    prediction, metric = value(row["predictions"]), value(row["metrics"])
                    assert len(prediction) == 1 and prediction[0] and metric["extractive_match"] in (0., 1.)
                    exported.append({"index":index, "example":row["example"], "full_prompt":row["full_prompt"],
                        "gold":value(row["gold"]), "prediction":prediction[0],
                        "correct":bool(metric["extractive_match"]), "extracted":value(row["specifics"])})
                correct = sum(r["correct"] for r in exported)
                accuracy = summary["results"][f"custom:{task}:0"]["extractive_match"]
                assert abs(correct / expected - accuracy) < 1e-12
                if write:
                    (path / "predictions.jsonl").write_text("".join(json.dumps(r, ensure_ascii=False)+"\n" for r in exported))
                tasks[task] = {"samples":expected, "correct":correct, "accuracy":accuracy, "source":str(path / "summary.json")}
                verified += expected
            models[f"{experiment}_{step}"] = {"tasks":tasks,
                "six_benchmark_mean":sum(r["accuracy"] for r in tasks.values())/6 if len(tasks)==6 else None}
    result = {"updated_utc":datetime.now(timezone.utc).isoformat(), "status":"complete" if not pending else "partial",
        "models":models, "pending":pending, "predictions_verified":verified,
        "input_parity_verified":True, "single_inference_seed":1234,
        "scope":"Two independently initialized training runs, same seed42; fixed checkpoints500/850, not chosen by test scores. Clean run changes only the length reward's correctness dependence."}
    if write:
        (ROOT / "results.json").write_text(json.dumps(result, indent=2)+"\n")
        write_report(result)
    return result


def write_report(result):
    core = json.loads(Path("artifacts/core_suite/results.json").read_text())
    lines = ["# TINA 核心机制：格式奖励与正确性信号", "",
        "检验论文第6.2节关于格式学习、准确性奖励和训练转折的解释。原RS3/RS2及全部既有结果保留。", "",
        "## 已确认的实现问题", "",
        "作者的format-only配置包含format、tag_count、length、reasoning_steps、repetition_penalty五项等权奖励。length调用len_reward，其中解析参考答案并核验正误：短答案正确时可得正奖励，错误时该正奖励被截为0。因此公开配方并非严格无正确性监督。",
        "", "固定同一输出，只把参考答案从2改为3，官方length奖励从[0.5,-0.5]变为[0,-0.5]；去除答案依赖的版本始终为[0.5,-0.5]。其余四项奖励均不读取参考答案。证据：[reward_semantics_audit.json](artifacts/mechanism/reward_semantics_audit.json)。",
        "", "来源：[作者配置](https://github.com/shangshang-wang/Tina/blob/106376c7c3388a30ae7769087276534c47cf0b0c/recipes/DeepSeek-R1-Distill-Qwen-1.5B/grpo/train_model_open_rs3_format_ablation.yaml)、[作者len_reward实现](https://github.com/shangshang-wang/Tina/blob/106376c7c3388a30ae7769087276534c47cf0b0c/tina/post_train_hf/rewards.py#L101)、[论文第6.2节及表20](https://proceedings.iclr.cc/paper_files/paper/2026/file/91be140e1524bb21531508c2a6a85c5e-Paper-Conference.pdf#page=9)。这是公开消融实现与文字解释的冲突，不代表整篇方法无效；作者原始run是否用过未公开改动尚不可确认。",
        "", "## 预先固定的实验", "",
        "两组均从同一基座独立初始化、训练seed42、相同数据/LoRA/优化器/采样/1500步学习率调度，850步停止。第一组复现作者的五项奖励；第二组仅把length改成不读取答案的0.5−归一化字符长度，其余四项及权重不变。第二组是用于机制核验的本地对照，不称为作者原始实验。",
        "", "固定500/850步各做六项评测，复用已核验的基座和RS3对照；不按测试分数事后挑步数。只能评估该长度奖励中正确性依赖的影响，无法由这两组结果单独证明所有格式奖励或LoRA的普遍机制。所有成绩仍为单训练seed、单推理seed的结果。",
        "", "每组滚动保留2份完整断点，另存500步adapter；850步保存最终adapter。合并权重是新实验专用临时缓存，每轮评测和审计完成后删除，可从保留adapter重建；逐题输出、manifest、校验和保留。原17份RS3、3份RS2及原合并权重均保留。",
        "", "## 六项评测", "",
        "| 模型 | AIME24 | AIME25 | AMC23 | MATH500 | GPQA | Minerva | 六项均值 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|"]
    all_rows = [("基座",core["models"]["base"]), ("RS3 500步",core["models"]["local500"]),
        ("RS3 850步",core["models"]["local850"])]
    all_rows += [(f"{EXPERIMENTS[exp]} {step}步", result["models"][f"{exp}_{step}"]) for exp in EXPERIMENTS for step in (500,850)]
    for label, data in all_rows:
        scores = [f"{data['tasks'][t]['accuracy']:.2%}" if t in data["tasks"] else "待完成" for t in TASKS]
        mean = f"{data['six_benchmark_mean']:.2%}" if data["six_benchmark_mean"] is not None else "待完成"
        lines.append(f"| {label} | "+" | ".join(scores)+f" | {mean} |")
    lines += ["", "论文原format-only：500步均值43.50%，850步50.56%（表20）。这两项是作者报告值，不能当作本机测量。",
        "", f"新增评测已核验{result['predictions_verified']}条输出，待完成{len(result['pending'])}项。状态：[status.json](artifacts/mechanism/status.json)；结果：[results.json](artifacts/mechanism/results.json)。",
        "", "观察命令：`bash /root/autodl-tmp/Tina/reproduction/watch_core.sh`。队列依次完成训练、保存、合并、评测、逐题核验并绘制训练曲线。"]
    if result["status"] == "complete":
        base = core["models"]["base"]["six_benchmark_mean"]
        conclusions = ["## 最终判断（2026-10-21复核）", "",
            "两组各850步训练、24项评测及4280条输出均已完成。原始Parquet、题目/提示词/参考答案顺序、分数汇总、权重有限性与训练文件哈希已复核，见 [完成后复核](artifacts/mechanism/completion_reaudit.json)。", "",
            "| 固定步数 | 作者格式配方 | 去除答案依赖 | 去除依赖后的差值（百分点） | 作者配方对基座 | 无答案依赖对基座 |",
            "|---|---:|---:|---:|---:|---:|"]
        for step in (500,850):
            official = result["models"][f"official_format_{step}"]["six_benchmark_mean"]
            clean = result["models"][f"label_free_format_{step}"]["six_benchmark_mean"]
            conclusions.append(f"| {step} | {official:.2%} | {clean:.2%} | {(clean-official)*100:+.2f} | {(official-base)*100:+.2f} | {(clean-base)*100:+.2f} |")
        conclusions += ["", "方法主线得到本次实验支持：原RS3 500步49.46%、RS2 450步47.53%、作者格式配方500步49.84%，均高于本地基座45.55%。这表明1.5B基座采用LoRA+GRPO可以在本轮六项平均分上取得收益，不代表每项或跨seed都提升。",
            "", "机制解释未被完整验证：作者公开format-only配方仍包含正确性监督；去掉长度奖励对答案的依赖后，500/850步均值分别降至46.09%/44.35%，收益明显缩小或转负。这一方向支持保留正确性信号的价值，不支持把原配方成绩视为纯格式学习充分性的证据；单seed不足以证明该信号普遍必要或唯一解释所有提升。",
            "", "原RS3后期格式奖励明显上升、输出长度下降，格式消融与它的变化不同。只能说明本次奖励设计伴随训练轨迹差异，不能由曲线直接证明“只学习格式、保持知识”。本实验没有通过独立知识保持测试证明知识不变。",
            "", "作者表20的格式配方850步50.56%，本机850步46.30%，且本机500步优于850步。论文最佳步数和幅度未完整复现。原RS3 500步总均值接近论文，并不等于整篇论文全部结论得到复现。", ""]
        lines[4:4] = conclusions
    Path("MECHANISM.md").write_text("\n".join(lines)+"\n")


if __name__ == "__main__":
    result = aggregate()
    print(json.dumps({"status":result["status"], "pending":len(result["pending"]), "predictions_verified":result["predictions_verified"]}))
