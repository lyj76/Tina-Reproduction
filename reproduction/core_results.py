"""Audit and summarize all six official tasks without selecting a best seed."""
import ast
from datetime import datetime, timezone
import json
from pathlib import Path

import pyarrow.parquet as pq


ROOT = Path("artifacts/core_suite")
TASKS = {"aime24":30, "aime25":30, "amc23":40, "math_500":500, "gpqa:diamond":198, "minerva":272}
MODELS = {"base":"artifacts/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/base",
    "local850":"artifacts/endpoint850/merged_local850", "local500":"artifacts/recheck/merged_local500",
    "official500":"artifacts/recheck/merged_official500", "rs2_450":"artifacts/core_suite/merged_rs2_450"}
LABELS = {"base":"基座", "local850":"本地 Open-RS3 850步", "local500":"本地 Open-RS3 500步",
    "official500":"作者 Open-RS3 500步权重", "rs2_450":"本地 Open-RS2 450步"}
# ICLR 2026 paper, Appendix B, Tables 12/13 (printed percentages).
PAPER = {"base": [23.33,16.67,62.50,82.60,31.82,30.15,41.18],
    "local500": [36.67,23.33,82.50,85.20,37.37,31.62,49.45],
    "local850": [26.67,26.67,75.00,83.80,31.82,27.94,45.32],
    "rs2_450": [43.33,26.67,77.50,87.00,36.36,32.72,50.60]}


def folder(name, task):
    if task in ("aime24", "amc23", "math_500") and name != "rs2_450":
        return Path(f"artifacts/endpoint850/{task}") if name == "local850" else Path(f"artifacts/recheck/official_pipeline_{name}/{task}")
    return ROOT / name / task.replace(":", "_")


def value(x):
    return ast.literal_eval(x) if isinstance(x,str) else x


def aggregate():
    output, references, pending = {}, {}, []
    for name in MODELS:
        scores = {}
        for task, count in TASKS.items():
            path = folder(name, task)
            if not (path / "summary.json").exists():
                pending.append(f"{name}/{task}"); continue
            summary = json.loads((path / "summary.json").read_text())
            manifest = summary["manifest"]
            assert manifest["engine_seed"] == 1234 and manifest["sampling_seed"] is None
            assert manifest["requests"] == count and manifest["data_parallel_size"] == 2
            assert manifest["temperature"] == 0.6 and manifest["top_p"] == 0.95
            assert manifest["max_new_tokens"] == manifest["max_model_length"] == 32768
            assert manifest["dtype"] == "bfloat16" and manifest["use_chat_template"] is True
            if task == "gpqa:diamond":
                assert manifest["prompt_seed"] == 1234
            files = list((path / "details").glob("**/*.parquet")); assert len(files)==1, path
            rows = pq.read_table(files[0], columns=["example","full_prompt","gold","predictions","metrics",
                "specifics","num_asked_few_shots","num_effective_few_shots"]).to_pylist()
            assert len(rows)==count and len({r["full_prompt"] for r in rows})==count
            inputs = [(r["example"], r["full_prompt"], value(r["gold"])) for r in rows]
            if task not in references: references[task]=inputs
            assert inputs==references[task], (name, task, "Input/choice order differs")
            exported=[]
            for i,r in enumerate(rows):
                assert r["num_asked_few_shots"]==r["num_effective_few_shots"]==0
                pred, metric = value(r["predictions"]), value(r["metrics"])
                assert len(pred)==1 and pred[0] and metric["extractive_match"] in (0.,1.)
                exported.append({"index":i,"example":r["example"],"full_prompt":r["full_prompt"],
                    "gold":value(r["gold"]),"prediction":pred[0],"correct":bool(metric["extractive_match"]),"extracted":value(r["specifics"])})
            correct=sum(r["correct"] for r in exported); score=summary["results"][f"custom:{task}:0"]["extractive_match"]
            assert abs(correct/count-score)<1e-12
            # Existing exported formats remain intact; new core-suite outputs get their own readable export.
            if ROOT in path.parents:
                (path/"predictions.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in exported))
            scores[task]={"samples":count,"correct":correct,"accuracy":score,"source":str(path/"summary.json")}
        output[name]={"tasks":scores,"six_benchmark_mean":sum(v["accuracy"] for v in scores.values())/6 if len(scores)==6 else None}
    result={"updated_utc":datetime.now(timezone.utc).isoformat(),"models":output,"pending":pending,
        "status":"complete" if not pending else "partial", "input_parity_verified":True,
        "predictions_verified":sum(v["samples"] for x in output.values() for v in x["tasks"].values()),
        "scope":"Single seed1234 per task; six-task mean is unweighted. GPQA uses byte-identical public CSV and predeclared prompt seed1234 to hold choice positions constant across models."}
    result["paper_reference"] = {"source":"https://proceedings.iclr.cc/paper_files/paper/2026/file/91be140e1524bb21531508c2a6a85c5e-Paper-Conference.pdf",
        "tables":[12,13], "task_order":list(TASKS), "values_percent":PAPER}
    baseline = output["base"]["six_benchmark_mean"]
    for name, data in output.items():
        mean = data["six_benchmark_mean"]
        data["delta_to_local_base_pp"] = (mean-baseline)*100 if mean is not None and baseline is not None else None
        if name in PAPER:
            data["delta_to_paper_mean_pp"] = mean*100-PAPER[name][-1] if mean is not None else None
    (ROOT/"results.json").write_text(json.dumps(result,indent=2))
    lines=["# TINA 六项基准与 Open-RS2 核心实验", "",
        f"状态：{result['status']}。已核验 {result['predictions_verified']} 条输出；五个模型题目、提示词、参考答案及顺序一致。下表为各任务准确率，六项均值为不加权平均。", "",
        "| 模型 | AIME24 | AIME25 | AMC23 | MATH500 | GPQA | Minerva | 六项均值 |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name, data in output.items():
        values=[f"{data['tasks'][t]['accuracy']:.2%}" if t in data['tasks'] else "待运行" for t in TASKS]
        avg=f"{data['six_benchmark_mean']:.2%}" if data['six_benchmark_mean'] is not None else "未完成"
        lines.append(f"| {LABELS[name]} | "+" | ".join(values)+f" | {avg} |")
    if not pending:
        lines += ["", "## 与论文对照", "",
            "| 模型 | 本地六项均值 | 论文六项均值 | 差值（百分点） | 本地相对基座增益（百分点） |",
            "|---|---:|---:|---:|---:|"]
        for name in ("base", "local500", "local850", "rs2_450"):
            data = output[name]
            lines.append(f"| {LABELS[name]} | {data['six_benchmark_mean']:.2%} | {PAPER[name][-1]:.2f}% | {data['delta_to_paper_mean_pp']:+.2f} | {data['delta_to_local_base_pp']:+.2f} |")
        lines += ["", "论文逐项准确率（与上表同序）：", "",
            "| 模型 | AIME24 | AIME25 | AMC23 | MATH500 | GPQA | Minerva | 六项均值 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for name, values in PAPER.items():
            lines.append(f"| {LABELS[name]} | "+" | ".join(f"{v:.2f}%" for v in values)+" |")
        lines += ["", "来源：[ICLR 2026 正式论文表12/13]("+result["paper_reference"]["source"]+")。",
            "", "Open-RS3 500步的六项均值接近论文，但逐项分数不同。本地基座45.55%也高于论文41.18%，因此本地增益+3.91个百分点没有复现论文约+8.27个百分点的增益。作者500步权重在同一本地协议下为46.57%，同样未得到论文49.45%。不能仅以均值接近判定精确复现。",
            "", "Open-RS2 450步均值47.53%，相对本地基座+1.98个百分点，低于论文50.60%约3.07个百分点。两种奖励配方的独立训练与六项评测已完成；仍只有一个训练seed，每任务本轮只采样一次。此前MATH-500三推理seed复核也未显示本地500步相对基座的稳定收益，见 [ATTRIBUTION.md](ATTRIBUTION.md)。"]
    lines += ["", "GPQA公开CSV与原始HF文件Git blob逐字节一致；选择题排列在Pipeline创建前固定seed1234，并核验所有模型提示词、参考答案及顺序一致。六项均值仅在全部完成后计算。", "",
        "Open-RS2：从原基座独立训练，format+2×accuracy，rank32/alpha128，训练seed42，保持1500步学习率调度器，450步停止。沿用本机验证过的vLLM显存0.25及关闭固定题生成回调的设置。每50步保存，滚动保留3份，仅作用于新Open-RS2目录。", "",
        "未完成任务："+str(len(pending))+"。单次采样不能证明稳定增益。", "",
        "状态：`artifacts/core_suite/status.json`；训练日志：`logs/core_rs2_train.log`；队列日志：`logs/core_queue.log`。观察：`bash reproduction/watch_core.sh`。"]
    Path("CORE_EXPERIMENTS.md").write_text("\n".join(lines)+"\n")
    return result


if __name__=="__main__":
    result=aggregate();print(json.dumps({"status":result["status"],"pending":len(result["pending"]),"predictions_verified":result["predictions_verified"]}))
