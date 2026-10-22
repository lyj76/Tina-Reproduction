"""Chinese read-only dashboard for the format-reward mechanism experiments."""
from datetime import datetime, timezone
from pathlib import Path
import shutil

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text

from reproduction.core_tui import ROOT, TASKS, read_json, tail, parse_progress, gpu_rows, duration, metric

STATE = ROOT / "artifacts/mechanism"
LABELS = {"official_format":"作者格式配方", "label_free_format":"去除答案依赖"}
STAGES = {"preflight":"检查奖励与实验配置", "training":"独立训练", "auditing_training":"核验训练与权重",
    "merging":"在CPU合并权重", "evaluating":"六项基准评测", "final_audit":"最终逐题核验",
    "complete":"机制实验全部完成", "failed":"队列异常"}


def build(console, show_raw=False):
    compact = console.height < 40
    state = read_json(STATE / "status.json"); stage = state.get("status", "unknown")
    experiment = state.get("experiment", "official_format")
    pid = state.get("pid"); alive = bool(pid and Path(f"/proc/{pid}").exists())
    color = "green" if stage=="complete" else "red" if stage=="failed" else "cyan"
    header = Table.grid(expand=True);header.add_column(ratio=2);header.add_column(justify="right")
    header.add_row(Text("TINA · 核心机制实验",style="bold cyan"), Text(datetime.now().astimezone().strftime("%m-%d %H:%M:%S %Z")))
    if not compact: header.add_row(Text(STAGES.get(stage,"启动中"),style=color),Text(f"后台进程 {pid or '—'}",style="dim"))
    panels = [Panel(header,border_style=color,box=box.ROUNDED)]
    logpath = ROOT / state.get("logfile", "logs/mechanism_queue.log")
    log = tail(logpath); workers, log_step, _ = parse_progress(log)
    metrics = read_json(STATE / experiment / "latest_metrics.json")
    resource = state.get("latest_training_resource",{})
    step = max(metrics.get("step",0),resource.get("step",0),log_step)
    completed = 0; scores = {}
    for name in LABELS:
        for checkpoint in (500,850):
            key=f"{name}_{checkpoint}";scores[key]={}
            for task in TASKS:
                path=STATE / name / f"eval{checkpoint}" / task.replace(":","_") / "summary.json"
                value=read_json(path).get("results",{}).get(f"custom:{task}:0",{}).get("extractive_match")
                if isinstance(value,(int,float)):scores[key][task]=value;completed+=1
    current = Table.grid(padding=(0,1),expand=True);current.add_column(width=12,style="dim");current.add_column(ratio=1)
    current.add_row("当前阶段",Text(f"{STAGES.get(stage,'启动中')} · {LABELS.get(experiment,experiment)}",style="bold"))
    if stage=="training":
        current.add_row(f"训练 {step}/850",ProgressBar(total=850,completed=min(step,850),complete_style="magenta"))
        elapsed=resource.get("elapsed_seconds");rate=elapsed/max(resource.get("step",0),1) if elapsed else None
        if console.height >= 30:
            current.add_row("时间估计",Text(f"已用 {duration(elapsed)} · 剩余约 {duration((850-step)*rate) if rate else '首步后估算'}",style="dim"))
        current.add_row("训练指标",Text(f"奖励 {metric(metrics.get('reward'))}  格式 {metric(metrics.get('rewards/format_reward'))}  长度 {metric(metrics.get('completion_length'),5)}  KL {metric(metrics.get('kl'))}"))
    if stage=="evaluating":
        task=state.get("task");expected=TASKS.get(task,("",0))[1]
        done=min(sum(w["done"] for w in workers.values()),expected)
        current.add_row("当前评测",Text(f"{state.get('checkpoint')}步 · {TASKS.get(task,(task,0))[0]} · {done}/{expected}题"))
        current.add_row("生成进度",ProgressBar(total=max(expected,1),completed=done,complete_style="cyan"))
    current.add_row("新增评测",Text(f"{completed}/24 项完成 · 两组独立训练，各测500/850步",style="green"))
    if not compact:
        trained = sum(read_json(STATE / name / "training_audit.json").get("status")=="passed" for name in LABELS)
        current.add_row("训练完成",Text(f"{trained}/2 组 · 学习率调度均保持1500步"))
    if stage=="failed":current.add_row("异常原因",Text(state.get("error","查看日志"),style="bold red"))
    elif stage not in ("complete","unknown") and not alive:
        current.add_row("进程状态",Text("队列进程已退出，请检查日志",style="bold red"))
    else:
        try:
            age=(datetime.now(timezone.utc)-datetime.fromisoformat(state["updated_utc"])).total_seconds()
            if age>95 and stage!="complete":current.add_row("心跳提醒",Text("队列状态较久未更新",style="yellow"))
        except (ValueError,KeyError):pass
    panels.append(Panel(current,title="任务与进度",border_style="blue"))
    gpu = Table(box=None,expand=True,padding=(0,1))
    for label in ("显卡","利用率","显存 / 总量","温度","功耗"):gpu.add_column(label,justify="left" if label=="显卡" else "right",no_wrap=True)
    rows,error=gpu_rows()
    for index,name,util,used,total,temp,power,limit in rows:
        gpu.add_row(f"{index} · {name.replace('NVIDIA GeForce ','')}",f"{float(util):.0f}%",
            f"{float(used)/1024:.1f}/{float(total)/1024:.1f} GiB",f"{temp}℃",f"{float(power):.0f}/{float(limit):.0f} W")
    if error:gpu.add_row("读取失败","—","—","—","—")
    free=shutil.disk_usage(ROOT).free/2**30
    panels.append(Panel(Group(gpu,Text(f"磁盘剩余 {free:.1f} GiB · 每组2份断点 + 500步权重 · 临时合并权重用后清理",style="yellow" if free<4 else "dim")),title="显卡与存储",border_style="green"))
    matrix=Table(box=None,expand=True,padding=(0,1));matrix.add_column("实验 / 步数",no_wrap=True)
    for title,_ in TASKS.values():matrix.add_column(title,justify="right",no_wrap=True)
    matrix.add_column("均值",justify="right",no_wrap=True)
    for name,label in LABELS.items():
        for checkpoint in (500,850):
            data=scores[f"{name}_{checkpoint}"];cells=[]
            for task in TASKS:
                active=stage=="evaluating" and experiment==name and checkpoint==state.get("checkpoint") and task==state.get("task")
                cells.append(f"{data[task]:.1%}" if task in data else "运行中" if active else "·")
            average=f"{sum(data.values())/6:.2%}" if len(data)==6 else "待完成"
            matrix.add_row(f"{label} {checkpoint}",*cells,Text(average,style="cyan"))
    reference=Text("既有六项均值：基座45.55% · RS3 500步49.46% · RS3 850步47.69%",style="dim")
    if console.width>=76:panels.append(Panel(Group(matrix,reference),title="六项成绩 · 固定检查点 · 单次采样",border_style="blue"))
    if console.height>=43 or show_raw:
        lines=[line.strip() for line in log.replace("\r","\n").splitlines() if line.strip()]
        text="\n".join(lines[-3:]) if show_raw else "作者配方的长度奖励含正确性；对照组只去掉这一依赖。\n两组奖励构成相同，不将该对照称为作者原始配方。\n日志："+str(logpath.relative_to(ROOT))
        panels.append(Panel(Text(text,style="dim",overflow="ellipsis"),title="原始日志" if show_raw else "实验说明",border_style="dim"))
    panels.append(Text("Q 退出观察  R 刷新  L 日志 · 每2秒更新 · SSH断开后后台继续运行",style="dim",justify="center"))
    return Group(*panels)
