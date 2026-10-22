"""Read-only Chinese terminal dashboard for the detached TINA experiments."""
import argparse
import ast
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import re
import select
import shutil
import subprocess
import sys
import termios
import time
import tty

from rich import box
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.table import Table
from rich.text import Text


ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "artifacts/core_suite"
TRAIN = ROOT / "artifacts/ckpts/models/DeepSeek-R1-Distill-Qwen-1.5B/grpo_open_rs2_core"
TASKS = {"aime24": ("AIME24", 30), "aime25": ("AIME25", 30), "amc23": ("AMC23", 40),
         "math_500": ("MATH500", 500), "gpqa:diamond": ("GPQA", 198), "minerva": ("Minerva", 272)}
MODELS = {"base": "基座模型", "local850": "本地850步", "local500": "本地500步",
          "official500": "作者500步", "rs2_450": "第二配方450步"}
STAGES = {"waiting_for_existing_aime25": "接续已有评测", "evaluating": "基准评测", "training_rs2": "训练第二配方",
          "auditing_existing_six_benchmark_results": "核验六项评测", "auditing_rs2": "核验训练权重",
          "merging_rs2_on_cpu": "在CPU合并权重", "final_audit": "最终逐题核验", "complete": "全部完成", "failed": "异常退出"}
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
PROMPTS = re.compile(r"Processed prompts:.*?(\d+)/(\d+)\s*\[([^\]]*)\]")
STEPS = re.compile(r"(?:^|\s)(\d+)/1500\s*\[([^\]]*)\]")


def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}


def tail(path, size=160_000):
    try:
        with path.open("rb") as handle:
            handle.seek(0, 2)
            length = handle.tell()
            handle.seek(max(0, length-size))
            return ANSI.sub("", handle.read().decode("utf-8", errors="replace"))
    except OSError:
        return ""


def result_folder(name, task):
    if task in ("aime24", "amc23", "math_500") and name != "rs2_450":
        return ROOT / (f"artifacts/endpoint850/{task}" if name == "local850" else f"artifacts/recheck/official_pipeline_{name}/{task}")
    return STATE / name / task.replace(":", "_")


def duration(seconds):
    if seconds is None or seconds < 0:
        return "—"
    seconds = int(seconds)
    hours, rest = divmod(seconds, 3600)
    minutes, seconds = divmod(rest, 60)
    return f"{hours}时{minutes:02}分" if hours else f"{minutes}分{seconds:02}秒"


def elapsed_text(text):
    try:
        out = 0
        for part in text.split(":"):
            out = out*60 + int(part)
        return out
    except ValueError:
        return None


def parse_progress(log):
    workers, worker, step, latest_metrics = {}, "单进程", 0, {}
    for line in log.replace("\r", "\n").splitlines():
        pid = re.search(r"pid=(\d+)", line)
        if pid:
            worker = pid.group(1)
        found = PROMPTS.search(line)
        if found:
            done, total = int(found[1]), int(found[2])
            remaining = re.search(r"<([^,]+)", found[3])
            workers[worker] = {"done": done, "total": total,
                               "remaining": elapsed_text(remaining[1]) if remaining else None}
        found = STEPS.search(line)
        if found:
            step = int(found[1])
        if "{'loss':" in line:
            try:
                latest_metrics = ast.literal_eval(line[line.index("{'loss':"):])
            except (ValueError, SyntaxError):
                pass
    return workers, step, latest_metrics


def gpu_rows():
    try:
        output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit",
            "--format=csv,noheader,nounits"], text=True, timeout=5)
        return [[field.strip() for field in row] for row in csv.reader(io.StringIO(output))], None
    except (OSError, subprocess.SubprocessError) as exc:
        return [], str(exc)


def metric(value, precision=4):
    return f"{value:.{precision}g}" if isinstance(value, (float, int)) else "—"


def build(console, show_raw=False, suite="auto"):
    if suite == "mechanism" or (suite == "auto" and (ROOT / "artifacts/mechanism/status.json").exists()):
        from reproduction.mechanism_tui import build as mechanism_build
        return mechanism_build(console, show_raw)
    compact = console.height < 38
    state = read_json(STATE / "status.json")
    stage = state.get("status", "unknown")
    running = stage not in ("complete", "failed", "unknown")
    pid = state.get("pid")
    alive = bool(pid and Path(f"/proc/{pid}").exists())
    color = "green" if stage == "complete" else "cyan" if alive and running else "red" if stage == "failed" else "yellow"
    try:
        age = (datetime.now(timezone.utc)-datetime.fromisoformat(state["updated_utc"])).total_seconds()
    except (KeyError, ValueError):
        age = None
    header = Table.grid(expand=True)
    header.add_column(ratio=2); header.add_column(justify="right")
    header.add_row(Text("TINA 论文复现 · 实验控制台", style="bold cyan"), Text(datetime.now().astimezone().strftime("%m-%d %H:%M:%S %Z")))
    if not compact:
        header.add_row(Text(STAGES.get(stage, "等待队列状态"), style=f"bold {color}"), Text(f"队列进程 {pid or '—'} · 心跳 {duration(age)}前", style="dim"))
    panels = [Panel(header, border_style=color, box=box.ROUNDED)]

    scores = {}; completed = 0
    for name in MODELS:
        scores[name] = {}
        for task in TASKS:
            summary = read_json(result_folder(name, task) / "summary.json")
            result = summary.get("results", {}).get(f"custom:{task}:0", {})
            if isinstance(result.get("extractive_match"), (float, int)):
                scores[name][task] = result["extractive_match"]; completed += 1
    logpath = ROOT / state.get("logfile", "logs/core_queue.log")
    log = tail(logpath)
    workers, step, metrics = parse_progress(log)
    resource = state.get("latest_training_resource", {})
    if stage == "training_rs2":
        step = max(step, resource.get("step", 0))
    elif (TRAIN / "trainer_state.json").exists():
        step = read_json(TRAIN / "trainer_state.json").get("global_step", step)
    else:
        step = resource.get("step", 0)
    current = Table.grid(padding=(0, 1), expand=True)
    current.add_column(style="dim", width=12); current.add_column(ratio=3)
    model = MODELS.get(state.get("model"), state.get("model", "—"))
    task = TASKS.get(state.get("task"), (state.get("task", "—"), 0))[0]
    detail = f"{model} / {task}" if stage == "evaluating" else "Open-RS2：格式奖励 + 答案正确性奖励" if stage == "training_rs2" else STAGES.get(stage, "等待状态文件")
    current.add_row("当前任务", Text(detail, style="bold"))
    if stage == "evaluating":
        expected = TASKS.get(state.get("task"), ("", 0))[1]
        done = min(sum(w["done"] for w in workers.values()), expected)
        if workers:
            bar = ProgressBar(total=max(expected, 1), completed=done, width=None, complete_style="cyan", finished_style="green")
            progress = Table.grid(expand=True)
            progress.add_column(ratio=1); progress.add_column(justify="right", width=20)
            progress.add_row(bar, Text(f"{done}/{expected}题 {done/max(expected,1):.0%}"))
            current.add_row("日志进度", progress)
            shard = "  |  ".join(f"分片{i+1} {w['done']}/{w['total']}" for i, w in enumerate(workers.values()))
            remaining = [w["remaining"] for w in workers.values() if w["remaining"] is not None]
            if not compact:
                current.add_row("分片 / 预计", Text(shard+ (f"  ·  剩余约 {duration(max(remaining))}" if remaining else ""), style="dim"))
        else:
            phase = "正在编译推理图" if "Capturing cuda" in log or "Capturing CUDA" in log else "正在加载模型 / 准备数据"
            current.add_row("任务进度", Text(phase, style="yellow"))
    current.add_row("累计评测", Text(f"{completed}/30 项完成（含已有结果） · {STAGES.get(stage,'等待状态')}" if compact else f"{completed}/30 项完成（含复用的已有结果）", style="green"))
    if stage == "training_rs2":
        rate = resource.get("elapsed_seconds", 0)/max(resource.get("step", 0), 1)
        current.add_row(f"训练 {step}/450步", ProgressBar(total=450, completed=min(step,450), width=None, complete_style="magenta"))
        current.add_row("训练估计", Text(f"已用 {duration(resource.get('elapsed_seconds'))} · 剩余约 {duration((450-step)*rate) if rate else '估算中'} · 调度器保持1500步", style="dim"))
        current.add_row("训练指标", Text(f"奖励 {metric(metrics.get('reward'))}  |  KL {metric(metrics.get('kl'))}  |  梯度范数 {metric(metrics.get('grad_norm'))}"))
    elif step and not compact:
        current.add_row("第二配方训练", Text(f"{step}/450 步"))
    elif not compact:
        current.add_row("后续训练", Text("六项评测补齐后 → 独立训练Open-RS2至450步", style="dim"))
    if stage == "failed":
        current.add_row("异常原因", Text(state.get("error", "查看队列日志"), style="bold red"))
    elif running and not alive:
        current.add_row("进程检查", Text("队列进程已退出，请检查日志；子进程可能仍在运行", style="bold red"))
    elif age is not None and age > 95 and running:
        current.add_row("心跳提醒", Text("状态更新较慢，以上显卡信息仍为实时采样", style="yellow"))
    panels.append(Panel(current, title="[bold]任务与进度[/bold]", border_style="blue"))

    gpus, gpu_error = gpu_rows()
    table = Table(box=None if compact else box.SIMPLE, expand=True, padding=(0,1))
    for name, justify in [("显卡","left"),("利用率","right"),("显存 / 总量","right"),("温度","right"),("功耗 / 上限","right")]:
        table.add_column(name, justify=justify, no_wrap=True)
    for row in gpus:
        try:
            index, name, util, used, total, temp, power, limit = row
            u = float(util)
            style = "green" if u >= 60 else "yellow" if u > 0 else "dim"
            table.add_row(f"{index} · {name.replace('NVIDIA GeForce ', '')}", Text(f"{u:.0f}%", style=style),
                f"{float(used)/1024:.1f} / {float(total)/1024:.1f} GiB", f"{temp}℃", f"{float(power):.0f} / {float(limit):.0f} W")
        except ValueError:
            table.add_row(row[0], "读取中", "—", "—", "—")
    if gpu_error:
        table.add_row("暂时无法读取显卡", "—", "—", "—", "—")
    disk = shutil.disk_usage(ROOT)
    disk_label = f"磁盘余量 {disk.free/2**30:.1f} GiB · 新训练保留3份断点 · 原17份保留" if compact else f"磁盘剩余 {disk.free/2**30:.1f} GiB / {disk.total/2**30:.0f} GiB  ·  新训练滚动保留3份断点  ·  原17份断点保留"
    disk_text = Text(disk_label, style="yellow" if disk.free < 6*2**30 else "dim")
    panels.append(Panel(Group(table, disk_text), title="[bold]显卡与存储[/bold]", border_style="green"))

    matrix = Table(box=None if compact else box.SIMPLE_HEAD, expand=True, padding=(0,1))
    matrix.add_column("模型", no_wrap=True)
    for title, _ in TASKS.values(): matrix.add_column(title, justify="right", no_wrap=True)
    matrix.add_column("六项均值", justify="right", no_wrap=True)
    for name, label in MODELS.items():
        cells = []
        for task in TASKS:
            if task in scores[name]: cells.append(Text(f"{scores[name][task]*100:.1f}%", style="green"))
            elif stage == "evaluating" and state.get("model")==name and state.get("task")==task: cells.append(Text("运行中", style="bold yellow"))
            else: cells.append(Text("·", style="dim"))
        avg = Text(f"{sum(scores[name].values())/6*100:.2f}%", style="bold cyan") if len(scores[name])==6 else Text("待完成", style="dim")
        matrix.add_row(label,*cells,avg)
    if console.width >= 76:
        panels.append(Panel(matrix, title="[bold]六项基准成绩 · 官方协议 · 单次采样[/bold]", border_style="blue"))

    if console.height >= 44 or show_raw:
        lines = [line.strip() for line in log.replace("\r", "\n").splitlines() if line.strip()]
        if show_raw:
            text = Text("\n".join(lines[-3:]), style="dim", overflow="ellipsis")
        else:
            notes = []
            if stage == "evaluating":
                notes.append(f"正在评测 {model} 的 {task}；双卡分片各自生成答案。")
                if workers: notes.append("进度取自分片日志，可能比实时生成略有延迟。")
            elif stage == "training_rs2": notes.append("仅显示新配方的450步训练目标；原始学习率调度保持1500步。")
            elif stage == "complete": notes.append("全部任务已结束，结果与逐题核验已保存。")
            elif stage == "failed": notes.append("队列异常已记录。可按 L 查看最近原始日志。")
            notes.append(f"日志：{logpath.relative_to(ROOT)}")
            text = Text("\n".join(notes), style="dim")
        panels.append(Panel(text, title="原始日志（L切换）" if show_raw else "运行摘要", border_style="dim"))
    footer = "Q 退出  R 刷新  L 日志 · 每2秒更新 · 实验后台运行" if compact else "Q 退出观察  ·  R 立即刷新  ·  L 原始日志  ·  每2秒刷新  ·  退出界面后实验继续在后台运行"
    panels.append(Text(footer, style="dim", justify="center"))
    return Group(*panels)


def main():
    parser = argparse.ArgumentParser(description="TINA中文实验终端看板")
    parser.add_argument("--once", action="store_true", help="输出一屏后退出")
    parser.add_argument("--width", type=int, help="指定显示宽度")
    parser.add_argument("--height", type=int, help="指定布局高度")
    parser.add_argument("--no-color", action="store_true", help="使用无彩色显示")
    parser.add_argument("--suite", choices=("auto", "core", "mechanism"), default="auto", help="选择观察当前机制实验或此前核心评测")
    args = parser.parse_args()
    console = Console(width=args.width, height=args.height, highlight=False, no_color=args.no_color)
    if args.once or not (sys.stdin.isatty() and sys.stdout.isatty()):
        console.print(build(console, suite=args.suite)); return
    original = termios.tcgetattr(sys.stdin.fileno())
    show_raw = False
    try:
        tty.setcbreak(sys.stdin.fileno())
        with Live(console=console, screen=True, auto_refresh=False, redirect_stdout=False, redirect_stderr=False) as live:
            refresh = 0
            while True:
                if time.monotonic() >= refresh:
                    live.update(build(console, show_raw, suite=args.suite), refresh=True)
                    refresh = time.monotonic()+2
                ready,_,_ = select.select([sys.stdin], [], [], .15)
                if ready:
                    key = os.read(sys.stdin.fileno(), 1).decode(errors="ignore").lower()
                    if key in ("q", "\x03", "\x04"): break
                    if key=="l": show_raw = not show_raw; refresh = 0
                    if key=="r": refresh = 0
    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, original)


if __name__=="__main__": main()
