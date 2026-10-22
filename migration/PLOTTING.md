# 绘图环境补充

训练与正式评测使用迁移说明中的Python3.10环境；部分离线绘图使用旧机 `/root/miniconda3/bin/python`（Python3.12），其中安装了matplotlib/numpy。

该绘图解释器不属于 `/root/autodl-tmp/envs/tina/`。不必为了画图搬整个base conda环境：可在新机器单独准备一个包含matplotlib、numpy的分析环境，再运行绘图脚本。旧机的解释器及这两个库版本见本目录 `plotting_environment.json`。

直接重画机制曲线：

```bash
cd /root/autodl-tmp/Tina
/path/to/plotting/python -m reproduction.mechanism_plots
```

`reproduction/mechanism_queue.py` 中绘图命令写死了 `/root/miniconda3/bin/python`。若将来启动新的机制队列，需将这一运行命令改为新机器的绘图解释器，并把路径改动作为新运行记录；不要修改历史训练配置、manifest或权重。单独评测已有模型不需要这个绘图解释器。

所有本轮已经生成的PNG/SVG仍随项目和证据包保留，阅读报告无需恢复绘图环境。
