# TINA 迁移与恢复说明

## 推荐搬法

直接搬完整目录 `/root/autodl-tmp/Tina/`，约35GiB，包含代码、文档、训练数据、权重、优化器状态、逐题输出、图表与日志。**`artifacts/`、`logs/` 被Git忽略，git clone/git push不是实验备份。** 本地新增的 `reproduction/` 和文档也尚未提交，须随文件一起搬。

本机不额外生成一份35GiB压缩包，避免磁盘不足。建议从新机器拉取或直接rsync到新机器。下列命令是迁移模板，不会由助手自动连接未知目标。

```bash
# 在新机器执行，替换旧机器地址；保留原路径最省事。
mkdir -p /root/autodl-tmp/Tina
rsync -a --partial --info=progress2 \
  root@OLD_HOST:/root/autodl-tmp/Tina/ /root/autodl-tmp/Tina/
```

目标磁盘建议至少60GiB容纳当前实验和运行环境；若继续长期训练，建议100GiB以上。以下清单为本次准备迁移时的规模，不包含未来新增实验。

## 按用途选择迁移范围

| 用途 | 搬哪些 | 迁移后能做什么 |
|---|---|---|
| 接手阅读、分析、撰写 | `TINA_evidence.zip` | 文档、代码、配置、逐题输出、训练指标、图表、审计；不含大模型权重、优化器、Python环境 |
| 保留全部实验资产 | 整个Tina目录，约35GiB | 保存全部现有证据、基座、adapter和断点 |
| 继续运行、评测、训练 | 整个Tina目录 + 外部训练环境 + 数据缓存；建议也带编译wheel | 以现有环境/路径继续工作；在新机再做运行核验 |

轻量证据包位于项目根目录 `TINA_evidence.zip`，校验文件 `TINA_evidence.zip.sha256`。权重不在这个包中，不能用它替代完整模型迁移。

## 仓库外依赖

| 路径 | 约大小 | 作用 |
|---|---:|---|
| `/root/autodl-tmp/envs/tina/` | 7.5GiB | Python3.10.22的完整conda训练环境，含PyTorch/vLLM/FlashAttention/CUDA依赖 |
| `/root/autodl-tmp/hf-cache/` | 14MiB | HF数据缓存；训练数据也已单独保存于项目内 |
| `/root/autodl-tmp/vllm-0.7.2+cu118-cp38-abi3-manylinux1_x86_64.whl` | 189MiB | 精确vLLM CUDA11.8 wheel |
| `/root/autodl-tmp/flash_attn-2.7.3+cu11torch2.5cxx11abiFALSE-cp310-cp310-linux_x86_64.whl` | 185MiB | 精确FlashAttention wheel |
| `/root/autodl-tmp/wheels/` | 1.5GiB | 原PyTorch CUDA11.8等安装包，便于离线重建 |

训练环境是自带Python的conda环境，`sys.prefix` 与 `sys.base_prefix` 均为 `/root/autodl-tmp/envs/tina`。两个项目内venv使用 `--system-site-packages` 并链接这套训练环境：

- `artifacts/recheck/eval_env`：正式LightEval 0.8.1评测环境。
- `artifacts/trajectory/author_env`：历史作者依赖对照，仅用于追溯。
- `artifacts/trajectory/setup_python312`：历史错误环境，未用于最终主实验，无需为了最终结果恢复它。

若保持路径和Linux x86_64环境，可同时搬训练环境：

```bash
mkdir -p /root/autodl-tmp/envs/tina /root/autodl-tmp/hf-cache /root/autodl-tmp/wheels
rsync -a --partial --info=progress2 \
  root@OLD_HOST:/root/autodl-tmp/envs/tina/ /root/autodl-tmp/envs/tina/
rsync -a --partial root@OLD_HOST:/root/autodl-tmp/hf-cache/ /root/autodl-tmp/hf-cache/
rsync -a --partial root@OLD_HOST:/root/autodl-tmp/wheels/ /root/autodl-tmp/wheels/
rsync -a --partial 'root@OLD_HOST:/root/autodl-tmp/*.whl' /root/autodl-tmp/
```

项目内已有的评测venv会随整个Tina目录搬走。仅搬venv而不搬训练环境，会留下无法运行的Python软链接。换机器后的驱动、GPU架构、编译扩展缓存仍需按实际环境核验；当前已验证的是旧机，不声称新机器尚未执行过的恢复已通过。

## 环境版本与重建资料

精确记录见 `migration/train.freeze.txt`、`eval.freeze.txt`、`environment_snapshot.json`、`eval_overrides.txt`；最初安装记录见 `reproduction/installed.freeze.txt`、`constraints.txt`、`install_remaining.sh`。

关键版本：Python3.10、torch2.5.1+cu118、vLLM0.7.2+cu118、FlashAttention2.7.3、TRL0.15.2、DeepSpeed0.16.4、transformers4.50.0、PEFT0.15.0、accelerate1.5.2、math-verify0.5.2、xgrammar0.1.17；评测为LightEval0.8.1。

`installed.freeze.txt` 中vLLM/flash-attn的 `file://` 条目是旧机wheel路径，不能盲目在其他路径pip安装；普通PyPI的同版本vLLM wheel也不一定是CUDA11.8构建。已保留对应wheel及其校验信息。`migration/*.freeze.txt` 导出当前环境的版本，历史freeze另存不覆盖。

如果重建评测venv，先恢复训练环境，再用训练Python创建：

```bash
cd /root/autodl-tmp/Tina
/root/autodl-tmp/envs/tina/bin/python -m venv --system-site-packages artifacts/recheck/eval_env
# 使用迁移导出的评测差异列表，按原版本补齐；网络代理按新机情况设置。
artifacts/recheck/eval_env/bin/python -m pip install -r migration/eval_overrides.txt
```

原数据、模型revision见 `artifacts/download_manifest.json`；新增数据revision见 `artifacts/core_suite/dataset_access.json`；GPQA公开CSV和来源证明位于 `artifacts/core_suite/`。只阅读已有逐题结果无需访问HF或重新下载。

## 迁移后的完整性校验

```bash
cd /root/autodl-tmp/Tina
# 不需要GPU、PyTorch或评测环境，系统Python3即可。
python3 reproduction/verify_migration.py --quick
python3 reproduction/verify_migration.py
```

完整校验逐文件核对SHA-256与大小。快速校验只核对较小的文档、代码、配置、JSON/JSONL/CSV/Parquet和日志等证据，跳过大权重/优化器，不替代完整校验。覆盖清单见 `migration/file_manifest.json` 和 `SHA256SUMS`。

校验清单不覆盖 `.git`、Python环境目录、`__pycache__`、生成的证据zip，以及清单自身。环境另有版本快照；外部wheel有独立SHA文件。历史报告中的绝对路径按原样保留作为溯源记录。

轻量包单独校验：

```bash
sha256sum -c TINA_evidence.zip.sha256
unzip TINA_evidence.zip -d TINA_evidence
```

日志中少量W&B `debug-core.log` 是指向 `/root/.cache/wandb/` 的软链接，其内容已额外保存到 `migration/external_debug_logs/`，映射在 `external_symlinks.json` 中。原始训练stdout和离线`.wandb`记录保留在 `logs/`。

## 恢复后先检查什么

1. 阅读 [HANDOFF.md](HANDOFF.md) 确认已完成的实验；`migration/final_scores.csv` 为最终成绩入口，避免误读初轮legacy分数。
2. 用上述脚本做文件校验，确认没有漏搬`artifacts`或`logs`。
3. 若继续运行，执行 `source reproduction/env.sh`，检查训练和评测两个Python的版本、关键库导入及新机 `nvidia-smi`。
4. 保持原路径时原观察命令可直接用：`bash reproduction/watch_core.sh`。这是只读看板；已完成状态无需启动后台训练。
5. 若迁移到不同根目录，先修改运行脚本中的固定路径，重建venv；不要批量改写历史manifest、原始输出和校验清单。先保存迁移副本的原始校验结果，再记录新的路径修改。

## 重新合并adapter或开展新实验

机制阶段的临时合并权重已按计划删除，但500/850步adapter都在。重建示例使用新输出目录：

```bash
cd /root/autodl-tmp/Tina
source reproduction/env.sh
artifacts/recheck/eval_env/bin/python reproduction/recheck_merge.py \
  --adapter artifacts/mechanism/label_free_format/adapter500 \
  --output artifacts/migration_checks/merged_label_free500
```

该操作额外需要约3.4GiB磁盘；只是示例，准备迁移时没有执行它。

新RS3训练示例：从基座初始化，使用新目录，不覆盖原结果。实际运行前确保双卡和空间已准备好。

```bash
cd /root/autodl-tmp/Tina
source reproduction/env.sh
export TINA_RUN_NAME=grpo_rs3_new_seed42
export TINA_STOP_STEP=850
python -m accelerate.commands.launch \
  --config_file recipes/accelerate_ds_cfgs/ds_zero2.yaml \
  --num_processes 2 --main_process_port 29541 tina/post_train_hf/grpo.py \
  --config reproduction/open_rs3.yaml --cosine_max_len 3584
```

原 `train.sh main` 固定使用既有RS3目录；已完成的850步目录不应拿它盲目重跑。格式组500步只有adapter快照；如需严格恢复优化器轨迹，使用其保留的800/850完整断点。迁移准备未改变任何训练配置或权重。
