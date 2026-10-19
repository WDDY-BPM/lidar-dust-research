# 完整验证加速、间隔调整与断点续训

用户传回的 `msg_b32_seed42` 中，训练平均每轮 225.9 秒，第 5 轮完整验证 1,287 帧用了 14,040.6 秒。`last.pt` 已提交第 10 轮全部 651 批训练，尚未提交该轮验证。以下命令仅在指定的 AutoDL GPU 服务器运行。

## 当前调整：安全停止后改为每 10 轮验证

不需要重新训练。若进程仍在运行，在原训练终端按一次 `Ctrl+C`，等待 `Stopped safely`、`Stopped during validation` 或其他检查点已保存的退出提示，再更新分支并恢复。训练中暂停会保留已提交批次；验证中暂停保留该轮已完成的训练，部分验证结果不会用于选模。若已经退出，直接执行下面的更新步骤。

本页续训命令现已显式使用 `--eval-every 10 --resume-eval-schedule-update`。第 10 轮仍需完整验证，之后验证第 20、30……100 轮；最终轮即使不是 10 的倍数也会验证。已有验证记录及 best.pt 保留。若当前检查点已到更后面的轮次，则从实际保存位置继续。

## 加速内容与边界

- 可选 `cuda-graph` 后端捕获完整 MSG 前向，用一次图重放代替大量 CPU 启动操作。FPS 四层共 1,360 次循环，当前代码很可能受启动开销影响；实际瓶颈占比和加速倍率仍须测量。
- FPS 的 CPU 随机起点作为每次图重放的输入。预热和捕获不消耗随机数。保持原来的每批 4 块、4096 点、逐帧 seed、分块/补点、四重投票、阈值和完整验证范围。末尾不足 4 块时仍执行普通前向。
- 去掉无重复真实索引累加中的 `np.add.at`，语义统计不再反复复制同一标签数组。逐帧日志增加耗时和验证 ETA。
- 用户在 3090 上使用干净 `27ee46b`、第 10 轮权重完成了 6 帧对照：普通前向 60.21 秒，CUDA Graph 15.71 秒，含捕获提速 3.83 倍；6 帧概率最大差异均为 0，投票、分类结果和混淆矩阵一致。这是小样本实测，不代表所有帧完全一致或完整验证必然具有相同倍率。
- 本次只进一步调整验证间隔，训练计算和图后端保持原实现。已完成上述对照的服务器无需重复运行第 2 步。首次图捕获会较慢，日志会显示捕获开始/结束。
- 原理和静态输入要求见 [PyTorch CUDA Graph 官方说明](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)。无需安装新 CUDA 扩展或重建 mc 环境。

## 1. 更新分支并保留原结果

确认原训练进程已停止。使用服务器现有的 SSH 别名先测试连接（该命令只验证身份，不会更新代码）：

```bash
ssh -T git@github-autodl
```

出现 GitHub 的认证成功提示后，在 tmux 内激活环境并更新代码：

```bash
tmux new -A -s lidar
conda activate /root/autodl-tmp/zl/mc
cd /root/autodl-tmp/Lin/lidar-dust-research
git remote set-url origin git@github-autodl:WDDY-BPM/lidar-dust-research.git
git fetch origin refs/heads/fix/validation-interval-resume:refs/remotes/origin/fix/validation-interval-resume &&
git switch fix/validation-interval-resume &&
git pull --ff-only origin fix/validation-interval-resume
git status --short
```

若 GitHub 网络不通，上传本地提供的 `lidar-dust-interval.bundle` 到 `/root/autodl-tmp/Lin/`，用下面两条替换 fetch/switch/pull：

```bash
git fetch /root/autodl-tmp/Lin/lidar-dust-interval.bundle fix/validation-interval-resume:fix/validation-interval-resume &&
git switch fix/validation-interval-resume
```

保留现有 run 的独立副本；若备份目录已存在，下面的保护条件会停止复制：

```bash
test ! -e runs/msg_b32_seed42_before_interval &&
cp -a runs/msg_b32_seed42 runs/msg_b32_seed42_before_interval
```

## 2. 同一权重先对照 6 帧

```bash
CUDA_VISIBLE_DEVICES=0 python -u benchmark_validation.py \
  --data-root /root/autodl-tmp/Lin/LidarDustX_complete \
  --manifest artifacts/lidardustx/manifest.json \
  --checkpoint runs/msg_b32_seed42/last.pt \
  --max-frames 6 \
  --output outputs/validation_benchmark_seed42.json
```

默认从验证集中跨传感器选 6 帧，使用同一个检查点，对比普通前向和图重放。报告包括每帧耗时、概率最大差异、全部有限点的阈值预测、混淆矩阵、投票和调用者随机状态；不会更新检查点或用测试集选设置。

`sample_parity_passed` 只有在实际发生图重放，概率最大差异不超过 `1e-6`，且投票、阈值预测、混淆矩阵一致时才为 true。失败会先保存报告再报错。样本对照不代表已证明所有帧数值完全一致。

`speedup_including_capture` 为普通耗时 / 图重放耗时，包含首次图捕获及分块、特征、传输、投票，排除两者共用的文件读取/哈希验证。6 帧中的捕获占比可能较高；可使用新的 output 路径增加到 `--max-frames 24` 复核。不要仅根据 GPU 利用率判断效果。

若捕获报 CUDA 错误或显存不足，退出该进程，保留报告/日志；使用 `eager` 后端可按原方式继续。不要在捕获失败的 CUDA 进程中继续使用图。

## 3. 对照通过后，从原 last.pt 继续

```bash
mkdir -p logs
CUDA_VISIBLE_DEVICES=0 python -u train_dust.py \
  --data-root /root/autodl-tmp/Lin/LidarDustX_complete \
  --manifest artifacts/lidardustx/manifest.json \
  --output runs/msg_b32_seed42 \
  --seed 42 --epochs 100 \
  --batch-size 32 --workers 4 \
  --lr 0.001 --eval-batch-size 4 \
  --eval-every 10 \
  --eval-backend cuda-graph \
  --save-every-steps 50 \
  --resume runs/msg_b32_seed42/last.pt \
  --resume-eval-schedule-update \
  2>&1 | tee -i -a logs/msg_b32_seed42.log
```

首次间隔迁移仅接受已审核的干净 `4e7b7ae` / `27ee46b` 检查点和干净的新源码，或同一干净版本下的间隔修改。除 `eval_every` 与已知的源码更新外，训练/验证配置必须相同。它保留模型、Adam、学习率、随机状态、累计损失、epoch 和批次进度；记录旧检查点 SHA256、旧/新源码、旧/新间隔、生效轮次和批次到 `resume_lineage`，并保存归一化的 `validation_schedule`。

这份第 10 轮检查点恢复时会显示下一批 `652/651`，表示训练已全部提交，直接补做完整验证，不会重训第 10 轮。补完后继续第 11 轮，到第 20 轮再验证。启动日志会显示 `Validation interval changed: 5 -> 10` 和 `Full validation every 10 epochs`。后续在同一版本恢复可去掉 `--resume-eval-schedule-update`，但必须继续保留 `--eval-every 10`。

若在原本要验证的第 15 轮暂停后改为每 10 轮，新间隔会跳过尚未完成的第 15 轮验证，记录该轮训练后继续；已完成的验证不会被删除。最终轮验证始终保留。

`--eval-backend eager` 可保留普通前向。后端为执行选项；模型/随机起点/批次/覆盖协议相同，报告单独记录后端。不要直接把 `--eval-batch-size 4` 改为 32 后恢复：原始 FPS 随机起点依赖批次划分，且恢复检查会拒绝配置变化。

兼容迁移保留旧 best.pt 的真实来源；若以后验证超过原最佳值，则正常更新 best.pt。这是有来源记录的续训，不是把之前 10 轮伪装成由新版本训练。正式多种子汇总仍应冻结统一源码/配置，并检查各 best.pt 的来源一致性。

改变间隔会改变最佳模型候选集合。这次保留之前的验证候选，不把旧记录改写为“从头每 10 轮验证”。测试报告会携带间隔历史，汇总会拒绝混合不同历史；后续正式多种子/消融实验应统一整个验证计划。新实验从头使用 `--eval-every 10` 即可，不加迁移参数。

## 4. 测试与预计时间

训练完成后，仍使用完整测试集和 best.pt；在原 `evaluate_dust.py` 命令中添加 `--backend cuda-graph` 即可，不增加 frame limit。

按 6 帧实测的 3.83 倍推算，单次完整验证约 61 分钟。每 10 轮验证时，100 轮训练加 10 次验证约 16～17 小时；从第 10 轮附近继续约需 15～16 小时，取决于第 10 轮验证是否完成。这些是条件估算，不是完整验证实测；以新日志的整次 `validation_seconds` 为准。

合成回归检查放在远程 GitHub Actions。CPU 检查涵盖 FPS 输入/RNG、一致性保护和验证续训；CUDA 图回归在有 CUDA 的指定服务器可运行：

```bash
python -m unittest discover -s tests -p 'test_dust*.py' -v
```

普通 CPU CI 会明确跳过 CUDA 图实测。本机仅做源码与静态检查，不运行训练、推理或实验评价。
