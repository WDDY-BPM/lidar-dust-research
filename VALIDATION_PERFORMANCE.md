# 完整验证加速与第 10 轮续训

用户传回的 `msg_b32_seed42` 中，训练平均每轮 225.9 秒，第 5 轮完整验证 1,287 帧用了 14,040.6 秒。`last.pt` 已提交第 10 轮全部 651 批训练，尚未提交该轮验证。以下命令仅在指定的 AutoDL GPU 服务器运行。

## 加速内容与边界

- 可选 `cuda-graph` 后端捕获完整 MSG 前向，用一次图重放代替大量 CPU 启动操作。FPS 四层共 1,360 次循环，当前代码很可能受启动开销影响；实际瓶颈占比和加速倍率仍须测量。
- FPS 的 CPU 随机起点作为每次图重放的输入。预热和捕获不消耗随机数。保持原来的每批 4 块、4096 点、逐帧 seed、分块/补点、四重投票、阈值和完整验证范围。末尾不足 4 块时仍执行普通前向。
- 去掉无重复真实索引累加中的 `np.add.at`，语义统计不再反复复制同一标签数组。逐帧日志增加耗时和验证 ETA。
- 不修改训练计算，也不根据当前验证结果改变训练预算。CUDA Graph 的速度和数值一致性尚未在用户的 3090 上实测；先运行下面的对照。首次捕获会较慢，日志会显示捕获开始/结束。
- 原理和静态输入要求见 [PyTorch CUDA Graph 官方说明](https://pytorch.org/blog/accelerating-pytorch-with-cuda-graphs/)。无需安装新 CUDA 扩展或重建 mc 环境。

## 1. 更新分支并保留原结果

确认原训练进程已停止。在服务器进入 tmux，更新代码：

```bash
tmux new -A -s lidar
conda activate /root/autodl-tmp/zl/mc
cd /root/autodl-tmp/Lin/lidar-dust-research
git fetch origin fix/validation-cuda-graphs &&
git switch fix/validation-cuda-graphs &&
git pull --ff-only origin fix/validation-cuda-graphs
git status --short
```

若 GitHub 网络不通，上传本地提供的 `lidar-dust-validation.bundle` 到 `/root/autodl-tmp/Lin/`，用下面两条替换 fetch/switch/pull：

```bash
git fetch /root/autodl-tmp/Lin/lidar-dust-validation.bundle fix/validation-cuda-graphs:fix/validation-cuda-graphs &&
git switch fix/validation-cuda-graphs
```

保留现有 run 的独立副本；若备份目录已存在，下面的保护条件会停止复制：

```bash
test ! -e runs/msg_b32_seed42_before_eval_update &&
cp -a runs/msg_b32_seed42 runs/msg_b32_seed42_before_eval_update
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
  --eval-backend cuda-graph \
  --save-every-steps 50 \
  --resume runs/msg_b32_seed42/last.pt \
  --resume-evaluation-update \
  2>&1 | tee -i -a logs/msg_b32_seed42.log
```

首次兼容迁移仅接受已审核的干净 `4e7b7ae` 检查点和干净的新源码，所有训练/验证配置必须相同。它保留模型、Adam、学习率、随机状态、累计损失、epoch 和批次进度；记录旧检查点 SHA256、旧/新源码版本到 `resume_lineage`。不通过放宽所有检查来接受其他源码或 batch size。

这份第 10 轮检查点恢复时会显示下一批 `652/651`，表示训练已全部提交，直接补做完整验证，不会重训第 10 轮。补完后继续第 11 轮。后续在同一版本恢复可去掉 `--resume-evaluation-update`。

`--eval-backend eager` 可保留普通前向。后端为执行选项；模型/随机起点/批次/覆盖协议相同，报告单独记录后端。不要直接把 `--eval-batch-size 4` 改为 32 后恢复：原始 FPS 随机起点依赖批次划分，且恢复检查会拒绝配置变化。

兼容迁移保留旧 best.pt 的真实来源；若以后验证超过原最佳值，则正常更新 best.pt。这是有来源记录的续训，不是把之前 10 轮伪装成由新版本训练。正式多种子汇总仍应冻结统一源码/配置，并检查各 best.pt 的来源一致性。

## 4. 测试与预计时间

训练完成后，仍使用完整测试集和 best.pt；在原 `evaluate_dust.py` 命令中添加 `--backend cuda-graph` 即可，不增加 frame limit。

原耗时推算：100 轮训练约 6.3 小时，20 次完整验证约 78 小时。若实测验证整体提速 3 倍，总计约 32.3 小时；若提速 5 倍，约 21.9 小时。这些是条件估算，不是已测结果；以新日志的整次 `validation_seconds` 为准。

合成回归检查放在远程 GitHub Actions。CPU 检查涵盖 FPS 输入/RNG、一致性保护和验证续训；CUDA 图回归在有 CUDA 的指定服务器可运行：

```bash
python -m unittest discover -s tests -p 'test_dust_validation.py' -v
```

普通 CPU CI 会明确跳过 CUDA 图实测。本机仅做源码与静态检查，不运行训练、推理或实验评价。
