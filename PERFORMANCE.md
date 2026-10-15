# 服务器加速与轮内断点续训

本版对应分支 `fix/msg-throughput-resume`，依赖第一版基线分支；不合并或覆盖旧实验记录。
下面命令只在用户指定的 AutoDL 服务器执行。本机只准备源码、文档与离线包，不运行模型或实验。

## 改动与边界

- FPS 的距离更新使用固定形状的 `torch.where`，去掉采样循环中的动态布尔索引；保留 CPU RNG 产生起点的方式、采样点数和距离计算。
- 球邻域查询使用 `topk` 选取最小输入索引，代替全量排序；仍按原始索引选邻居，不改成按距离选邻居。保留特征传播的距离排序，以免改变等距离点的选择。
- loss 分母去掉动态布尔索引，损失统计留在设备上，日志或检查点时才读取标量。非有限 loss 或全 ignore 批次在下一次日志/保存检查时报错，不保存污染的状态。
- 每个 worker 默认缓存 16 帧；worker 跨 epoch 保留，epoch 随每个采样任务传入，避免持久 worker 使用旧 epoch。
- 新训练顺序采用按 seed 和 epoch 独立生成的排列，允许精确定位下一批。它不同于旧版的连续 DataLoader 排列；因此旧检查点不能直接作为新版精确续训。
- 不依赖额外 CUDA 编译，不启用 AMP，不修改标签、4096 点输入、MSG 结构、每帧每轮 4 块、强度统计、完整验证覆盖和阈值。

这些是源代码优化，尚无 RTX 3090 上的实测提速承诺。以服务器上的 `samples/s` 和实际训练/验证耗时为准。

## 1. 停止旧训练并更新源码

在旧训练的会话中先查看是否有 `runs/msg_seed42/last.pt`。旧版仅在整轮训练和该轮验证完成后保存：没有这个文件时，中断会丢失尚未保存的进度；新版无法补回旧进程的内存状态。
保留旧目录。如果希望保留本轮，等待它完成并保存；需要立即停止时，在原会话按 `Ctrl+C`。
测速和新训练开始前应确保旧训练已退出，避免两个训练进程争用同一 GPU。

GitHub 能访问时，在项目目录执行：

```bash
git fetch origin fix/msg-throughput-resume
git switch fix/msg-throughput-resume
```

GitHub 无法访问时，将配套完整 Git bundle 上传到 `/root/autodl-tmp/Lin/lidar-dust-perf.bundle`，然后：

```bash
cd /root/autodl-tmp/Lin/lidar-dust-research
git fetch /root/autodl-tmp/Lin/lidar-dust-perf.bundle fix/msg-throughput-resume:fix/msg-throughput-resume
git switch fix/msg-throughput-resume
```

如果工作区有未提交源码改动，先检查 `git status`，不要强制覆盖。环境和数据无需重装、重新划分。

## 2. 环境与软件检查

在 tmux 会话里激活已有环境，设置已校验完成的数据路径：

```bash
conda activate /root/autodl-tmp/zl/mc
cd /root/autodl-tmp/Lin/lidar-dust-research
export DATA_ROOT=/root/autodl-tmp/Lin/LidarDustX_complete
python -m unittest discover -s tests -p 'test_dust*.py' -v
```

测试只使用合成小文件。FPS/邻域回归在 CPU 执行；服务器 CUDA 可用时也比较 GPU 与旧算法的采样索引。MSG 前后向、轮内恢复、多进程预取、SIGINT 和验证中断用独立检查验证。

## 3. 先测试 4、8、16 的吞吐量

```bash
CUDA_VISIBLE_DEVICES=0 python -u benchmark_dust.py \
  --data-root "$DATA_ROOT" --manifest artifacts/lidardustx/manifest.json \
  --batch-sizes 4 8 16 --workers 4 --warmup 5 --steps 30 \
  --output outputs/throughput_3090.json
```

该脚本仅从训练集选取固定子集，包含读取、取块、前向、反向和优化器更新。每种 batch 独立初始化临时模型，不输出模型权重、不作为正式实验结果。
每组先预热，在计时边界等待 CUDA 完成，输出 `seconds_per_batch`、`samples_per_second` 和峰值 allocated/reserved 显存；OOM 会记录并继续处理下一组。
吞吐量越高越好，不要求 GPU 利用率到 100%。测速不包含整帧验证，也不能直接证明完整数据吞吐量与收敛效果。

## 4. 新训练，先运行 50 步验证保存

以下 batch size 8 是起始示例，根据上一步实测选择 4/8/16。batch size 改变会影响训练过程，正式预算冻结后所有对比模型、三种子使用相同配置，不在同一个 run 中修改。

```bash
mkdir -p logs
CUDA_VISIBLE_DEVICES=0 python -u train_dust.py \
  --data-root "$DATA_ROOT" --manifest artifacts/lidardustx/manifest.json \
  --output runs/msg_fast_seed42 --seed 42 --epochs 100 \
  --batch-size 8 --workers 4 --cache-frames 16 \
  --save-every-steps 50 --log-every 25 --stop-after-steps 50 \
  2>&1 | tee -i logs/msg_fast_seed42_first50.log
```

50 步后正常保存 `runs/msg_fast_seed42/last.pt` 并退出；它是总预算为 100 轮的新训练的前 50 步，不是丢弃后重新训练的独立 smoke 实验。此时没有经过验证，通常还没有 `best.pt`。
日志给出 `s/batch`、`samples/s` 和本轮纯训练的 `train_ETA_min`；第一步包含预热，后续窗口更有参考意义。每轮记录 `train_seconds` 和 `validation_seconds`，完整验证耗时单列，训练 ETA 不包含它。

## 5. 从下一批继续，重复运行同一恢复命令

确认以上保存成功后，去掉 `--stop-after-steps`，加 `--resume`：

```bash
CUDA_VISIBLE_DEVICES=0 python -u train_dust.py \
  --data-root "$DATA_ROOT" --manifest artifacts/lidardustx/manifest.json \
  --output runs/msg_fast_seed42 --seed 42 --epochs 100 \
  --batch-size 8 --workers 4 --cache-frames 16 \
  --save-every-steps 50 --log-every 25 \
  --resume runs/msg_fast_seed42/last.pt \
  2>&1 | tee -i -a logs/msg_fast_seed42.log
```

- `last.pt` 保存模型、Adam、学习率调度器、随机状态、已提交批次位置、部分轮的累计损失及训练时间。预取中的批次不计作完成；恢复时重新生成同一轮顺序，直接从下一批开始，不重新读取并丢弃前面所有样本。
- 新版按 `Ctrl+C` 或收到 SIGTERM 后，完成当前训练批次并保存再退出。避免连续发送信号或 `kill -9`；强制关机、掉电等最多恢复到最近一次成功保存的位置。临时文件写完后原子替换，原检查点不会在写入途中被覆盖。
- 命令中的 `tee -i` 让日志进程忽略 SIGINT，避免 `Ctrl+C` 时它先退出、训练进程向已关闭管道输出。也可直接在 tmux 内运行 Python，不接 tee。
- 验证前另存 `last.pt`。验证中断时，恢复会跳过该轮训练、重新完成该次完整验证，再推进学习率和 epoch；不把部分验证当作选模结果。
- `best.pt` 仍只由完整验证 F1 选择。history 写入中断时，以检查点中的提交位置和最后一条记录恢复，避免重复行。
- 保持同一源码 commit、manifest、batch size、seed 和训练/验证配置。workers、cache、日志和保存频率可以调整；epoch 可延长，但最终三种子汇总前必须统一预算。
- 关闭 SSH 时，用 `Ctrl+B`，松开后 `D` 离开 tmux；它不能阻止平台真正关机。

## 6. 旧权重的使用

旧版 v1 检查点需要旧源码才能按原协议续训。新版可显式使用 `--init-checkpoint /绝对路径/旧last.pt` 初始化模型，但必须指定新的 output，且不同时传 `--resume`。
这会重置 Adam、学习率、epoch 和最佳 F1，在 config 中记录旧权重哈希及来源；属于基于旧权重重新训练，不是原实验精确续训。没有旧检查点时，使用上述从头训练命令即可。

正式测试仍使用 BASELINE.md 的 `evaluate_dust.py`，更换为新 run 的 `best.pt`。当前默认按序列划分，正式跨场景结论前仍需核查同场景/同步采集分组。
