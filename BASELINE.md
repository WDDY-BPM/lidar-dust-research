# PointNet++ MSG / LiDARDustX 第一阶段

本版实现我们约定的粉尘过滤基线：XYZI 输入，逐点预测有效点与粉尘，输出 F1、误删率和 PLY 可视化。它是基于 yanx27 源码的实验协议，未声称复现某篇论文的官方训练配置或指标。结构预测、保护损失和修复模块留到后续阶段。

更新：分支 `fix/msg-throughput-resume` 提供 FPS/邻域查询优化、短时吞吐量比较、轮内检查点及安全停止。AutoDL 离线更新、现有 `mc` 环境和断点重训命令见 [PERFORMANCE.md](PERFORMANCE.md)。本页保留第一版操作示例；新版恢复语义以下文更新及 PERFORMANCE.md 为准。

以下命令全部在实验服务器运行。本机不运行训练、推理或实验性预处理。GitHub 自动检查使用合成数据验证软件逻辑，不下载真实数据集。

## 1. 获取代码和建立环境

```bash
git clone --branch feat/lidardustx-baseline https://github.com/WDDY-BPM/lidar-dust-research.git
cd lidar-dust-research
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-dust.txt
# 示例：选择与服务器驱动兼容的 CUDA 11.8 wheel。
python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
python -m unittest discover -s tests -p 'test_dust*.py' -v
```

PyTorch 2.5.1 提供 cu118、cu121、cu124 和 CPU wheel，参见 [官方安装说明](https://pytorch.org/get-started/previous-versions/#v251)。选择具体 CUDA wheel 前核对服务器驱动。仅运行服务器上的 CPU 软件检查时，可将安装 URL 换成 `https://download.pytorch.org/whl/cpu`。推荐 Python 3.10、NumPy 1.26.4；不依赖 torchvision、Open3D 或自定义 CUDA 编译。

把 LiDARDustX 放到服务器，例如 `/data/LidarDustX`，该目录下面直接是 ls128、ls64、ly150、ly300、m1、ouster。私有仓库需要你已有的 GitHub 登录方式。

## 2. 核验数据并冻结协议

```bash
python prepare_dust.py --data-root /data/LidarDustX --output artifacts/lidardustx --seed 42
```

检查文件配对、字节数、原始标签范围、非有限点，并为每对文件计算 SHA-256。输出：

- `manifest.json`：相对文件路径、哈希、分组与划分、标签协议、坐标尺度、训练集强度统计和类别权重；带自身内容指纹。
- `audit.json`：标签直方图、非有限点、分集合与传感器统计。
- `train.txt / val.txt / test.txt`：便于人工审阅的帧清单；程序以 manifest 为准，不能单独修改这些 txt 来改变划分。

准备目录须为空；新协议使用新目录。若文件缺失、长度不匹配、标签超出 0..28，直接报错。训练与评价开始前，重新检查所用帧的内容哈希。这里能核验存储格式和数量对应关系；语义是否正确对应仍需人工检查点云标注。

默认按文件名里的 `sequence_XXXX` 分组，按传感器组成分层，将采集组约 70%/15%/15% 划为 train/val/test；比例是组数比例，帧数比例可能不同。无粉尘帧保留。三个集合须覆盖所有传感器且都有有效点与粉尘。

序列编号不能自动证明场景独立。正式跨场景评价前，核查同场景或同步采集的序列，提供完整 CSV：

```csv
sequence,group
sequence_0000,site_a_capture_01
sequence_0001,site_a_capture_01
sequence_0002,site_b_capture_02
```

每个序列必须恰好一行，相关序列用同一个 group。确认后在新目录执行：

```bash
python prepare_dust.py --data-root /data/LidarDustX --output artifacts/lidardustx_grouped --group-map capture_groups.csv --seed 42
```

后续命令示例使用 `artifacts/lidardustx/manifest.json`；若采用分组协议，全部命令使用对应 grouped manifest。三种训练随机种子及第二阶段模型必须复用同一 manifest。

## 3. 固定标签与预处理

| 原始 ID | 二分类标签 | 用法 |
|---|---|---|
| 9（Dust） | 1 | 粉尘，预测过滤 |
| 1–8、12–24、26–28 | 0 | 真实有效点 |
| 0、10、11、25 | -100 | 不参与损失和指标，有限坐标仍作为网络上下文 |

此映射是本实验选择；[官方数据说明](https://github.com/vincentweikey/LiDARDustX) 提供原始类别定义。Noise、Rainwater、Unknown 没有强行当作真实点。

- 点按小端 float32 的 XYZI 读取；标签是 uint8。用同一个 finite mask 同步筛点、筛标签、保留原始索引。
- 对所有有限强度做 `log1p(max(intensity, 0))`，按传感器使用该传感器**训练集所有有限点**的均值与标准差做标准化，验证与测试复用统计。负强度裁剪数量记录在 manifest；坐标不裁剪。
- 默认 10m × 10m XY 块、保留全 Z，高度中心为该块有限点的平均 Z；XYZ 减块中心再除以固定 10m。第一层归一化半径 0.05/0.1 对应实际 0.5/1m。
- 训练每帧每 epoch 随机取 4 个块，每块 4096 点。密集块无放回采样；稀疏块重复补齐。块中心从有监督点中选取并保留锚点，保证不会出现全 ignore 批次。各帧等量取块，不人为强制粉尘比例。
- 训练只做 Z 轴旋转；类别权重为训练集监督点数量的逆平方根，再归一化到均值 1。

原始 MSG 编码器/解码器通道、四层采样点数 1024/256/64/16、半径和 Dropout 保留。输入特征为 4 通道（第一层会再拼接 3 个局部坐标），输出两个 log-probability；使用带 ignore_index 的加权 NLL。平方距离对浮点舍入引起的负值归零。原 S3DIS 模型默认仍为 9 通道。

## 4. 先做服务器流程检查

```bash
CUDA_VISIBLE_DEVICES=0 python train_dust.py \
  --data-root /data/LidarDustX --manifest artifacts/lidardustx/manifest.json \
  --output runs/smoke --epochs 1 --batch-size 2 --workers 0 \
  --max-train-frames 12 --max-val-frames 6 --samples-per-frame 1

CUDA_VISIBLE_DEVICES=0 python evaluate_dust.py \
  --data-root /data/LidarDustX --manifest artifacts/lidardustx/manifest.json \
  --checkpoint runs/smoke/best.pt --split val --max-frames 6 \
  --output outputs/smoke --allow-smoke
```

确认损失有限、反向传播成功、每点覆盖正确、文件导出正常。smoke 结果带标记，不用于论文结果；真实服务器 GPU 尚未指定，以上 batch size 是起始值，不是显存适配保证。

## 5. 正式训练与续训

```bash
CUDA_VISIBLE_DEVICES=0 python train_dust.py \
  --data-root /data/LidarDustX --manifest artifacts/lidardustx/manifest.json \
  --output runs/msg_seed42 --seed 42 --epochs 100 --batch-size 4 --workers 4
```

默认 Adam，初始学习率 0.001，weight decay 0.0001；每 10 epochs 学习率乘 0.7，下限 0.00001。每 5 epochs 及最后一轮在**全部验证帧**上整帧评价，按固定阈值 0.5 的验证 F1 保存 `best.pt`，每轮保存 `last.pt`。测试集不参与权重计算、选模或阈值选择。

`config.json` 记录协议指纹、源码 commit、超参和软件环境；`history.jsonl` 记录训练损失、学习率、验证指标以及训练/验证耗时。每次验证生成独立结果目录。新版默认每 100 个 batch、验证前及 epoch 结束保存 `last.pt`，并在 `Ctrl+C`/SIGTERM 后完成当前批次再保存。检查点包含已提交批次位置和 NumPy/Python/PyTorch/CUDA 随机状态，支持从轮内下一批恢复：

```bash
CUDA_VISIBLE_DEVICES=0 python train_dust.py \
  --data-root /data/LidarDustX --manifest artifacts/lidardustx/manifest.json \
  --output runs/msg_seed42 --seed 42 --epochs 100 --batch-size 4 --workers 4 \
  --resume runs/msg_seed42/last.pt
```

续训要求相同源码、协议和训练设置；只能从原目录的 last.pt 续训。默认拒绝覆盖已有新训练目录。本版使用上游纯 PyTorch FPS 与邻域查询，整帧验证成本较高；先测一轮耗时，再估计 100 轮预算。确定学习率、块尺度或 epoch 预算时只能用训练/验证集，冻结后对所有对比模型保持一致。

## 6. 整帧测试和可视化

```bash
CUDA_VISIBLE_DEVICES=0 python evaluate_dust.py \
  --data-root /data/LidarDustX --manifest artifacts/lidardustx/manifest.json \
  --checkpoint runs/msg_seed42/best.pt --split test \
  --output outputs/msg_seed42 --visualize-per-sensor 1
```

XY 采用 10m 网格，默认以 5m 偏移得到四组覆盖。只遍历有点的格子；密集格子打乱后分批，保证每个有限点在每组覆盖中恰好有一次有效预测。稀疏格子补齐点不计额外票，四次有效概率取平均；覆盖断言失败直接报错。评价 FPS 随机性按固定种子和帧路径冻结，并恢复训练 RNG。保持同一评价 batch size，避免 FPS 随机起点随批次变化。

`metrics.json` 输出点数加权的全局 Precision、Recall、F1、Dust IoU、误删率，按传感器/序列的结果，以及无粉尘帧和各有效语义类别的误删率。`frames.json` 输出逐帧混淆计数、非有限点和投票数。

粉尘为正类：`F1 = 2TP/(2TP+FP+FN)`；`误删率 = FP/(FP+TN)`。非有限点和 ignore 标签不进入混淆矩阵，分母为零时指标写 null。不能用简单准确率替代这些指标。

默认每种传感器导出首个测试帧的以下文件，可用 CloudCompare 查看：

- `*_ground_truth.ply`：有效点绿、粉尘蓝、忽略类灰。
- `*_prediction.ply`：预测保留绿、预测粉尘蓝。
- `*_errors.ply`：误删有效点红、漏删粉尘橙，其他沿用真实类别颜色。
- `*_filtered.ply / *_filtered.bin`：按预测概率过滤后的有限点，过滤不依赖真实标签。
- `*_prediction.npz`：原始点顺序的概率、预测、语义标签和有限点原始索引；非有限点概率为 NaN、预测为 -100。

导出的是有限点；原始点云文件未改写。默认保留每种传感器一个可视化帧，可提高 `--visualize-per-sensor` 来检查更多帧。阈值来自 checkpoint，测试入口不允许另调阈值。

## 7. 三个种子汇总

复用同一 manifest，分别按以上正式命令训练和测试 seeds 42、43、44，使用不同 runs/outputs 目录，然后：

```bash
python summarize_dust.py \
  outputs/msg_seed42/metrics.json outputs/msg_seed43/metrics.json outputs/msg_seed44/metrics.json \
  --output outputs/three_seed_summary.json
```

汇总检查不同种子、相同划分/源码/阈值/评价设置、相同训练设置与训练预算且均为完整测试，输出均值与样本标准差。若通过续训延长预算，较早保存的 best.pt 仍记录其当时的请求预算；统一设置预算并核对日志后再汇总。第一阶段完成须有数据核验、固定分组清单、三次正式训练日志与权重、完整测试指标和人工检查的可视化。当前提交只提供代码；合成软件测试通过也不能替代这些真实实验结果。
