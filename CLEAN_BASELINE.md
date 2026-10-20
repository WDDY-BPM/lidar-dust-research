# 修正划分、核查车辆误删、固定三种子协议

本页所有命令在 AutoDL 服务器的 `mc` 环境执行。本机仅准备源码、阅读已有结果及静态检查。旧的 `runs/msg_b32_seed42`、`outputs/msg_b32_seed42_test` 和 `artifacts/lidardustx` 保留；新实验使用不同目录，原始 BIN/LABEL/TXT 文件不删除、不改写。

2026-10-20 返回报告已复核，见 [AUDIT_REVIEW_20261020.md](AUDIT_REVIEW_20261020.md)。旧 `lidardustx_clean_v2` 已生成，但 train 仅占 33.96%、test 中 LS64 Car 为 0、Ouster Dust 仅 110 点，暂不冻结、不启动三种子。下面使用 v3 分配器和新目录；已完成的旧模型诊断可以沿用。

## 已核实与待核实

- [官方原始类别表](https://github.com/vincentweikey/LiDARDustX#semantic-segmentation-label-definition) 中 ID 12 是 Car，ID 9 是 Dust，ID 16 是 Pickup。当前 Car→keep、Dust→dust 的映射正确。旧测试 Car 误删率 50.05%，不能通过把 Car 改成忽略类来消除。
- 已保存的全数据审计哈希包含 25 组重复点云，23 组的标签文件不同，其中 21 组点云跨旧 train/val。标签文件不同不等于二分类标签一定矛盾；服务器诊断会区分原始语义差异、keep/dust 对立和 ignore/监督差异。
- 旧 `sequence_0114` 的整体误删率为 27.52%，部分帧接近 99%。映射错误已排除；标签语义对齐、车辆监督覆盖、模型的距离/强度偏置和结构保护不足尚不能只靠这些平均指标判定。

## 新的确定性数据规则

1. 所有相同 UTC 采集日的数据跨传感器绑定，整条序列作为不可拆分单位。相同点云所在的序列也先绑定，避免只去掉重复帧后让相邻帧继续跨集合。完整用户 capture map 可以进一步合并组，不能拆开已绑定的日期/点云关系。
2. 相同点云、相同标签只保留字典序最小的一个有效副本。相同点云但标签文件冲突时，隔离所有副本，等待独立标注核查；不根据模型分数挑选标签。
3. 文件名秒数落在固定 `[1500000000, 1900000000)` 范围之外时视为无法确定 UTC 采集日。默认隔离。当前保存清单中有 266 帧属于这种情况，主要是 M1 和 Ouster 的相对/异常时钟。只有提供完整、经过核查的真实采集映射时才可按映射纳入。
4. 固定 seed=42，以完整组做一次计数约束分配，接近 70/15/15 的帧/点份额，而非组数比例。每个 split/sensor 的 keep、dust 各至少 `min(1000, max(1, total_points // 100))` 点；某传感器 Car 至少出现在三个独立组时，各 split 至少 `min(100, max(1, total_car_points // 100))` 个 Car 点。Car 独立组不足三个时明确报告限制，不伪造覆盖。强度统计、类别权重只从保留后的训练帧计算。不能为改善模型分数反复挑划分种子。
5. 该协议证明的是规则下的**采集日/采集组分离**，没有矿区或物理场景元数据时，不声称物理场景独立。公开名称为 `capture_day_disjoint_v2`，不是官方 benchmark 配置。
6. 分配器另记为 `capture_coverage_balanced_v3`，使用 [SciPy 1.14.1 的确定性 MILP 求解器](https://docs.scipy.org/doc/scipy-1.14.1/reference/generated/scipy.optimize.milp.html)，只读取分组、帧数、审计类别计数。目标是归一化绝对份额偏差：全局帧数权重 4、各传感器帧数合计 1、全局二分类点数合计 1、各传感器二分类点数合计 1、具备三组 Car 的传感器 Car 点数合计 0.25。seed 只用于极小的固定平局项。30 秒/100000 节点内未找到满足最优性间隙 1e-6 的解就停止，不自动降级。完整组下实际份额未必能精确达到目标。

## 1. 用已有 GitHub SSH 别名更新

```bash
conda activate /root/autodl-tmp/zl/mc
cd /root/autodl-tmp/Lin/lidar-dust-research
git remote set-url origin git@github-autodl:WDDY-BPM/lidar-dust-research.git
git fetch origin refs/heads/fix/balanced-capture-coverage:refs/remotes/origin/fix/balanced-capture-coverage &&
git switch fix/balanced-capture-coverage &&
git pull --ff-only origin fix/balanced-capture-coverage
python -m pip install -r requirements-dust.txt
mkdir -p logs
```

在 tmux 中执行后续命令。新增分配器需要固定的 SciPy 1.14.1；沿用现有 mc/PyTorch 环境。

## 2. 先核查旧模型的车辆误删

若已传回本次六帧诊断，且 `all_review_predictions_available` 和 `all_review_confusions_match` 均为 true，直接沿用报告，跳到第 3 步。不要对非空诊断目录重复执行。

```bash
CUDA_VISIBLE_DEVICES=0 python -u diagnose_dust.py \
  --data-root /root/autodl-tmp/Lin/LidarDustX_complete \
  --manifest artifacts/lidardustx/manifest.json \
  --results outputs/msg_b32_seed42_test \
  --checkpoint runs/msg_b32_seed42/best.pt \
  --sequence sequence_0114 --review-frames 6 \
  --backend cuda-graph \
  --output audits/legacy_seed42_review \
  2>&1 | tee -i logs/diagnose_legacy_seed42.log
```

先扫描已验证标签，统计旧 train/val/test 各传感器的 Car 点数与含 Car 帧数；然后逐点比较重复帧的标签差异。只对目标序列选取少量高误删、低误删和中间表现的帧，用固定旧 best.pt 做诊断前向。保持旧阈值、FPS seed 和批次，不训练、不调阈值、不把选出的诊断帧当作新的完整测试成绩。

输出目录包括：

- `summary.json`：类别映射、采集问题、代表帧的 Car 误删率、范围/强度/概率分布及与原逐帧计数的一致性。
- `label_inventory.json`：各 split/sensor/sequence 的有限点原始类别计数和 Car 覆盖，定位监督稀缺/分布差异。
- `duplicate_label_differences.json`：语义差异、真正的 keep↔dust 矛盾、ignore↔监督差异。
- `capture_audit.json`、`capture_groups.csv`、`excluded_frames.csv`：按元数据得到的修正规则与隔离清单。自动生成的 CSV 不是已经核实的物理场景真值，不要用它冒充人工采集记录。
- `review/*_car_review.ply`：Car 被保留为绿、Car 被删除为红、真实粉尘为蓝、其他点为灰。
- `review/*_boxes.json`：原 TXT 框，以及“角度为弧度、Z 旋转、X 长/Y 宽”下中心 Z/底面 Z 两种解释。用于人工检查，不据此自动修正点标签。
- `visualizations/`：代表帧的完整预测、误删图和原始索引 NPZ。

`all_review_confusions_match: true` 表示新诊断复现了原预测计数，不代表标注已经人工确认正确。距离/强度相关性也不能单独证明因果。若该项为 false，工具会保存报告再报错，先核查模型/数据/环境身份。

## 3. 在新目录重建修正协议的数据清单

```bash
python -u prepare_dust.py \
  --data-root /root/autodl-tmp/Lin/LidarDustX_complete \
  --output artifacts/lidardustx_clean_v3 \
  --seed 42 \
  2>&1 | tee -i logs/prepare_clean_v3.log
```

该步骤重新核验全部输入、绑定采集组、执行隔离/去重，再生成新的 manifest 和训练统计。不移动原始文件。输出 `audit.json` 包含实际保留/隔离数量、新划分和各 split/sensor 原始语义分布；同时保存采集组、隔离清单和 `split_quality.json`，后者给出实际帧份额、覆盖约束、求解状态与稀缺 Car 覆盖限制。新 manifest 的指纹与旧实验不同，正式基线要从头训练，旧权重只用于上面的故障定位。

已完成的三个旧诊断报告继续保留；本次只需传回下面新增的 v3 `audit.json` 与 `split_quality.json`，检查新划分后再进入长时间训练：

```text
audits/legacy_seed42_review/summary.json
audits/legacy_seed42_review/label_inventory.json
audits/legacy_seed42_review/duplicate_label_differences.json
artifacts/lidardustx_clean_v3/audit.json
artifacts/lidardustx_clean_v3/split_quality.json
```

## 4. 冻结统一基线协议

诊断与数据清单对应同一份输入数据、代表帧确有 Car 且复现原预测，并通过 v3 覆盖检查后，可生成带指纹的固定协议。旧 v2 清单会被明确拒绝：

```bash
python freeze_baseline.py \
  --data-root /root/autodl-tmp/Lin/LidarDustX_complete \
  --manifest artifacts/lidardustx_clean_v3/manifest.json \
  --diagnostics audits/legacy_seed42_review/summary.json \
  --output protocols/clean_msg_v3
```

固定 seeds 42/43/44、同一新 manifest、源码 commit、100 epochs、batch 32、4096 点、每帧 4 块、lr 0.001、weight decay 0.0001、阈值 0.5、每 10 轮及最终轮完整验证、验证 batch 4、CUDA Graph。协议不包含自动调参；旧测试仅用于故障定位，新模型选择只用新训练/验证集。

## 5. 运行三个种子的基线

```bash
bash protocols/clean_msg_v3/run_baseline.sh
```

脚本先校验冻结的源码、协议和 manifest，按 seed 顺序串行训练、测试并汇总。训练中按一次 Ctrl+C 安全停止；重新运行同一脚本会从当前 seed 的 last.pt 恢复，已完整完成且身份匹配的训练/测试会跳过。部分测试目录会明确报错，保留/改名该目录后再重跑该次测试；不覆盖已有结果。

新目录：`runs/clean_msg_v3_seed42`、`seed43`、`seed44`（均有完整前缀）；测试在 `outputs/clean_msg_v3_seed*_test`，最终均值/样本标准差在 `outputs/clean_msg_v3_three_seed_summary.json`。已完成的旧 seed42 不混入这组三种子。

## 6. 结构保护实验的后续顺序

本次提交准备数据修正、诊断和统一基线，不声称结构模块已经实现或实验完成。根据诊断报告区分标签问题、监督覆盖、距离/强度偏置和模型结构问题后，在独立分支实现以下消融：

| 组别 | 改动 | 必须对照 |
|---|---|---|
| B0 | 原 PointNet++ MSG | 新协议三种子基线 |
| S1 | 多尺度几何结构特征 | B0，确认特征本身的作用 |
| S2 | S1 + 辅助结构预测 | S1，确认辅助任务的作用 |
| S3 | S2 + 结构保护损失 | S2，确认保护损失的作用 |

复用同一份采集组划分、三种子与训练预算。结构输入/推理只能依赖可观测点云，真实语义不作为推理输入。各组报告总体及各传感器 F1、粉尘召回、真实点误删率、Car/Pickup 等语义保护情况与失败案例；超参及选择条件先在新验证集固定，最终测试不用于选择损失权重。先证明跨种子保护稳定，再进入研究计划第三阶段的 GIMUN 修复数据。
