# LiDAR 粉尘过滤与选择性修复

本仓库用于管理五阶段研究的源码、配置和文档。研究顺序见 [RESEARCH_PLAN.md](RESEARCH_PLAN.md)。

## 当前状态

- 初始源码来自 https://github.com/yanx27/Pointnet_Pointnet2_pytorch ，版本 `eb64fe0b4c24055559cea26299cb485dcb43d8dd`。
- 导入该版本的源码快照，保留原 README 和 MIT LICENSE；未导入上游 log 目录中的权重、日志及模型副本，未导入上游提交历史。
- 新增 LiDARDustX 数据核验、固定划分、PointNet++ MSG 二分类训练与整帧评价入口，使用方法见 [BASELINE.md](BASELINE.md)。原始 train_semseg.py 仍是 S3DIS 流程。
- 当前为基线实现；真实数据训练和指标需在实验服务器上验证，不代表已复现出实验结果。
- 数据集在仓库外管理，不上传到 GitHub。不在本机开展实验。

## 分支与审阅

- main 保存初始源码及后续经用户确认合并的版本。
- 每项改动先创建独立分支，再提交并推送，通过 PR 查看差异。
- 分支示例：feat/lidardustx-loader、feat/dust-binary-training、feat/dust-evaluation。
- 未经用户要求不合并 PR。后续任务应从适当的已审阅版本出发，存在依赖时在 PR 中说明。

## 数据检查记录

2026-10-12 对用户提供的 LiDARDustX 解压文件进行了只读检查：7,562 对 bin/label，6 种传感器；文件大小与提供的 ZIP 目录一致；按每点四个 float32 和一个 uint8 标签计算，全部配对的点数和标签数一致。这不等同于全量内容校验或语义对齐验证。

抽查发现 M1 中存在非有限坐标，三个抽查帧中这些点的标签均为 0；Ouster 强度超过 255。后续加载器需同步处理点和标签，并单独确定强度归一化协议。
