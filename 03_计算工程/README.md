# 可继续开发的计算工程

在本目录运行Python。完整说明见 `../01_使用说明/复现与交付指南.md`。

| 路径 | 内容 |
|---|---|
| src | 数据读取、预测、优化、回放、导出和复核程序 |
| configs/q1_q2_baseline.json | 当前完整基线的模型与运行参数 |
| tests | 现有20项检查 |
| data/raw | 原始题面和附件快照，勿覆盖 |
| data/processed | 整理后的CSV，当前可直接使用 |
| reports | 数据检查和预处理依据 |
| outputs/runs | 后续运行自动创建的新实验目录 |

本次已完成的运行记录保存在 `../05_复核记录/本次完整运行/`，正式表格和图保存在 `../04_正式成果/`。
旧诊断入口run_q1.py和q1_diagnostic.json保留供追溯；当前主入口为run_pipeline.py和q1_q2_baseline.json。
