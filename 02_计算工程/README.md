# 第四问计算工程（精简版）

模型和数值代码保持原样。提交表与论文图统一放在相邻的 `../01_论文成果/`，复核记录在 `../03_复核记录/`。本工程不再附重复成品、预测缓存、29组中间实验目录和预览图。

## 使用已有成果

阅读“第四问结果与合理性复核.md”，使用两份XLSX和论文用图。10分钟CSV已无损压缩到 `10分钟完整轨迹.zip`。查看成果无需重算。

在本目录验证精简包与既有结果：

```powershell
python src/verify_delivery.py
python -m unittest discover -s tests -v
```

## 从输入重新生成完整实验

数值环境和精确版本见 `requirements.txt` 与 `../03_复核记录/environment.json`。输入包括四份原始数值CSV和附件5空白模板。数值求解依赖SciPy内置HiGHS。

在本工程根目录执行以下完整顺序。`outputs/`会重新创建，首次预测需要重新训练每日模型；不能跳过prepare直接调用plot或audit。

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
python src/prepare_q4.py
python src/run_all_q4.py --run full_q4_20260913 --workers 3
python src/audit_q4.py
python src/structural_checks.py
python src/prepare_deliverables.py
python src/plot_q4.py
```

这会恢复11组全年方案、热身和参数验证、全部分支日志、CSV载荷和图表。新输出位于本工程 `outputs/full_q4_20260913/`，不会覆盖相邻的论文成果文件夹。更改数值代码或输入后请使用新的run名称，并向后续脚本传入对应的 `--run` 参数。

XLSX导出使用Codex随附Node及 `@oai/artifact-tool`，不打包依赖目录。在Codex环境中连接其依赖后执行：

```powershell
node src/export_workbooks.mjs full_q4_20260913
python src/verify_workbooks.py
python src/report_q4.py
```

普通Python环境可以重算数值、CSV和论文图。两份已经完成的XLSX保留原文件，可直接使用。openpyxl只用于读取与核对，不用于写表。

## 核心入口

- `q4_dispatch.py`：共同物理模型、依赖收缩、条件前瞻及LP净化。
- `q4_forecasting.py`：因果预测、配对残差及信息分组。
- `run_q4.py`：热身、历史选参、逐日回放。
- `audit_q4.py`、`structural_checks.py`：完整重算后的独立复核。
- `verify_delivery.py`：不依赖已归档实验日志的精简包核对。

保留P1/P3及官方预报插值的兼容模块，以维持代码摘要和数值复现一致性。仅移除旧的桌面打包脚本、空白模板预览脚本；数学模型、预测与求解源文件未修改。
