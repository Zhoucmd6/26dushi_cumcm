# 第三问计算工程

当前按建模手 `model-q3.pdf` 的条件尺度随机滚动模型计算。主方案为B3分箱尺度；B4只在同一个B3上加入CVaR。入口为 `src/run_q3.py`，完整说明见上一级 `01_使用说明/第三问复现与文件指南.md`。

|目录|内容|
|---|---|
|src|校准、历史选参、MILP、诊断、独立复核、报表及绘图|
|tests|20项专用测试，含9项PDF一致性反例|
|configs|本次运行设置、实验注册表和实际选中参数|
|data|原始附件快照与预处理数据，输入文件保持不变|
|reports|数据检查与预处理依据|
|outputs|用新运行名计算时产生的结果，不自动覆盖正式成果|

已交付计算档案在 `../05_复核记录/pdf_aligned_20260912`，当前论文与提交结果在 `../04_正式成果`。`../05_复核记录/q3_20260912` 为旧版历史，不用于当前论文取数。

```powershell
$env:PYTHONIOENCODING='utf-8'
$env:PYTHONDONTWRITEBYTECODE='1'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:MKL_NUM_THREADS='1'
& 'D:\Anaconda\python.exe' src/run_q3.py --run outputs/q3_new_run --workers 3
& 'D:\Anaconda\python.exe' src/run_tests_q3.py outputs/q3_new_run
& 'D:\Anaconda\python.exe' src/q3_diagnostics.py outputs/q3_new_run
& 'D:\Anaconda\python.exe' src/prepare_q3.py outputs/q3_new_run
& 'D:\Anaconda\python.exe' src/summarize_q3.py outputs/q3_new_run
```

数值计算为Python；Excel模板导出调用本机共用 `@oai/artifact-tool`，见复现指南。Python工程不依赖前两问目录。
