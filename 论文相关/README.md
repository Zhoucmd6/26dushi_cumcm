# C题论文工程

本目录用于维护“微网与外部电网电力调控策略”参赛论文。当前版本已写入有计算证据支撑的第一、二问结果，并完成第三问模型表述；第三问数值结果和第四问模型、结果以显式占位保留，不含虚构数据。

## 目录

- `paper/main.tex`：论文总入口。
- `paper/sections/`：按章节拆分的可编辑 LaTeX 源文件。
- `paper/figures/assets/`：论文使用的 PNG 与对应可编辑 SVG。
- `paper/figures/source/`：论文自行绘制的结构图源文件。
- `supporting-material/AI工具使用详情.tex`：2026 规则要求的 AI 使用详情源稿。
- `output/pdf/`：通过渲染核验的阶段 PDF。
- `写作与验收指南.md`：官方硬性规范、获奖导向和逐章验收表。
- `待补内容清单.md`：后续建模手、编程手需要回填的内容与证据。

## 编译

在本目录执行：

```powershell
xelatex -interaction=nonstopmode -halt-on-error -output-directory build/paper paper/main.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory build/paper paper/main.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory build/ai supporting-material/AI工具使用详情.tex
```

正式提交前须由三名队员逐项核对模型、结果、引用和 AI 使用详情，并将附录中的程序占位替换为完整可运行代码或按赛区要求整理支撑材料。

