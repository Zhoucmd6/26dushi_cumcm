# C题论文工程

本目录用于维护“微网与外部电网电力调控策略”参赛论文。当前终稿已写入四问全部模型、样本外结果、消融、指定日期、统计区间和数值审计；正文不使用工程版本名，所有数值均来自按提交时间确认的正式成果。

## 目录

- `paper/main.tex`：论文总入口。
- `paper/sections/`：按章节拆分的可编辑 LaTeX 源文件。
- `paper/figures/assets/`：论文使用的 PNG 与对应可编辑 SVG。
- `paper/figures/source/`：论文自行绘制的结构图源文件。
- `supporting-material/AI工具使用详情.tex`：2026 规则要求的 AI 使用详情源稿。
- `output/pdf/C题微网电力调控策略_最终稿.pdf`：通过逐页渲染核验的终稿，正文30页、附录从第31页开始。
- `output/pdf/AI工具使用详情.pdf`：按2026规则单独提交的AI使用说明。
- `写作与验收指南.md`：官方硬性规范、获奖导向和逐章验收表。
- `待补内容清单.md`：后续建模手、编程手需要回填的内容与证据。

## 编译

在本目录执行：

```powershell
New-Item -ItemType Directory -Force output/pdf
xelatex -interaction=nonstopmode -halt-on-error -output-directory output/pdf -jobname C题微网电力调控策略_最终稿 paper/main.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory output/pdf -jobname C题微网电力调控策略_最终稿 paper/main.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory output/pdf -jobname AI工具使用详情 supporting-material/AI工具使用详情.tex
```

终稿附录已编入建模所用完整源程序。正式提交前仍须由三名队员逐项核对模型、结果、引用、附件5和AI使用详情，并按官方提交系统要求整理支撑材料。
