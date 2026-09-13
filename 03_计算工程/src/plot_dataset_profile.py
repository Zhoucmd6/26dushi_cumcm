"""读取已有附件统计，使用统一科研样式生成数据概览，不参与日前预测。"""
from pathlib import Path
from academic_figures import plot_data_overview
from scientific_style import apply_style, save_figure

ROOT=Path(__file__).resolve().parents[1]


def main():
    apply_style()
    output=ROOT/'outputs/figures'
    save_figure(plot_data_overview(ROOT),output,'C题数据概览')
    print(output/'C题数据概览.png')


if __name__=='__main__':
    main()
