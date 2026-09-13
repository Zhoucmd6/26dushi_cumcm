"""科研图表的统一视觉约定；颜色表达变量身份，标记和线型提供第二种区分。"""
from pathlib import Path
import hashlib
import json
import warnings
import textwrap

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.text import Text

STYLE_VERSION = 'academic_v2_20260912'
COLORS = {
    'load': '#0072B2', 'pv': '#E69F00', 'purchase': '#009E73',
    'charge': '#56B4E9', 'discharge': '#D55E00', 'state': '#8C6BB1',
    'price': '#48424C', 'planned_cost': '#4477AA', 'emergency': '#D55E00',
    'ink': '#202630', 'muted': '#62717D', 'rule': '#A6AFB7', 'light': '#EFF3F6',
}
METHOD_COLORS = ['#009E73', '#56B4E9', '#D55E00', '#8C6BB1']


def apply_style():
    for name in ('arial.ttf', 'msyh.ttc'):
        file = Path('C:/Windows/Fonts') / name
        if file.exists():
            font_manager.fontManager.addfont(str(file))
    plt.rcParams.update({
        'font.family': ['Arial', 'Microsoft YaHei', 'DejaVu Sans'],
        'font.size': 9, 'axes.labelsize': 9, 'axes.titlesize': 9.5,
        'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
        'axes.unicode_minus': False, 'axes.spines.top': False,
        'axes.spines.right': False, 'axes.linewidth': .7, 'axes.edgecolor': COLORS['ink'],
        'axes.labelcolor': COLORS['ink'], 'text.color': COLORS['ink'],
        'xtick.color': COLORS['ink'], 'ytick.color': COLORS['ink'],
        'xtick.direction': 'out', 'ytick.direction': 'out',
        'xtick.major.width': .7, 'ytick.major.width': .7,
        'xtick.major.size': 3, 'ytick.major.size': 3, 'axes.grid': False,
        'legend.frameon': False, 'legend.handlelength': 2.2,
        'lines.linewidth': 1.25, 'figure.facecolor': 'white', 'axes.facecolor': 'white',
        'savefig.facecolor': 'white', 'savefig.dpi': 450,
        'svg.fonttype': 'none', 'pdf.fonttype': 42, 'ps.fonttype': 42,
        'svg.hashsalt': STYLE_VERSION,
    })


def step(ax, values, **kwargs):
    """144个区间值映射到0—24时，保留真实阶梯，不插值平滑。"""
    values = np.asarray(values, dtype=float)
    if values.shape != (144,) or not np.isfinite(values).all():
        raise ValueError('A daily step series must have 144 finite values')
    return ax.step(np.arange(145)/6, np.r_[values, values[-1]], where='post', **kwargs)


def time_axis(ax, *, label=True, ticks=4):
    ax.set_xlim(0, 24)
    ax.set_xticks(np.arange(0, 25, ticks))
    if label:
        ax.set_xlabel('时刻 (h)')
    ax.margins(x=0)


def panel(ax, letter, title):
    ax.set_title(title, loc='left', pad=9, fontweight='normal')
    ax.text(-.105, 1.04, letter, transform=ax.transAxes, fontsize=11,
            fontweight='bold', va='bottom', ha='left')


def note(fig, text, *, y=.025):
    text='\n'.join(textwrap.fill(line,width=60,break_long_words=True) for line in text.splitlines())
    fig.text(.12, y, text, fontsize=7.2, color=COLORS['muted'], va='bottom')


def save_figure(fig, folder, name, *, pdf=None, dpi=450):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    # 保存时记录缺字及布局警告，避免仅凭程序未报错便交付。
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        outside = []
        for obj in fig.findobj(match=Text):
            if not obj.get_visible() or not obj.get_text() or obj.get_clip_on():
                continue
            box = obj.get_window_extent(renderer)
            if box.width and box.height and (box.x0 < -1 or box.y0 < -1 or
                    box.x1 > fig.bbox.width+1 or box.y1 > fig.bbox.height+1):
                outside.append(obj.get_text())
        if outside:
            raise ValueError(f'{name}: text extends outside figure: {outside}')
        files = []
        for suffix in ('png', 'svg'):
            target = folder/f'{name}.{suffix}'
            fig.savefig(target, dpi=dpi, facecolor='white')
            files.append({'file': target.name, 'bytes': target.stat().st_size,
                          'sha256': hashlib.sha256(target.read_bytes()).hexdigest()})
        if pdf is not None:
            pdf.savefig(fig, dpi=dpi)
        problems = sorted({str(w.message) for w in caught
                           if 'Glyph' in str(w.message) or 'layout' in str(w.message).lower()})
        if problems:
            raise ValueError(f'{name}: rendering warnings: {problems}')
    result = {'name': name, 'width_mm': round(fig.get_figwidth()*25.4, 1),
              'height_mm': round(fig.get_figheight()*25.4, 1), 'png_dpi': dpi,
              'axes': len(fig.axes), 'text_bounds': 'passed', 'glyph_check': 'passed',
              'files': files}
    plt.close(fig)
    return result
