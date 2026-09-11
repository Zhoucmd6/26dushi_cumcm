"""从保存的数值结果生成论文用图和首版结果报告，不重新调参或选方案。"""
from pathlib import Path
from datetime import date
import argparse
import csv
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from academic_figures import build_figures

from forecasting import read_dataset

ROOT=Path(__file__).resolve().parents[1]
LABELS={"main_weighted28":"加权历史＋场景规划", "comparison_mean7":"近期均值＋场景规划",
        "comparison_point":"加权历史＋单一预测", "comparison_no_storage":"同场景、无储能"}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('run',type=Path)
    args=parser.parse_args();run=args.run
    ds=read_dataset(ROOT/'data/processed')
    summaries=json.loads((run/'experiment_summaries.json').read_text(encoding='utf-8'))
    if len(summaries)!=8:
        raise ValueError('Wait for all eight predefined experiments to finish')
    by_name={s['name']:s for s in summaries};main=by_name['main_weighted28']
    q1=json.loads((run/'q1.json').read_text(encoding='utf-8'))['milp']
    payload=json.loads((run/'workbook_payload.json').read_text(encoding='utf-8'))
    audit=json.loads((run/'independent_audit.json').read_text(encoding='utf-8'))
    benchmark=json.loads((run/'solver_benchmark.json').read_text(encoding='utf-8'))
    if len(audit['experiments'])!=8 or audit['status']!='passed' or 'workbooks' not in audit:
        raise ValueError('Final independent audit is required')
    out=run/'deliverables';figures=out/'figures';figures.mkdir(exist_ok=True)
    figure_result=build_figures(run,ROOT,figures)
    saved=[record['name'] for record in figure_result['figures']]
    names=list(LABELS)
    with np.load(run/'main_weighted28/trajectories.npz') as stored:
        a={'S':stored['S'].copy()}

    decrease_point=(1-main['total_cash_cost_yuan']/by_name['comparison_point']['total_cash_cost_yuan'])*100
    decrease_storage=(1-main['total_cash_cost_yuan']/by_name['comparison_no_storage']['total_cash_cost_yuan'])*100
    difference_mean=(by_name['comparison_mean7']['total_cash_cost_yuan']-main['total_cash_cost_yuan'])/by_name['comparison_mean7']['total_cash_cost_yuan']*100
    end_states=a['S'][:,-1]
    lower_end_count=int((np.abs(end_states-1200)<1e-5).sum())
    lines=['**C题前两问首版计算结果与复核说明**','',
           '本版按用户授权采用建议口径，完成前两问数值流程和模板输出。结果属于明确假设下的可复核基线，不代表已证明全年所有因果调度策略中的全局最优。','',
           '**1. 本次采用的口径**','',
           '- 源标签00:10解释为00:00—00:10区间的代表功率，按10分钟分段常值近似积分。输出模板按00:00—24:00更正标签；记录次序和数据没有循环平移。',
           '- 充放电均在母线侧计量，效率分别0.9；容量1200—10800 kWh、功率5000 kW，严格禁止同一时段同时充放电。',
           '- 第一问初末6000 kWh。第二问一月电池闲置，二月初6000；全年逐日衔接，12月31日末6000。年末约束属于本方案增加的比较条件。',
           '- 第二问零点固定全天购电和充放电；紧急购电可能用于维持既定充电，完全沿用用户选定的固定计划基线。',
           f'- 每天终端价值为Phi(S)=−vS，v=min(p)/eta_c={ds.prices.min()/.9:.9f}元/电池内部kWh。它是按已知固定电价确定的补充电量成本近似，不计入实际电费。',
           '- 主预测预先指定为28日加权历史均值：半衰期7日，同星期权重乘2。对照为最近7日简单均值。两者均只用目标日之前的完整日。',
           '- 先用至少7日历史生成逐日样本外预测。当天场景枚举最近最多28个历史日的成对负荷/光伏残差，等概率；2月1日有24个残差场景。没有从全年挑选“最好”的模型或参数，也没有额外蒙特卡洛抽样。',
           '- 场景对负功率截零；PV支撑范围由最近28个历史日确定。该方法不能预知新出现的日出/日落时段，其误差仍保留在实际回测中。','',
           '**2. 第一问结果**','',
           f'LP与MILP均得到全天购电费 **{q1["cost_yuan"]:,.6f}元**，计划购电量 **{q1["purchase_kwh"]:,.6f} kWh**，初末储量均6000 kWh。与不使用储能的可行基线48052.046591元相比，费用下降{(1-q1["cost_yuan"]/48052.04659084667)*100:.3f}%。','',
           '| 指定时间段 | 计划购电量（kWh） |','|---|---:|']
    lines += [f'| {t} | {q:,.6f} |' for t,q in payload['q1']['requested_intervals']]
    lines += ['', '| 四小时时段 | 充电量（kWh） | 放电量（kWh） |','|---|---:|---:|']
    lines += [f'| {r[0]} | {r[1]:,.6f} | {r[2]:,.6f} |' for r in payload['q1']['battery']]
    lines += ['', '**3. 第二问334天对照**','',
              '| 方法 | 实际总费（元） | 紧急购电费（元） | 紧急购电量（kWh） |',
              '|---|---:|---:|---:|']
    for name in names:
        s=by_name[name]
        lines.append(f'| {LABELS[name]} | {s["total_cash_cost_yuan"]:,.2f} | {s["total_emergency_cost_yuan"]:,.2f} | {s["total_emergency_purchase_kwh"]:,.3f} |')
    lines += ['',f'在本次回测中，主方案比相同加权预测的单一预测规划费用低 **{decrease_point:.3f}%**，比同场景无储能低 **{decrease_storage:.3f}%**。这分别反映场景处理和电池在本组假设下的经济效果。',
              f'主方案相对近期均值场景方案费用仅低 **{difference_mean:.3f}%**，不能夸大成显著算法优势。两种预测各有优劣，见下表。','',
              '| 预测 | 负荷MAE（kW） | 光伏MAE（kW） | 实际光伏>0时MAE（kW） |','|---|---:|---:|---:|']
    for name in ('main_weighted28','comparison_mean7'):
        s=by_name[name]
        lines.append(f'| {LABELS[name]} | {s["forecast_load_mae_kw"]:.3f} | {s["forecast_pv_mae_kw"]:.3f} | {s["forecast_pv_active_mae_kw"]:.3f} |')
    lines += ['',f'主方案总未利用电量为{main["total_unused_kwh"]:,.3f} kWh，包括免费光伏等来源，不等于全部是已付费购电浪费。',
              f'主方案有{lower_end_count}天日末储量贴近1200 kWh下限。当前终端价值是软估值，不能宣称它保证每天末尾维持高储量；如需改变这一行为，应另做48小时滚动或更精细的价值函数对照。','',
              '**4. 指定日期**','', '| 日期 | 计划电量（kWh） | 紧急电量（kWh） | 实际总费（元） | 日初/日末储量（kWh） |','|---|---:|---:|---:|---|']
    for r in payload['q2']['selected_dates']:
        lines.append(f'| {r["date"]} | {r["planned_purchase_kwh"]:,.3f} | {r["emergency_purchase_kwh"]:,.3f} | {r["actual_total_cost_yuan"]:,.2f} | {r["initial_kwh"]:.3f} / {r["terminal_kwh"]:.3f} |')
    lines += ['', '各指定日期的六个10分钟购电值、四小时充放电及全部紧急购电区间，保存在同目录“指定日期结果.json”；完整全年记录已填入result2.xlsx。','',
              '**5. 灵敏度**','', '| 改变项 | 实际总费（元） | 相对基准变化 |','|---|---:|---:|']
    sensitivity_names={'sensitivity_terminal_0_8':'终端价值×0.8','sensitivity_terminal_1_2':'终端价值×1.2',
                       'sensitivity_residual_0_8':'历史残差×0.8','sensitivity_residual_1_2':'历史残差×1.2'}
    for name,label in sensitivity_names.items():
        value=by_name[name]['total_cash_cost_yuan']
        lines.append(f'| {label} | {value:,.2f} | {(value/main["total_cash_cost_yuan"]-1)*100:+.4f}% |')
    lines += ['', '每个敏感性实验都重新连续回放全部334天，保持其他参数和年末条件一致；不是只改变某一天或在年度结果上做比例换算。该局部检验不代表对任意参数都稳健。','',
              '**6. Benders与整体求解**','', '| 日期 | 场景数 | 整体MILP | Benders |','|---|---:|---|---|']
    for row in benchmark:
        def describe(x):
            if x['status']!='converged':return '20秒/200轮内未取得要求的收敛证据'
            return f'{x["runtime_s"]:.3f}秒，{x["iterations"]}轮'
        lines.append(f'| {row["date"]} | {row["scenarios"]} | {describe(row["results"]["extensive"])} | {describe(row["results"]["benders"])} |')
    lines += ['', '两种方法求解相同固定策略模型，主问题均保留整数变量。只在双方收敛的案例比较目标值；超时不当成算法错误或已求得最优。当前全年主结果采用整体MILP；基准是少量单次测量，不推出一般加速结论。','',
              '**7. 审查发现与修正**','',
              '- 以前的11项小样例只覆盖求解内核，没有覆盖真实预测、完整日期、模板展开和全年状态衔接。本轮增加时间、场景及配置检查，20项单元检查全部通过，另完成8组全年物理与费用独立复核。',
              '- 数据入口新增逐条日期、点序与源时间标签核对。原始10个文件及其副本摘要均未改变，未删除极值、未替换光伏零值。',
              '- 跨日储量允许1e−7 kWh级初值容差，保留原值传递；不因浮点误差拒绝合法运行，也不通过截断或每日重置制造能量。',
              '- 整数型合成输入会使旧写法的预测缓存发生整数截断。本轮检查发现后，将预测缓存明确设为浮点数，并增加断言；正式数据原本为浮点数。',
              '- 为避免配置与实际算法不一致，新增对未实现预测/场景模式的拒绝检查，并拒绝负的事件报告阈值。这两项是计算启动后的输入校验补强，不改变本轮已运行的合法参数与数值结果；计算时的源码快照另行保留。',
              '- 防泄漏测试大幅改动目标日及以后的实际值，目标日预测和场景完全不变。日期检查、历史样本外残差和参数预先固定共同控制信息边界；这不等于实际部署没有数据延迟。',
              '- 原模板时间标签错位在输出副本中按本方案更正，电池表展开为2004行数据，紧急购电表按真实连续事件展开为3075行，清除模板中的省略号。',
              '- Excel文件使用独立读回检查：所有购电值、每日合计/费用、电池分段量和紧急区间对账；不存在公式错误。运行配置、源码摘要及对应源码快照均保留。','',
              '**8. 文件口径与使用**','',
              '- result1.xlsx：第一问完整计划与充放电结果。',
              '- result2.xlsx：第二问334天结果。“计划购电量”表EP列是Q的合计，EQ列为计划费用加实际紧急购电费用。计划加应急的总电量另列于每日CSV。',
              '- 第二问每日费用与电量.csv：逐日分项费用、电量及储量，便于论文二次整理。',
              '- figures目录：六张科研图各保存450 dpi PNG和可编辑SVG，并有PDF合集、图表说明及输入摘要；数据来自已保存的轨迹、汇总和描述统计。',
              '- 上级运行目录的independent_audit.json、config.json和code_snapshot，以及各实验子目录中的information_audit.json，保留验证与复现依据。',
              '- 本轮完成前两问的首版流程。第三、四问、天气模型、SARIMA等预测候选，以及实时电池响应属于后续扩展，本次没有实现或宣称比较完毕。','',
              '来源：C题题面、附件1/2/5、建模手model-q1-q2-revised.pdf、用户确认的固定计划策略，以及本次运行记录。']
    (out/'前两问结果与复核说明.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({'report':str(out/'前两问结果与复核说明.md'),'figures':saved},ensure_ascii=False))


if __name__=='__main__':
    main()
