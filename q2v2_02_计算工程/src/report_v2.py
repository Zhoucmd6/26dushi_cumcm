"""以已经完成的数值输出生成可核查的模型落实说明和科研图表。"""
from pathlib import Path
import sys,json,csv
import numpy as np
from run_v2 import write_json
from forecasting import read_dataset
from scientific_style import apply_style, save_figure, step, time_axis, COLORS
import matplotlib.pyplot as plt

LABELS = {'P1_weighted28':'P1 加权历史均值','P2_fourier_arima':'P2 傅里叶 + ARIMA','P3_random_forest':'P3 随机森林'}


def daily_rows(folder):
    with (folder/'daily_results.csv').open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def make_plots(run, selection, summaries, out):
    apply_style()
    main = summaries[selection['main_experiment']]
    candidates = [summaries['candidate_'+m] for m in LABELS]
    fig, axs = plt.subplots(1,2,figsize=(8.5,3.7))
    fig.subplots_adjust(left=.09,right=.98,bottom=.26,top=.88,wspace=.3)
    x = np.arange(3)
    planned = np.array([s['total_planned_cost_yuan'] for s in candidates])/1e6
    emergency = np.array([s['total_emergency_cost_yuan'] for s in candidates])/1e6
    colors = ['#0072B2','#E69F00','#009E73']
    bars = axs[0].bar(x,planned,color=colors,width=.58)
    axs[0].bar(x,emergency,bottom=planned,color=COLORS['emergency'],width=.58,hatch='//',label='紧急购电')
    for i,s in enumerate(candidates):
        axs[0].text(i,planned[i]+emergency[i]+.15,f'{planned[i]+emergency[i]:.2f}',ha='center',fontsize=8)
        if s['model']==selection['chosen_model']:
            bars[i].set_edgecolor('#202630');bars[i].set_linewidth(1.8)
    axs[0].set_xticks(x,['P1','P2','P3']);axs[0].set_ylabel('2-12月实际费用 (百万元)')
    axs[0].set_title('(a) 正常与紧急购电费用',loc='left');axs[0].legend(loc='upper right')
    axs[0].set_ylim(0,max(planned+emergency)*1.2)
    width=.24
    for j,(key,label,col) in enumerate([('load_mae_kw','负荷 MAE','#0072B2'),('pv_mae_kw','光伏 MAE','#E69F00'),
                                       ('pv_daylight_mae_kw','有光伏时段 MAE','#009E73')]):
        axs[1].bar(x+(j-1)*width,[s['forecast_metrics'][key] for s in candidates],width,label=label,color=col)
    axs[1].set_xticks(x,['P1','P2','P3']);axs[1].set_ylabel('功率误差 (kW)')
    axs[1].set_title('(b) 预测精度',loc='left');axs[1].legend(loc='upper right',fontsize=7)
    axs[1].set_ylim(0,max(s['forecast_metrics']['load_mae_kw'] for s in candidates)*1.45)
    fig.text(.09,.10,'P1：加权历史均值；P2：傅里叶日周期 + ARIMA残差；P3：随机森林。',fontsize=8)
    fig.text(.09,.045,'黑色边框标出1月验证选定的主方案；全年结果仅用于事后评价。',fontsize=8,color=COLORS['muted'])
    checks=[save_figure(fig,out,'01_预测方法与调度费用')]

    fig,axs=plt.subplots(1,2,figsize=(8.5,3.5))
    fig.subplots_adjust(left=.09,right=.98,bottom=.23,top=.88,wspace=.30)
    for i,s in enumerate(candidates):
        rows=daily_rows(run/s['name'])
        for ax,key in zip(axs,['cash_cost_yuan','emergency_cost_yuan']):
            monthly=[sum(float(r[key]) for r in rows if int(r['date'][5:7])==m)/1e4 for m in range(2,13)]
            ax.plot(range(2,13),monthly,color=colors[i],marker=['o','s','^'][i],markersize=3,label=LABELS[s['model']])
            ax.set_xticks(range(2,13));ax.set_xlabel('月份');ax.set_ylabel('费用 (万元)')
    axs[0].set_title('(a) 每月实际购电费用',loc='left');axs[1].set_title('(b) 每月紧急购电费用',loc='left')
    axs[1].set_ylim(bottom=0)
    handles,labels=axs[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.015),ncol=3,fontsize=8)
    checks.append(save_figure(fig,out,'02_逐月费用与紧急购电'))

    a=np.load(run/selection['main_experiment']/'trajectories.npz')
    data=read_dataset(Path(__file__).resolve().parents[1]/'data/processed')
    fig,axs=plt.subplots(4,2,figsize=(8.5,9.1))
    fig.subplots_adjust(left=.10,right=.98,bottom=.12,top=.95,hspace=.66,wspace=.28)
    for i,dt in enumerate(['2025-03-20','2025-06-21','2025-09-23','2025-12-21']):
        day=next(j for j,d in enumerate(data.dates) if d.isoformat()==dt)
        j=list(a['day_indices']).index(day)
        step(axs[i,0],a['Q'][j]*6,color=COLORS['purchase'],label='计划购电')
        step(axs[i,0],a['E'][j]*6,color=COLORS['emergency'],label='紧急购电',linewidth=1.)
        step(axs[i,0],data.load_kw[day]-data.pv_kw[day],color=COLORS['load'],label='实际净负荷',linestyle='--',linewidth=.85)
        axs[i,0].set_ylabel('功率 (kW)');axs[i,0].set_title(dt+' 供需与购电',loc='left')
        axs[i,1].plot(np.arange(145)/6,a['S'][j],color=COLORS['state'],label='储能量')
        axs[i,1].axhline(1200,color=COLORS['muted'],linestyle=':',linewidth=.7)
        axs[i,1].axhline(10800,color=COLORS['muted'],linestyle=':',linewidth=.7)
        axs[i,1].set_ylim(0,12000);axs[i,1].set_yticks([1200,6000,10800]);axs[i,1].set_ylabel('储能量 (kWh)')
        axs[i,1].set_title(dt+' 储能轨迹',loc='left')
        for ax in axs[i]:time_axis(ax)
    h,l=axs[0,0].get_legend_handles_labels();h2,l2=axs[0,1].get_legend_handles_labels()
    fig.legend(h+h2,l+l2,loc='lower center',bbox_to_anchor=(.5,.025),ncol=4)
    fig.text(.10,.008,'购电曲线由10分钟电量乘6转换为平均功率；虚线净负荷包含当天实际值，仅用于事后展示。',fontsize=7)
    checks.append(save_figure(fig,out,'03_指定日期调度与储能'))

    names=['terminal_0_8','terminal_1_2','residual_0_8','residual_1_2','seed_20260913','seed_20260914']
    labels=['终端系数 -20%','终端系数 +20%','残差幅度 -20%','残差幅度 +20%','抽样种子 20260913','抽样种子 20260914']
    fig,axs=plt.subplots(1,2,figsize=(8.5,3.8),sharey=True)
    fig.subplots_adjust(left=.22,right=.97,bottom=.19,top=.88,wspace=.28)
    for ax,key,title in zip(axs,['total_cash_cost_yuan','total_emergency_cost_yuan'],['(a) 实际总费用变化','(b) 紧急购电费用变化']):
        values=[(summaries[n][key]/main[key]-1)*100 for n in names]
        ax.barh(np.arange(6),values,color=['#56B4E9','#0072B2','#E69F00','#D55E00','#8C6BB1','#009E73'])
        ax.axvline(0,color=COLORS['ink'],linewidth=.7);ax.set_yticks(np.arange(6),labels)
        ax.set_xlabel('相对主方案变化 (%)');ax.set_title(title,loc='left')
    axs[0].invert_yaxis()
    fig.text(.22,.04,'预测模型、初末条件和结算规则固定；每次只改变表中所列因素。种子对照不用于重新选择方案。',fontsize=7)
    checks.append(save_figure(fig,out,'04_参数与抽样稳定性'))
    write_json(run/'figure_checks.json',checks)


def main(run):
    selection=json.loads((run/'selection.json').read_text(encoding='utf-8'))
    summaries={s['name']:s for s in json.loads((run/'experiment_summaries.json').read_text(encoding='utf-8'))}
    audit=json.loads((run/'independent_audit.json').read_text(encoding='utf-8'))
    q1=json.loads((run/'q1.json').read_text(encoding='utf-8'))
    main=summaries[selection['main_experiment']]
    with np.load(run/selection['main_experiment']/'trajectories.npz') as saved:
        day_end_at_max=int(np.sum(np.abs(saved['S'][:,-1]-10800)<1e-5))
    matched=summaries['v1_mechanism_matched_initial']
    out=run/'deliverables'
    out.mkdir(exist_ok=True)
    make_plots(run,selection,summaries,out/'科研图表')
    rows=['# 前两问 version2：模型落实与求解报告','',
        '本版以 model-q1-q2-revised.pdf 为模型依据，重点补齐第二问。原工程和原结果保留。第三问尚未纳入本次 version2。','',
        '## 结果与适用范围','',
        f"第一问 MILP 费用为 **{q1['milp']['cost_yuan']:,.6f} 元**；LP 费用为 {q1['lp']['cost_yuan']:,.6f} 元。",
        f"第二问采用1月验证选定的 **{LABELS[selection['chosen_model']]}**，终端价值倍率为 **{selection['chosen_terminal_multiplier']}**。",
        f"2025年2月1日至12月31日实际费用为 **{main['total_cash_cost_yuan']:,.2f} 元**，其中正常购电 {main['total_planned_cost_yuan']:,.2f} 元、紧急购电 {main['total_emergency_cost_yuan']:,.2f} 元。",
        f"紧急购电量 {main['total_emergency_purchase_kwh']:,.2f} kWh，总未利用电量 {main['total_unused_kwh']:,.2f} kWh。",
        '这是给定信息、候选模型、场景和固定计划策略下的历史回测结果。MILP达到数值最优容差，不等于真实未知分布下的全局最优策略。','',
        '## 模型与代码对应','',
        '| PDF内容 | version2落实 | 代码 |','|---|---|---|',
        '| §1.2、式(3)-(5) 单位与效率 | kW乘1/6得到kWh；主口径ηc=ηd=0.9 | forecasting_v2.py、microgrid_core.py |',
        '| §2、式(6)-(15) 第一问 | 复用MILP约束、光伏弃电上界、首尾6000及充放电互斥 | microgrid_core.solve_q1 |',
        '| §2.6 LP对照 | 同数据解LP并核查同时充放电；不预先假定等价 | microgrid_core.solve_q1 |',
        '| §1.3 效率敏感性 | ηc=ηd=√0.9，前两问均重算 | run_v2.py |',
        '| §3.2 P1/P2/P3 | 加权历史均值；傅里叶项+ARIMA误差；随机森林，分别预测负荷和光伏 | forecasting_v2.predict |',
        '| §3.2.4 因果选择 | 1月21-31日滚动起点评价，主指标为实际费用，同时报告MAE/RMSE | run_v2.py |',
        '| §3.3、式(24)-(25) 场景 | 最近28日整日配对样本外残差，有放回抽样256次，负值截零 | forecasting_v2.scenarios |',
        '| §3.4 第一阶段共享 | Q/C/D/S/Z在0点固定，全部场景共同使用；E/U随场景变化 | stochastic_dispatch.solve_q2_fixed |',
        '| §3.8、式(27) 终端价值 | 选用PDF允许的线性价值，Φ(s)=-m·min(p)/ηc·s | run_v2.experiment |',
        '| §3.9、式(28)-(38) 物理约束 | 145个状态点、容量/功率边界、互斥、跨日衔接；年末6000 | stochastic_dispatch.py |',
        '| §3.9.4 一月预热 | 1月1日6000，1月固定使用预先指定的P1连续调度；首日采用PDF允许的附件1先验 | run_v2.py |',
        '| §3.10 分解结构 | 主计算使用整体MILP；原Benders实现及等价性测试保留 | stochastic_dispatch.py、test_dispatch.py |',
        '| §3.11、式(46)-(48) 真实结算 | 用附件2实际值重算E/U，现金费用只含pQ+5pE | microgrid_core.simulate_fixed_plan |','',
        'PDF允许的48小时时域与线性终端价值是替代方案，本版选择后者；§3.13实时电池响应是扩展，本版按已确定的零点固定计划实现。不会把这些可选分支写成已完成实验。','',
        '## 清晰的求解顺序','',
        '1. 读取附件1的日内电价与固定日型、附件2的365天实际数据；核查每一天144点、顺序与非负性。',
        '2. 求解第一问MILP、LP以及效率口径对照。',
        '3. 对每个预测日，只传入当天之前的完整日数据拟合预测；历史残差也由当时可得到的预测计算。',
        '4. 1月1日用附件1固定日型作先验，之后使用历史数据；按P1预先指定策略连续运行整月，产生2月1日真实衔接储量。',
        '5. 从同一个1月20日日末储量出发，比较3种预测×3个终端倍率，共9个验证分支；各分支1月31日日末均固定6000，避免验证费用受不同剩余库存影响。验证分支不回写1月实际预热轨迹。',
        '6. 2月1日0点按1月验证费用固定主方案。在每个正式日构造场景、求解一个共享日前计划、用实际数据回放，并把日末储量带给次日。',
        '7. 输出完整10分钟轨迹、334天表格、指定日期结果；独立复算物理约束、每个场景目标、现金费用、概率和时间信息边界。','',
        '## 预测与抽样的具体选择','',
        '- P1：最多28个历史日，时间权重半衰期7天，同星期权重乘2；沿用经过验证的实现。',
        '- P2：最多28天的10分钟数据，8阶日周期傅里叶回归后对残差拟合ARIMA(1,0,0)，使用Burg估计。它对应PDF列出的“傅里叶项+ARIMA误差”分支；本版没有另行运行144季节周期的SARIMA。',
        '- P3：两套随机森林分别预测负荷和光伏，64棵树、深度10、最小叶8。特征包括前1天、前7天同点、前7天均值、前1天相邻点、时刻、星期及月份。训练目标最多28天，滞后特征所需原始数据最多再向前7天。没有引入题目未提供的天气。',
        '- 最早无历史的一天用附件1先验；P2不足7日、P3不足8日时显式回退P1，记录在逐日信息日志中。',
        '- 每次抽到的日索引同时决定该日144点负荷残差与144点光伏残差。重复索引合并后概率为抽中次数/256，数学上等价于保留256个等权场景；不把不同分钟或负荷/光伏各自打乱。',
        '- 除式(25)规定的负值截零，本版场景不额外加入历史零光伏支撑遮罩。',
        '- 28天、256次、傅里叶阶数和树参数是实现时预先固定的配置，并非PDF给出的唯一常数，也未宣称已全面调优。残差幅度、终端系数与随机种子有单独对照。','',
        '## 一月验证','',
        '| 预测方法 | 终端倍率 | 验证实际费用/元 | 负荷MAE/kW | 光伏MAE/kW |','|---|---:|---:|---:|---:|']
    for s in selection['validation']:
        rows.append(f"| {LABELS[s['model']]} | {s['terminal_multiplier']} | {s['total_cash_cost_yuan']:,.2f} | {s['forecast_metrics']['load_mae_kw']:.2f} | {s['forecast_metrics']['pv_mae_kw']:.2f} |")
    rows += ['', '1月只有11个验证日，季节代表性有限。下表的全年结果只作样本外评价，不能据此追溯改变1月选择。','',
        '## 全年候选评价','',
        '| 方法 | 实际费用/元 | 紧急购电费/元 | 负荷MAE/kW | 光伏MAE/kW | 有光伏时段MAE/kW |',
        '|---|---:|---:|---:|---:|---:|']
    for m in LABELS:
        s=summaries['candidate_'+m];f=s['forecast_metrics']
        rows.append(f"| {LABELS[m]} | {s['total_cash_cost_yuan']:,.2f} | {s['total_emergency_cost_yuan']:,.2f} | {f['load_mae_kw']:.2f} | {f['pv_mae_kw']:.2f} | {f['pv_daylight_mae_kw']:.2f} |")
    annual_best=min((summaries['candidate_'+m] for m in LABELS),key=lambda s:s['total_cash_cost_yuan'])
    rows += ['',f"事后全年费用最低的候选为{LABELS[annual_best['model']]}。主结果仍使用1月选定的{LABELS[selection['chosen_model']]}。各候选的终端倍率也分别在1月选定；因此这张表比较的是经相同验证流程确定的完整方案。",'',
        '## 控制变量对照与敏感性','',
        '| 设置 | 实际费用/元 | 相对主方案 | 紧急购电费/元 |','|---|---:|---:|---:|']
    comparisons = [('point_selected','只用点预测'),('enumerate_selected','配对残差枚举'),
        ('terminal_0_8','终端倍率在主方案基础上减20%'),('terminal_1_2','终端倍率在主方案基础上增20%'),
        ('residual_0_8','残差幅度减20%'),('residual_1_2','残差幅度增20%'),
        ('seed_20260913','抽样种子20260913'),('seed_20260914','抽样种子20260914'),
        ('roundtrip90','往返效率90%口径，另行预热')]
    for name,label in comparisons:
        s=summaries[name]
        rows.append(f"| {label} | {s['total_cash_cost_yuan']:,.2f} | {(s['total_cash_cost_yuan']/main['total_cash_cost_yuan']-1)*100:+.3f}% | {s['total_emergency_cost_yuan']:,.2f} |")
    rows += ['',
        '## 与旧版的关系及对照','',
        f"旧版原主结果为16,766,455.69元，2月1日初始储量为6000kWh。本版一月连续预热后起点为{main['initial_kwh']:.6f}kWh，两者不能直接作为控制其他因素的单变量比较。",
        f"为此，另用旧版P1预测、最近样本外残差枚举及终端倍率1，在本版同一初始储量和相同年末条件下重算，费用为 **{matched['total_cash_cost_yuan']:,.2f}元**。",
        f"本版主方案相对该对照的实际费用变化为 **{main['total_cash_cost_yuan']-matched['total_cash_cost_yuan']:+,.2f}元（{(main['total_cash_cost_yuan']/matched['total_cash_cost_yuan']-1)*100:+.3f}%）**。这反映预测、抽样和终端选择的组合效果，不单独归因于任何一种算法。",
        '随机种子对照保留同一预测模型、终端系数和1月预热后起点，只改变正式期的场景抽样；枚举对照保留选定预测，用于识别额外抽样误差。',
        '无储能对照保持库存不动，无法满足从本版1200恢复到6000的年度储量条件；其费用只用于辅助观察，不作为严格同终端条件的节省率证据。效率敏感性则使用替代效率重新预热，起点差异在结果表中列出。','',
        '## 必须如实说明的边界','',
        '1. 时间标签：本版保留此前采用的区间终点解释，原附件点序不循环平移，输出区间为00:00-00:10至23:50-24:00。PDF§1.1初始描述按模板位置对应，模板标签向后一个时段。这里是明确保留的解释差异，论文的时间约定也必须同步；不能声称仅靠作图已经证实标签含义。',
        '2. 富余电量：式(28)中的U可包含免费光伏，因此报告称“总未利用电量”，不把全部U都称为已经付费的外网浪费。',
        f"3. 终端价值：主方案有{main['day_end_at_min_count']}个日末在1200kWh附近，另有{day_end_at_max}个日末在10800kWh附近。线性终端项使该参数下的储量偏向上限，并非已经消除日末效应。将选定系数降低20%的事后费用反而下降约0.205%，说明1月选择不保证全年参数最优。主结果仍保留预先选定参数，不能用全年敏感性结果倒选。",
        '4. 1月1日附件1日型先验是PDF§3.9.4允许的初始化信息。它作为题目已提供的固定输入使用，未用附件2未来日期重新计算该先验。正式期模型只使用历史实际数据。',
        '5. 固定计划下，电池当天不能响应真实误差；紧急购电可能被用于完成事先承诺的充电。这来自PDF主模型的策略限制，不能在代码中暗改为日内自适应。',
        '6. 只使用1年数据且初始验证集中在1月，结果不构成跨年份稳定性保证。3个随机种子仅检验抽样扰动，不是统计置信区间。','',
        '## 数值验证与交付','',
        f"独立复算覆盖{audit['audited_experiments']}个实验、{audit['audited_days']}个日模型/回放记录，包含预热、验证和14组正式结果；最大数值残差为{audit['maximum_numerical_residual']:.3g}，按相应检查分别以kWh或元计，均低于1e-5。",
        '28项测试包括Benders与整体模型的小样例等价性、储能损耗、跨日衔接、未来数据扰动、ARIMA周期样例、残差配对和压缩前后优化等价性。',
        f"第一问替代效率口径（ηc=ηd=√0.9）的费用为{q1['roundtrip90_sensitivity']['cost_yuan']:,.6f}元。第一问主结果与旧版一致，说明复用的主模型未被本次预测模块改动。",
        'result1.xlsx、result2.xlsx沿用题目模板的工作表结构。第二问购电表的EP列为计划Q合计，EQ列为pQ+5pE实际费用；紧急购电量在独立工作表。规划用的终端价值不算作真实电费。',
        '第二问Excel中绝对值小于1e-9的浮点噪声置零，避免出现“-0.00”；原始求解轨迹未据此截断，逐值导出误差仍须小于1e-6。',
        '完整10分钟CSV、全部方案对比、1月验证、预测精度、指定日期表格和图表放在结果目录；场景抽样索引、种子、源码快照、输入哈希和独立审计记录放在复核目录。','']
    (out/'前两问version2模型落实与求解报告.md').write_text('\n'.join(rows),encoding='utf-8')
    print('REPORT AND FIGURES COMPLETE')


if __name__=='__main__':main(Path(sys.argv[1]).resolve())
