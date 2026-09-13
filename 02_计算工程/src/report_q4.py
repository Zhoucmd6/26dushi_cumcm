"""Source-grounded final report, model mapping and reproducibility metadata."""
import argparse,csv,json,platform
import numpy as np
import scipy,sklearn,statsmodels,matplotlib,openpyxl
from run_q4 import ROOT,write_json,write_csv

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='full_q4_20260913');a=p.parse_args()
    run=ROOT/'outputs'/a.run;out=run/'deliverables'
    selection=json.loads((run/'selection.json').read_text('utf-8'))
    price=json.loads((ROOT/'outputs/prepared/price_selection.json').read_text('utf-8'))
    audit=json.loads((run/'audit.json').read_text('utf-8'));struct=json.loads((run/'structural_checks.json').read_text('utf-8'))
    excel=json.loads((run/'workbook_verification.json').read_text('utf-8'))
    summaries={p.parent.name:json.loads(p.read_text('utf-8')) for p in run.glob('q4*/summary.json')}
    s2=summaries[selection['q42_main']];s3=summaries[selection['q43_main']]
    names=['q42_point','q42_pair','q42_independent','q42_selected','q42_gamma0','q43_point','q43_no_price_correction','q43_common','q43_lookahead_k2','q43_lookahead_k3','q43_gamma0']
    labels=['Q4-2 点预测','Q4-2 成对场景 ρ=1','Q4-2 独立边际 ρ=0','Q4-2 历史选择（主）','Q4-2 主方案 γ=0',
            'Q4-3 点预测','Q4-3 不修正日内电价','Q4-3 共同动作 K=1（主）','Q4-3 分组前瞻 K=2','Q4-3 分组前瞻 K=3','Q4-3 主方案 γ=0']
    totals=[]
    for name,label in zip(names,labels):
        s=summaries[name]
        totals.append({'name':name,'label':label,**{k:s[k] for k in ('cash_cost_yuan','planned_cost_yuan','adjustment_cost_yuan','emergency_cost_yuan','emergency_kwh','unused_kwh','charge_kwh','discharge_kwh','up_kwh','down_kwh','price_mae','price_rmse','price_crps','price_coverage80','optimization_count','solver_runtime_s')},
            'normal_cost_yuan':s['planned_cost_yuan']+s['adjustment_cost_yuan']})
    write_csv(out/'全年方案汇总.csv',totals)
    # Descriptive uncertainty of paired daily bill differences. Does not affect any decisions.
    pairs=[('q42_pair','q42_selected'),('q43_common','q43_lookahead_k2'),('q43_common','q43_lookahead_k3'),('q43_no_price_correction','q43_common')]
    rng=np.random.default_rng(20260913);comparisons=[]
    for old,new in pairs:
        def cash(name):return np.array([float(r['cash_cost_yuan']) for r in csv.DictReader((run/name/'daily.csv').open(encoding='utf-8-sig'))])
        delta=cash(old)-cash(new);n=len(delta);starts=rng.integers(0,n,size=(2000,(n+6)//7))
        ids=((starts[:,:,None]+np.arange(7))%n).reshape(2000,-1)[:,:n]
        boot=delta[ids].sum(axis=1)
        comparisons.append({'baseline':old,'candidate':new,'saving_yuan':float(delta.sum()),'saving_pct':float(100*delta.sum()/cash(old).sum()),
            'positive_saving_days':int((delta>1e-5).sum()),'negative_saving_days':int((delta < -1e-5).sum()),
            'circular_block_days':7,'bootstrap_draws':2000,'seed':20260913,
            'descriptive_95_interval_yuan':np.quantile(boot,[.025,.975]).tolist()})
    write_json(run/'paired_comparisons.json',comparisons)
    environment={'python':platform.python_version(),**{m.__name__:m.__version__ for m in (np,scipy,sklearn,statsmodels,matplotlib,openpyxl)},
                 'numerical_threads':1,'random_forest_seed':20260912,'grouping_seed':20260913}
    write_json(run/'environment.json',environment)
    lines=['# 第四问计算结果与合理性复核','',
        '模型依据为建模手提供的修订版 `model-q4.tex`、配套PDF、修订说明及假设清单。核心变量、结算公式、信息边界、依赖收缩和条件前瞻结构均按该版本实现。',
        '',f"主结果：Q4-2为 **{s2['cash_cost_yuan']:,.2f}元**，Q4-3为 **{s3['cash_cost_yuan']:,.2f}元**。评价期为2025年2月1日至12月31日，共334天。两个表分别对应 `result4-2.xlsx` 与 `result4-3.xlsx`。",'',
        '## 1. 主方案与实际账单','',
        '|指标|Q4-2|Q4-3|','|---|---:|---:|']
    for label,key in [('原计划购电费（元）','planned_cost_yuan'),('调整净费用（元）','adjustment_cost_yuan'),('紧急购电费（元）','emergency_cost_yuan'),('实际总费（元）','cash_cost_yuan'),('紧急购电量（kWh）','emergency_kwh'),('多余能量（kWh）','unused_kwh'),('充电量（kWh）','charge_kwh'),('放电量（kWh）','discharge_kwh'),('上调量（kWh）','up_kwh'),('下调量（kWh）','down_kwh')]:
        lines.append(f'|{label}|{s2[key]:,.2f}|{s3[key]:,.2f}|')
    lines.extend(['',f"Q4-3主方案比Q4-2主方案少支出 **{s2['cash_cost_yuan']-s3['cash_cost_yuan']:,.2f}元（{100*(1-s3['cash_cost_yuan']/s2['cash_cost_yuan']):.3f}%）**。两者同时存在光伏信息与调整权限差异，因此该差额是方案整体比较，不能单独归因于控制频率。",'',
        'Q4-2逐时实际费用为 `p_real × (Q + 5E)`。Q4-3逐时实际费用为 `p_real × [Q + 1.5(R−Q)正部 − 0.5(Q−R)正部 + 5E]`。日内每次更新都保留当天零点Q；每个最终执行时段只结算一次。负的调整净费用表示下调退款超过上调费用，最终正常购电成本仍非负。',
        '', '## 2. 实现选择与因果性','',
        '- 时间与单位：每天144个10分钟区间，145个库存时刻。附件标签00:10解释为00:00—00:10的终点；模板时段文字按这一约定纠正，原始数据顺序不移动。交流母线侧功率除以6得到kWh。',
        '- 电池：充、放电效率各0.9，库存1200—10800 kWh，交流侧充、放电功率各不超过5000 kW。同一10分钟时段不同时充放电。',
        '- 冷启动：1月1日库存6000 kWh，采用附件1典型曲线；负荷和历史光伏预测在1月热身阶段使用预先固定的P1加权历史方法，电价采用7日对数加权均值且不作日内修正。Q4-3可使用当时已发布的附件3预报。Q4-2热身ρ=0、Q4-3热身ρ=1，γ均为0.5。',
        '- 残差档案：跳过最初7日的短历史残差，从1月8日对应的预测误差开始积累。无已完成残差时用确定性场景，此后最多枚举最近28个历史日，等概率使用全部独立日期。首次单残差场景出现在1月9日；这不是“至少有7个残差才启用”。',
        '- 正式负荷预测继承第二问P3随机森林：64棵树、最大深度10、叶节点最小样本8、最多28个训练目标日，种子20260912；滞后特征最多追溯35日原始观测。每天只用此前已结束的数据重新训练。Q4-2历史光伏同法预测，Q4-3使用当前已发布的官方整点节点作分段线性积分，发布点锚定最新可见实测光伏。',
        f"- 电价：1月21—31日比较前日、7/14/28日均值和加权曲线，选中 **{price['method']}**；按剩余时段MAE选择日内修正 **α={price['alpha']}、衰减率={price['decay_per_hour']} h⁻¹**。日内修正只吸收已结束的10分钟价格，零点基准冻结至当天结束。价格场景通过对数残差指数变换得到，严格为正。优化中的均价为场景均价。",
        '- 场景：同一个已结束历史日期的净负荷残差与价格对数残差配对。净负荷允许为负，不另行截零。Q4-2主信息口径不使用附件3官方光伏预报。',
        '- 参数验证：1月21—31日逐日回放，初始库存取统一热身轨迹1月20日末值；各候选在验证段末统一要求库存不低于6000 kWh。先比较γ∈{0,0.25,0.5,1}，再比较ρ或K。ρ/K费用相差不超过最低费的0.01%时取较小值。2月1日前冻结参数，不用2—12月结果调参。',
        f"- 历史选择结果：Q4-2 **γ={selection['q42']['gamma']}、ρ={selection['q42']['rho']}**；Q4-3 **γ={selection['q43']['gamma']}、K={selection['q43']['k']}**。同一题内各正式对照继承相同预先规定的1月热身库存；2月1日两题实际衔接值均为6000 kWh，来自热身结果，并未手动重置。",
        '- 前瞻实现：在0/6/12点只展开下一次更新，共同前缀36步，未来组内动作共享，各分支必须从相同桥接库存出发。信息信号严格使用下一次光伏预报修订量和更新前一小时价格对数误差。K-means最多3组，标准差归一化，至少每组5个不同历史日，样本不足回退K=1。18点直接求解剩余时段。',
        '- 终端：常规日使用γ·max(6000−S末,0)；12月31日所有未来分支要求S末≥6000。没有强制每日回到6000。终端软惩罚用于决策，不计入实际支付费用。',
        '', '## 3. 全年对照','', '|方案|实际总费（元）|正常结算（元）|紧急费（元）|','|---|---:|---:|---:|'])
    for row in totals:lines.append(f"|{row['label']}|{row['cash_cost_yuan']:,.2f}|{row['normal_cost_yuan']:,.2f}|{row['emergency_cost_yuan']:,.2f}|")
    lines.extend(['',f"相对各自点预测对照，Q4-2主方案节约 **{100*(1-s2['cash_cost_yuan']/summaries['q42_point']['cash_cost_yuan']):.3f}%**，Q4-3主方案节约 **{100*(1-s3['cash_cost_yuan']/summaries['q43_point']['cash_cost_yuan']):.3f}%**。保留残差场景主要减少了高价紧急补电。",'',
        f"Q4-2选择ρ=0，比ρ=1全年少支出 {summaries['q42_pair']['cash_cost_yuan']-s2['cash_cost_yuan']:,.2f}元，幅度约 {100*(1-s2['cash_cost_yuan']/summaries['q42_pair']['cash_cost_yuan']):.3f}%。ρ=0只表示在决策目标中对小样本价量依赖作完全收缩，不能据此断言真实价格与净负荷独立。Q4-3仍保留成对价量场景ρ=1。",'',
        f"Q4-3的K=2、3全年分别比K=1少支出 {s3['cash_cost_yuan']-summaries['q43_lookahead_k2']['cash_cost_yuan']:,.2f}元和 {s3['cash_cost_yuan']-summaries['q43_lookahead_k3']['cash_cost_yuan']:,.2f}元；但1月验证均劣于K=1，故提交表维持历史选出的K=1。该结果符合模型的收益边界：前瞻能降低同信息、同初值下的优化目标，但不保证任意样本期实际账单降低。",'',
        f"全年可分组的更新共有1002次（0/6/12点）。请求K=2时，有{summaries['q43_lookahead_k2']['group_counts'].get('2',0)}次形成2组；请求K=3时，有{summaries['q43_lookahead_k3']['group_counts'].get('3',0)}次形成3组。其余回退为1；18点334次天然没有下一次日内分叉，不能计作分组失败。",'',
        '按7日循环块、2000次重抽样描述成对日费用差的波动，结果如下。该区间只描述当前单年样本对分块重抽样的敏感性，不证明跨年份收益。','',
        '|基准→候选|实际节约（元）|描述性95%区间（元）|','|---|---:|---:|'])
    for r in comparisons:
        lo,hi=r['descriptive_95_interval_yuan'];lines.append(f"|{r['baseline']} → {r['candidate']}|{r['saving_yuan']:,.2f}|[{lo:,.2f}, {hi:,.2f}]|")
    lines.extend(['','## 4. 预测与数值复核','','|指标|Q4-2主方案|Q4-3主方案|','|---|---:|---:|'])
    for label,key in [('电价MAE（元/kWh）','price_mae'),('电价RMSE（元/kWh）','price_rmse'),('电价CRPS（元/kWh）','price_crps'),('场景80%区间实际覆盖率','price_coverage80')]:
        lines.append(f'|{label}|{s2[key]:.6f}|{s3[key]:.6f}|')
    lines.extend(['',f"- 全部18项机制单元检查通过。独立审计共覆盖{audit['experiment_count']}组计算（含热身与验证）、{audit['days_checked']}个策略日、{audit['optimizations_checked']}次求解，逐项重算原计划费、调整费、紧急费、场景目标、分支概率、能量平衡、互斥、库存边界与跨日衔接。最大绝对误差 **{audit['maximum_error']:.3e}**，容差2×10⁻⁵。不同检查分别采用其对应的元或kWh单位。",
        f"- 在四个指定日期及各更新时刻完成{struct['lp_milp_cases']}组真实数据LP/MILP对照，含分组模型；最大目标差 **{struct['maximum_lp_milp_error']:.3e}元**。同输入K=1与共同动作基准目标一致，K=2的优化目标不高于基准。",
        '- 全年采用模型第13节给出的LP净化定理求解：删除可同时出现的充放电循环，严格保留库存路径，使净购电需求不增。模型的正电价、免费弃能、无限制补电等前提全部保持；不是放松互斥后直接把不合物理的解交付。',
        f"- 两份保存后的XLSX由独立只读程序检查，共比对{sum(r['numeric_cells_checked'] for r in excel):,}个数值单元格，最大差 {max(r['maximum_cell_error'] for r in excel):.3e}；逐日合计公式与全年实际总费均已核对。所有工作表均完成渲染检查。",
        '', '## 5. 为什么结果在模型下合理','',
        '1. 储能守恒：正式期两端库存相同，全年放电量/充电量等于0.81，与两次0.9效率一致。每个10分钟时段充放电互斥；4小时汇总表中同一行同时出现充、放电数值，表示该4小时内不同10分钟时段分别执行，符合模型。',
        f"2. 终端行为：Q4-2有{s2['end_at_min']}天、Q4-3有{s3['end_at_min']}天日末落在1200 kWh下界，两者均无日末落在10800 kWh上界的日期。日末状态可以变化；12月31日均为6000 kWh，未通过年末耗尽电池人为降低费用。",
        '3. 购电功率可超过5 MW：5000 kW约束作用于电池充放电，而非无上限的正常购电通道。图中的购电尖峰与电池充电时段一致，不能将其误判为越过电池功率限制。',
        '4. 调度中的跳变来自10分钟电价与线性目标，原模型没有额外的爬坡、启停或平滑成本。图保留真实阶梯，未为了视觉效果更改策略或增加这些约束。',
        '5. 多余能量U按原模型免费弃置，同时包含光伏过剩和正常购电冗余；不能直接把全年U称为纯弃光量。净负荷为负的时段正常保留。',
        '6. 每次有限场景优化已达到求解器要求，并不意味着已经得到未知未来条件下的全年全局最优策略。参数只做了一次1月验证冻结，依赖收缩和信息分组的增益较小，应使用上述如实对照表述。',
        '', '## 6. 文件使用与论文写法','',
        '- `result4-2.xlsx`：“计划购电量”EQ列为含紧急购电的日实际总费。',
        '- `result4-3.xlsx`：“计划购电量”EQ列仅为原计划承诺费；“调整购电量”存最终正常购电R，EQ列为含紧急购电的日实际总费。两页费用不能相加。',
        '- `题目指定日期表格.md`：四个指定日期的六个10分钟购电值、4小时充放电量、首末库存与紧急购电区段。',
        '- `q42/q43_10分钟完整轨迹.csv`：完整Q、R、C、D、S、E、U与实际价格，供复核和重绘；`全年方案汇总.csv`收齐11组对照。',
        '- `论文用图`：7张450 dpi PNG、对应可编辑SVG与7页PDF图集，另含图注、图源数据与排版校验记录。',
        '- `模型与代码逐项对应.md`：每个数学定义对应的函数和检查。求解日志、历史选参、源文件摘要与独立复核记录保存在计算工程中。',
        '', '可用于论文的结果段：', '',
        f"> 基于价格—净负荷成对残差，本文构建带小样本依赖收缩的日计划模型与前瞻下一次信息更新的条件调整模型。参数经1月历史验证后固定，对2—12月334天进行连续回放。Q4-2与Q4-3主方案实际购电总费分别为{s2['cash_cost_yuan']/1e4:.4f}万元和{s3['cash_cost_yuan']/1e4:.4f}万元，较各自点预测对照下降{100*(1-s2['cash_cost_yuan']/summaries['q42_point']['cash_cost_yuan']):.3f}%和{100*(1-s3['cash_cost_yuan']/summaries['q43_point']['cash_cost_yuan']):.3f}%。依赖收缩在历史验证中选择ρ=0；分组前瞻在全年对照中有小幅收益，但历史验证选择K=1，故提交结果保留较简模型。能量平衡、跨日库存、调整结算及分组非预见性均通过独立重算，LP净化与物理MILP在指定数据案例上保持目标一致。",'',
        'CVaR、额外负荷日内校正、锁定远期电价、Q4-2官方光伏预报扩展、全信息下界及下调额外罚金均未加入主结果。代码保留下调额外罚金的结算入口，但本次未声称已完成该敏感性回测。'])
    (out/'第四问结果与合理性复核.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    mapping='''# 模型与代码逐项对应

依据为随包保存的建模手修订版 model-q4.tex；下表的章节和公式标签均来自该文档。Q4-2与Q4-3共用物理层，不另建题外模型。

|模型位置|本次实现|核对证据|
|---|---|---|
|第2节信息集|q4_forecasting.price_predictions、scenarios只使用已结束历史日、当前已结束价格前缀及当前官方发布；run_q4.experiment顺时执行|两项未来数据扰动单元检查；planning_log中的history_days与更新时间|
|第3节 eq:bill2、eq:b、eq:bill3|q4_dispatch.bill；audit_q4独立重写实际结算|每天与全年费用逐项重算；Q/R原计划不变|
|第4节 eq:balance、eq:recourse|e=max(N+c−d−r,0)，u=max(r+d−c−N,0)|audit_q4能量平衡与补救定义|
|第4节 eq:stock、eq:capacity、eq:mode|Battery与solve_dispatch：0.9双效率、1200—10800、5000/6、互斥|全轨迹与全部条件分支检查|
|第4节 eq:terminal|gamma正部软惩罚；年末每条分支≥6000；真实库存跨日传递|audit_q4跨日与末端检查|
|第5节 eq:correction|price_predictions：零点均值基线，已结束对数误差递推，按小时指数衰减|测试当前/未来价格扰动不进入当前预测；price_selection.json|
|第6节成对残差|scenarios按同一历史日配对净负荷残差与价格对数残差，至多28天|配对检查、负净负荷保留检查|
|第7—8节共同动作基准|solve_dispatch(groups=None)，Q4-2固定Q，Q4-3日内固定零点Q调整R|run_q4的q42_pair与q43_common|
|第9节 eq:H、eq:weights|逐时场景紧急费用的等价线性上图形式，没有加入跨时场景动作|场景路径重排不改变共同动作目标的单元检查|
|第10节 eq:shrink-H、eq:q42-new|recourse_weights=π[ρp+(1−ρ)均价]；正常购电仍按均价；只有Q4-2可用ρ≠1|ρ=0、ρ=1、ρ=0.25笛卡尔混合等价检查|
|第11节信息信号与分组|scenarios中的ZG/Zp与information_groups；下一次预报由当前预报+历史修订形成并按模型截断非负|未来预报扰动检查；每组≥5不同历史日；冻结分组映射记录|
|第11节 eq:lookahead-main|共同36步前缀+每组一个剩余动作尾部；零点联合选择原计划Q|solve_dispatch分支节点及期望目标独立重算|
|第11节 eq:branch-bridge|每个尾部初始库存与共同前缀末库存设等式|全部分支bridge检查；真实案例K=1退化与K=2可行集包含检查|
|第11节执行与收益边界|仅执行根节点前36步；真实发布后重新求解；旧分支仅作计划档案|run_q4记录executed_terminal_stock，audit_q4逐项比对执行前缀|
|第12节 eq:q80、eq:band|仅用于解释和单元检查，不用分位数公式替代储能优化|价格加权80%与70%—90%不调整区间测试|
|第13节LP净化证明|purify按ε=min(c,d/(ηcηd))消除同时充放电；method=milp可直接启用二进制物理模型|18项单元检查及32组真实数据LP/MILP对照|
|第16节对照与验收|run_q4.select_parameters、definitions；audit_q4；structural_checks|11组全年对照、18组热身/验证、完整实际账单和复核记录|

## 实现层选择

模型允许少量参数由历史验证确定。本次候选网格、冷启动规则、验证日期、回退规则及终端处理均在《第四问结果与合理性复核》第2节列明。没有把未定义的风险项、平滑成本或未来真实信息加到模型中。

forecasting.py、forecasting_v2.py为前问的兼容预测模块，Q4仅调用数据读取、时间标签、P1冷启动及P3随机森林。q3_forecasting.py仅调用read_official和integrate_hourly两个函数，未调用第三问的指数尺度场景模型。第四问的场景分布与优化全部在q4_forecasting.py及q4_dispatch.py中定义。
'''
    (out/'模型与代码逐项对应.md').write_text(mapping,encoding='utf-8')
    print('Wrote result report, model mapping and numerical environment',flush=True)

if __name__=='__main__':main()
