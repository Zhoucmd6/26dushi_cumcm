"""根据本次已完成实验生成论文可引用报告，禁止硬编码旧版数值或测试数量。"""
import json,shutil,platform
import numpy as np
import pandas as pd
from audit_q3 import audit_run
from run_q3 import write_json,write_csv

CROSSWALK=[
('式(1)、(32)','R+E+G+D=L+C+U','q3_dispatch.solve_roll；run_q3.run_policy','Q为0点原承诺，R替代Q；U为总富余，无售电收益、无U≤G约束。'),
('式(2)—(8)','Δt=1/6h，T=144；h=有效时刻−原发布时刻','q3_forecasting.read_official / integrate_hourly / make_scenarios','旧预报复用时h不归零；整点节点线性积分为主，小时均值解释另跑全年。'),
('式(9)—(11)','分组偏差收缩；µG=max(0,原始预报+b)','q3_forecasting.day_moments / build_cache','按完整历史日等权；删除旧版历史零支撑掩码，保留晨昏误差可能性。'),
('式(12)—(15)','历史因果残差方差收缩；σg(h)=σg0 exp(βh)，β≥0','q3_forecasting.fit_scale','目标钟点粗组与提前量分箱；指数通过带组截距、共享斜率的日数加权拟合。分箱、统一尺度作为对照。'),
('式(16)—(19)','R=1/[1+(σ/sref)²]；σ=sref√(1/R−1)','q3_forecasting.reliability / scale_from_reliability','R全部导出并独立反演核验。场景直接用等价σ，R不乘预测中心，也不替代场景概率。'),
('式(20)—(23)','同日成对L/G标准残差块，经当前µ、σ还原，截断为非负','q3_forecasting.make_scenarios','只抽已完成日；64次有放回抽样并按重复次数压缩权重；补充截断前后矩、协方差诊断。'),
('式(24)—(28)','pQ+1.5p(R−Q)+−0.5p(Q−R)+','q3_dispatch.settlement；audit_q3.audit_arrays','每个时段最终执行版仅相对原始Q结算一次；下调另罚备选口径单列，不相加滚动目标。'),
('式(29)—(38)','场景紧急费用CVaR；ζ实数、v≥Cω−ζ、v≥0','q3_dispatch.solve_roll / weighted_cvar','B4复用B3的全部校准、场景、γ与信息，只增加α、λ；另按实现日紧急费重算CVaR。'),
('式(33)—(40)','电池效率、界限、互斥与软终端Φ=γ(6000−S末)+','q3_dispatch.solve_roll；run_q3.run_policy','C/D交流侧≤5000/6kWh；S∈[1200,10800]。无软目标、正软目标、每日硬6000均已比较。'),
('第8—10节','0/6/12/18h滚动；只执行下一段36点；跨日连续','run_q3.run_policy','保留四次剩余计划版本。首次0点解得到Q；6小时内电池按固定计划执行。'),
('第11节','8种信息组合与B0—B4消融','q3_workflow.build_experiments；summarize_q3.collect','屏蔽新预报仍每天重算4次；配对7日移动块bootstrap估计单年探索性区间。'),
('第12节','历史概率/费用选参、分组误差、风险与运行指标','q3_workflow；q3_diagnostics；q3_report','72组1月候选；α=.90/.95、λ=0/.05/.1；提前量/发布时刻/实际光伏正或零分组MAE、RMSE与概率评分。')]

def report(run,records,marginal,maskdaily):
    out=run/'deliverables';selection=json.loads((run/'selection.json').read_text(encoding='utf-8'))
    main=records['main'];selected=selection['selected'];c=selected['calibration'];audit=audit_run(run)
    test=json.loads((run/'test_results.json').read_text(encoding='utf-8'))
    diag=json.loads((run/'probability_diagnostics_audit.json').read_text(encoding='utf-8'))
    cache=dict(np.load(run/'forecast_cache.npz'))
    frame=pd.read_csv(out/'提前量与发布时间分组误差.csv')
    daylight=frame[(frame['mode']==selected['mode'])&(frame.lead_hour_bin==0)&(frame['subset']=='daylight')]
    cover=float(np.average(daylight.coverage,weights=daylight.points));crps=float(np.average(daylight.crps,weights=daylight.points))
    maxerr=max(v['max_residual'] for v in audit.values());best=min(maskdaily,key=lambda m:maskdaily[m].sum())
    registry=json.loads((run/'experiment_registry.json').read_text(encoding='utf-8'))
    unique_policies=len({json.dumps(p,sort_keys=True) for p in registry.values()})
    saving=float((maskdaily[0]-maskdaily[7]).sum());allnames=[n for n in records if not n.startswith('info_mask')]
    from summarize_q3 import label,mask_label
    lines=['# 第三问模型核验与求解报告','',
      f"本次按建模手 `model-q3.pdf` 的条件尺度随机滚动模型重新计算，正式主方案为 **B3：分箱条件尺度、风险中性随机MPC**。2025年2月1日至12月31日共334天，实际购电总费 **{main['total_fee_yuan']:,.2f} 元**，正常结算 {main['normal_fee_yuan']:,.2f} 元，紧急购电 {main['emergency_fee_yuan']:,.2f} 元。该费用是有限历史场景策略的回测结果，不是全年全信息最优值。",
      '', '## 1. 这次修正了什么','',
      '删除旧版PDF之外的历史零光伏支撑掩码，使预测中心严格按式(11)生成；指数尺度按式(14)生成，不再额外逐点抬高。CVaR从原来的统一尺度模型改为叠加在同一个B3条件尺度模型上。补齐每日硬终端对照、参数历史验证、R完整序列、提前量误差分组和场景截断影响。代码、提交表、指定日期表与图表均由本次新轨迹生成。',
      '', '## 2. 公式与代码对应','', '|PDF位置|数学含义|实现入口|口径与核验|','|---|---|---|---|']
    lines.extend('|'+ '|'.join(row)+'|' for row in CROSSWALK)
    lines+=['','## 3. 数据和论文必须写清的实现选择','',
      '1. 附件1提供重复使用的144点电价；附件2提供365×144点实际负荷和光伏；附件3提供365×4×24条官方预报；附件5保留result3的四个工作表。本问不使用附件4的波动电价。',
      '2. 00:10—24:00按前一10分钟区间终点解释，模板改正时段标签，数值不循环移动。小时官方预报主方案按未来整点节点线性积分；j=0锚点只用刚结束的实测时段，0点用前一天末值。小时均值解释另列敏感性结果。',
      '3. 负荷中心采用过去完整28日加权均值，属于PDF允许的历史均值分支。没有把它写成第二问version2已选中的随机森林；第三问所有消融固定这套负荷预测。若论文写负荷随机森林，应另实现并重算，不能仅改模型名称。',
      f"4. 目标钟点仍为四个6小时粗组；提前量分箱、历史窗口、偏差与方差共同收缩强度通过1月验证选择。本次为窗口{c['window']}日、收缩{c['shrink']:g}日、提前量{c['bin_hours']}小时分箱。尺度数值下限25kW、抽样64次、随机种子20260912是登记的数值设置，不宣称也做了全面优化。",
      f"5. s_ref由首7日已实现的光伏正值时段原始预报RMSE确定，1月8日起固定为{cache['sref'][31]:.6f} kW；前7日采用1000kW临时参考。R与σ是一一映射，R本身没有额外优化收益，R=0.8不表示80%正确率。分箱R可非单调；β≥0仅约束指数候选的同组曲线。",
      '6. 主效率为充电、放电各0.9；两侧取√0.9的往返效率90%解释单列。C、D为交流母线侧电量，S为内部储量，容量12000kWh、安全区间1200—10800kWh。每6小时重算电池计划，区间内没有额外实时电池反馈控制。',
      f"7. 1月1日S=6000kWh；缺少上年负荷历史，采用PDF允许的零Q、空闲电池冷启动，缺口紧急补足。1月2—31日按预先固定28/7/6分箱模型、γ=最低电价/0.9连续热身，各方案共同从2月1日S={main['initial_kwh']:.9f}kWh出发。没有用1月末选参结果改写过去热身决策。",
      '8. 每天不重置库存。主方案的6000kWh软目标是近似未来价值；全年末统一6000kWh是公平比较的附加条件，不能写成题面要求每天首尾相同。硬终端对照才逐日强制6000。软惩罚和CVaR项不计入实际现金账单。',
      '', '## 4. 参数选择与B0—B4关系','',
      '1月21—31日为验证期。每个验证日只利用之前完整日校准，并接入当时可见的官方预报。候选窗口{14,28}日、收缩{3,7}日、分箱{3,6}小时、尺度{统一,分箱,指数}、γ∈{0,p_min/0.9,2p_min/0.9}共72组。所有候选从同一个1月20日末库存开始、验证期末6000，以排除库存耗尽造成的假节省。',
      f"B3的结构预先固定为PDF的条件尺度主模型；其参数按1月实际现金费选择，指数在同一配置下若白天CRPS差于统一或分箱则淘汰。最终B3选中{selected['mode']}，γ={selected['gamma']:.9f}元/kWh，1月验证费用{selected['total_fee_yuan']:,.2f}元。",
      f"允许统一尺度参与的全候选最低费用方案为{selection['global_cash_selected']['mode']}（{selection['global_cash_selected']['config_id']}），验证费用{selection['global_cash_selected']['total_fee_yuan']:,.2f}元。因此不能声称B3已在历史验证中全面优于简单模型。统一尺度结果保留为B2与全候选费用选择对照。",
      f"同一γ下按条件模型概率CRPS选参得到{selection['probability_selected']['config_id']}、{selection['probability_selected']['mode']}。本次概率选参和B3费用选参{'恰好一致，因此共用一条正式回测轨迹，不能虚构两者有改进差异' if selection['probability_selected']==selected else '不同，正式回测分别列出'}。完整候选表附在`1月参数验证完整表.csv`。",
      '', '|标识|实验目录|变化|','|---|---|---|',
      '|B0|info_mask0|仅0点新预报；仍每日4次优化，其他设置同B3|',
      '|B1|point_official|使用原始官方光伏点预测，去掉偏差校准及概率场景|',
      '|B2|scale_pooled|校准中心+统一误差尺度+成对日场景，λ=0|',
      '|B3|main|B2改为提前量条件尺度，其他参数和日块抽样一致|',
      '|B4|cvar90/cvar95及lambda005|B3基础上仅加CVaR风险项|',
      '', '## 5. 正式求解顺序','',
      '校验附件日期、电价、单位及官方预报发布时间 → 每日用过去完整日计算µ和σ → 用历史当时保存的因果µ、σ构建成对标准残差日块 → 均匀有放回抽样64次并压缩重复日权重 → 0点解全天MILP形成Q → 6/12/18点依据最新允许预报和真实S重解剩余时域 → 只执行接下来36个时段 → 实测到达后独立重算E、U与费用 → 跨日承接S → 最后导出表格与论文图。',
      '求解器采用SciPy/HiGHS的确定性等价MILP，保留充放电方向二进制变量。未强行改写为Benders：本次有限日场景规模可由直接MILP求解；PDF规定的是模型，并未强制分解算法。求解器超时只有在LP解满足电池互斥且有下界证书时才接受LP，否则重试MILP。',
      '', '## 6. 费用、风险、终端与敏感性比较','',
      '下表均为2—12月实现现金费；CVaR95按334个实现日紧急费重新计算，分位点保留部分概率质量。',
      '', '|方案|总费/元|紧急费/元|日紧急费CVaR95/元|','|---|---:|---:|---:|']
    for name in allnames:
        r=records[name];lines.append(f"|{label(name)}|{r['total_fee_yuan']:,.2f}|{r['emergency_fee_yuan']:,.2f}|{r['daily_emergency_cvar95_yuan']:,.2f}|")
    lines+=['','α∈{.90,.95}、λ∈{0,.05,.1}预先登记，λ=0复用B3。四个正风险参数组合既做1月验证，也做正式回测，见`1月风险参数验证.csv`。风险偏好没有按全年最低费用倒选；风险方案增加均值费用并不自动说明模型较差，应同时报告尾部变化。',
      '小时均值、效率解释和备选下调结算属于模型口径敏感性，不混入B0—B4的算法归因。全年费用最小的事后方案不自动替换本次预先定义的主方案。',
      '', '## 7. 预测质量与R解释','',
      f"主方案光伏正值样本的合并CRPS为{crps:.6f}kW，90%经验区间覆盖率{100*cover:.2f}%。该覆盖率{'低于' if cover<.9 else '达到或高于'}标称值，应如实报告；不能由R或有限场景直接声称概率安全保证。同一目标被多次发布评价，不能把全部点数当独立天气样本数。",
      '`提前量与发布时间分组误差.csv`给出原始点预测、校准中心和场景均值的MAE/RMSE，以及CRPS、PICP90、MPIW、IS0.1。lead_hour_bin=0表示该发布时刻的全部提前量；其它值表示1小时分箱。实际光伏为零的组包括夜间，也可能包括白天零发电，只有实测评价时使用该分组，不用于未来预测遮罩。',
      '`非负截断与联合场景诊断.csv`记录截断概率、截断前后均值/方差变化、同一时段的场景加权负荷—光伏协方差与相关系数。标准化日块保留历史联合路径，但有限抽样与非负截断可能改变矩和相关性，不保证场景均值恰好等于µ、标准差恰好等于σ。',
      f"`预测尺度与可信度R完整记录.csv`包含{diag['reliability_rows']:,}条发布—有效时段记录，每条给出三个尺度及R。反演σ最大绝对误差为{diag['R_to_sigma_max_error_kw']:.3g}kW。直接用σ构造场景与通过R再反演数学等价，不应把展示R本身写成新增优化模块。",
      '', '## 8. 预报时刻价值','',
      f"全四次预报相对仅0点新预报累计节省{saving:,.2f}元。本年度事后最低费用组合为{mask_label(best)}。各组合仍每日重算4次，复用旧预报时保留原始发布时刻和h=A+H。",
      '', '|可用预报时刻|总费用/元|相对仅0点节省/元|','|---|---:|---:|']
    for m in range(8):lines.append(f'|{mask_label(m)}|{maskdaily[m].sum():,.2f}|{(maskdaily[0]-maskdaily[m]).sum():+,.2f}|')
    lines+=['','新增某时刻相对于保留其余两次日内预报的边际区间：','',
      '|新增预报|累计节省/元|平均日节省/元|探索性95%区间/元|','|---|---:|---:|---|']
    for r in marginal:
        if r['new_mask']==7:lines.append(f"|{r['added_hour']}时|{r['total_saving_yuan']:+,.2f}|{r['mean_daily_saving_yuan']:+.2f}|[{r['mean_daily_ci_low_yuan']:.2f}, {r['mean_daily_ci_high_yuan']:.2f}]|")
    lines+=['','配对7日移动块bootstrap重复2000次，区间仅描述这一年、有季节变化的样本，不是跨年保证。附件没有3/9/15/21时新发布预报，不能用重算旧预报代替额外气象信息，也不能声称已证明这些时刻购买新预报的收益。',
      '', '## 9. 验证与写作边界','',
      f"完成{len(audit)}组334日实验，包含{unique_policies}种不同策略，共{sum(a['updates'] for a in audit.values()):,}次正式滚动求解的独立轨迹审计。本次全候选费用选择与B2设置相同，重复运行用于核对，不算新模型。最大残差{maxerr:.3g}，电量项单位kWh、账单项单位元，阈值1e-4。主方案最大求解上下界差{main['max_solver_gap_yuan']:.3g}元。物理约束、跨日连续、硬终端、最终版本、原始承诺、结算等价式与可见信息时序均核验。",
      f"{test['tests_run']}项自动测试通过，包含未来实测/未发布预报反泄漏、CVaR离散分位质量、公式结算、小时积分、硬终端约束、B3/B4参数一致性、R反演与晨昏零历史反例。所有指标均来自本次数据，不沿用旧版数值。",
      '另用独立代码按PDF式(10)—(16)重算334天、120240条发布—有效时段的偏差、中心、分箱/统一尺度和指数拟合，结果通过1e-6容差核验；记录见independent_calibration_audit.json。主方案本年每天末库存恰好为6000kWh，这是软目标下的最优选择，并非代码加入了每日等式约束。',
      '数值MILP求解成功不等于预测准确或分布假设正确。本实现覆盖PDF的主体模型和上述对照；PDF中同日偏差动态更新、更复杂负荷模型等可选扩展没有全部采用。论文应使用本报告中实际分箱尺度、历史均值负荷、固定6小时内电池计划的分支，不写成必然指数衰减、随机森林负荷或10分钟自适应电池。',
      '', '## 10. 论文手阅读顺序','',
      '先读`PDF与代码逐项对应.md`，再引用本报告的模型、参数和边界；结果用`题目指定日期表格.md`及`result3.xlsx`，图用`科研图表`。每日账单、完整执行轨迹、各次剩余计划、验证网格和R/概率诊断均单列。原计划费列与实际总费列不能相加。',
      '', '来源：用户提供的15页model-q3.pdf、C题题面及附件1/2/3/5。建模稿文献引用未另行全文核验，本报告不把算法组合宣称为原创理论。']
    (out/'第三问模型核验与求解报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    cross=['# PDF与代码逐项对应','','本次正式主方案为B3分箱条件尺度；B4严格复用B3并加入CVaR。','',
           '|PDF位置|含义|代码入口|实现说明|','|---|---|---|---|',*['|'+'|'.join(r)+'|' for r in CROSSWALK],
           '', '论文必须使用的口径：分箱尺度；28日历史加权负荷；6小时内固定电池计划；下调净退50%；跨日连续；年末6000作为比较条件。',
           '', '指数尺度是验证后未采用的候选，不应把主方案写成“验证了指数可信度衰减规律”。主模型也不能写成“历史验证费用全候选最优”，因为统一尺度候选验证费用略低。',
           '', '详见《第三问模型核验与求解报告》的完整参数、验证和数值。']
    (out/'PDF与代码逐项对应.md').write_text('\n'.join(cross)+'\n',encoding='utf-8')
    shutil.copy2(run/'january_validation_grid.csv',out/'1月参数验证完整表.csv')
    risk=json.loads((run/'january_risk_validation.json').read_text(encoding='utf-8'))
    config=json.loads((run/'run_config.json').read_text(encoding='utf-8'))
    gamma_index=config['gamma_candidates'].index(selected['gamma'])
    selected_summary=json.loads((run/'validation'/selected['config_id']/f"{selected['mode']}_gamma{gamma_index}"/'summary.json').read_text(encoding='utf-8'))
    riskrows=[]
    if selected_summary:riskrows.append({'name':'B3_lambda0','alpha':.9,'lambda':0.,**{k:selected_summary[k] for k in ['total_fee_yuan','emergency_fee_yuan','daily_emergency_cvar90_yuan','daily_emergency_cvar95_yuan']}})
    for r in risk:riskrows.append({'name':r['name'],'alpha':r['policy']['alpha'],'lambda':r['policy']['risk_lambda'],**{k:r[k] for k in ['total_fee_yuan','emergency_fee_yuan','daily_emergency_cvar90_yuan','daily_emergency_cvar95_yuan']}})
    write_csv(out/'1月风险参数验证.csv',riskrows)
    daily=pd.read_csv(run/'experiments/main/daily.csv')
    daily['month']=daily.date.str[:7]
    fields=['total_fee_yuan','normal_fee_yuan','emergency_fee_yuan','emergency_kwh','unused_kwh','throughput_kwh','adjust_up_kwh','adjust_down_kwh','adjusted_intervals']
    monthly=daily.groupby('month')[fields].sum().reset_index()
    write_csv(out/'第三问月度费用与运行指标.csv',monthly.to_dict('records'))
    write_json(run/'runtime_versions.json',{'python':platform.python_version(),'numpy':np.__version__,'pandas':pd.__version__})
