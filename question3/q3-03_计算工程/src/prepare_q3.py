"""从通过审计的最终执行轨迹生成模板载荷、论文表格和完整版本档案。"""
from pathlib import Path
import argparse,json
import numpy as np
from forecasting import read_dataset, interval_label, clock_text
from reporting import four_hour_rows,emergency_segments
from run_q3 import ROOT,write_json,write_csv
from audit_q3 import audit_run

def prepare(run):
    run=Path(run);audit=audit_run(run)
    if 'main' not in audit:raise RuntimeError('Main experiment has not completed')
    ds=read_dataset(ROOT/'data/processed');a=dict(np.load(run/'experiments/main/dispatch.npz'))
    out=run/'deliverables';out.mkdir(exist_ok=True)
    Qrows=[];Rrows=[];battery=[];emergency=[];daily=[];selected=[];timeline=[];versions=[];omitted=0.
    for i,day in enumerate(a['day_indices']):
        date=str(ds.dates[day]);q,r,c,d,e,u,s=[a[k][i] for k in ['Q','R','C','D','E','U','S']]
        normal=float(a['normal_fee'][i].sum());ecost=float(e@(5*ds.prices));total=normal+ecost
        Qrows.append([date,*q.tolist(),float(q.sum()),float(q@ds.prices)])
        Rrows.append([date,*r.tolist(),float(r.sum()),total])
        four=four_hour_rows(c,d,s)
        for j,row in enumerate(four):battery.append([date if j==0 else None,*row])
        segments,small=emergency_segments(e);omitted+=small
        for j,row in enumerate(segments):emergency.append([date if j==0 else None,row['interval'],row['energy_kwh']])
        daily.append({'date':date,'planned_kwh':float(q.sum()),'adjusted_normal_kwh':float(r.sum()),
                      'emergency_kwh':float(e.sum()),'original_plan_fee_yuan':float(q@ds.prices),
                      'up_surcharge_yuan':float(1.5*ds.prices@np.maximum(r-q,0)),
                      'down_refund_yuan':float(.5*ds.prices@np.maximum(q-r,0)),
                      'final_normal_fee_yuan':normal,'emergency_fee_yuan':ecost,'total_fee_yuan':total,
                      'initial_kwh':float(s[0]),'terminal_kwh':float(s[-1]),'unused_kwh':float(u.sum())})
        if date in ['2025-03-20','2025-06-21','2025-09-23','2025-12-21']:
            selected.append({**daily[-1],'purchases':[[interval_label(h*6),float(q[h*6]),float(r[h*6])] for h in [10,12,14,16,18,20]],
                             'battery':four,'emergency_segments':segments})
        for t in range(144):
            timeline.append({'date':date,'interval':interval_label(t),'load_kw':float(ds.load_kw[day,t]),
                'pv_actual_kw':float(ds.pv_kw[day,t]),'price_yuan_per_kwh':float(ds.prices[t]),
                **{k:float(a[k][i,t]) for k in ['Q','R','C','D','E','U','normal_fee']},
                'S_start_kwh':float(s[t]),'S_end_kwh':float(s[t+1])})
        for j,h in enumerate([0,6,12,18]):
            for t in range(h*6,144):
                versions.append({'date':date,'decision_hour':h,'interval':interval_label(t),
                                 'is_executed_version':int(h*6<=t<h*6+36),'original_Q_kwh':float(q[t]),
                                 **{k+'_kwh':float(a[k+'_versions'][i,j,t]) for k in ['R','C','D']}})
    if omitted>1e-4:raise RuntimeError('Material emergency energy omitted')
    payload={'headers':['日期\\时间',*[interval_label(t) for t in range(144)],'全天购电量','全天购电费'],
             'plan':Qrows,'adjusted':Rrows,'battery':battery,'emergency':emergency,
             'emergency_omitted_kwh':omitted,'total_fee_yuan':sum(x['total_fee_yuan'] for x in daily)}
    write_json(run/'q3_workbook_payload.json',payload);write_json(out/'指定日期结果.json',selected)
    write_csv(out/'第三问每日费用与电量.csv',daily);write_csv(out/'第三问10分钟完整执行轨迹.csv',timeline)
    write_csv(out/'第三问各时刻完整计划版本.csv',versions)
    text=['# 第三问：题目指定日期结果','', '单位：电量 kWh，费用元。Q为0点承诺；R为该时段最终执行的正常购电量，不是调整增量。全天总费用包括正常结算与紧急购电。','']
    for row in selected:
        text.extend([f"## {row['date']}",'','|时间段|0点计划 Q|最终调整 R|','|---|---:|---:|'])
        text.extend(f'|{t}|{q:.2f}|{r:.2f}|' for t,q,r in row['purchases'])
        text.extend(['',f"全天 Q = {row['planned_kwh']:,.2f}；R = {row['adjusted_normal_kwh']:,.2f}；紧急购电 = {row['emergency_kwh']:,.2f}。",
                     f"原计划费 = {row['original_plan_fee_yuan']:,.2f}；最终正常结算 = {row['final_normal_fee_yuan']:,.2f}；紧急费 = {row['emergency_fee_yuan']:,.2f}；总费 = **{row['total_fee_yuan']:,.2f}**。",'',
                     '|时间段|充电量|放电量|','|---|---:|---:|'])
        text.extend(f'|{b[0]}|{b[1]:.2f}|{b[2]:.2f}|' for b in row['battery'])
        text.extend(['',f"0:00储量 {row['initial_kwh']:,.2f}；24:00储量 {row['terminal_kwh']:,.2f}。",'','|紧急购电时间段|购电量|','|---|---:|'])
        text.extend(f"|{e['interval']}|{e['energy_kwh']:.2f}|" for e in row['emergency_segments'])
        if not row['emergency_segments']:text.append('|无|0.00|')
        text.append('')
    (out/'题目指定日期表格.md').write_text('\n'.join(text),encoding='utf-8')
    print(json.dumps({'days':len(daily),'emergency_segments':len(emergency),'version_rows':len(versions),'total_fee':payload['total_fee_yuan']}))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('run');args=parser.parse_args();prepare(args.run)
