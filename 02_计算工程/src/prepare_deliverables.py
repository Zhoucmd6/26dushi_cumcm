"""Tables and template payloads derived only from audited executed trajectories."""
import argparse,csv,json
import numpy as np
from forecasting import interval_label,clock_text
from q4_forecasting import read_inputs
from run_q4 import ROOT,write_json,write_csv
from structural_checks import DATES

def segments(e):
    mask=e>1e-6;rows=[];t=0
    while t<144:
        if not mask[t]:t+=1;continue
        begin=t
        while t<144 and mask[t]:t+=1
        rows.append([clock_text(10*begin)+'-'+clock_text(10*t),float(e[begin:t].sum())])
    return rows,float(e[~mask].sum())

def prepare(run):
    selection=json.loads((run/'selection.json').read_text('utf-8'));audit=json.loads((run/'audit.json').read_text('utf-8'))
    audited={r['experiment']:r for r in audit['experiments']}
    ds,prices,_=read_inputs(ROOT/'data/processed');out=run/'deliverables';out.mkdir(exist_ok=True)
    selected=[]
    for mode in ('q42','q43'):
        name=selection[mode+'_main'];assert name in audited
        folder=run/name
        with np.load(folder/'trajectory.npz') as z:a={k:z[k] for k in z.files}
        assert a['day_indices'].tolist()==list(range(31,365))
        daily=list(csv.DictReader((folder/'daily.csv').open(encoding='utf-8-sig')))
        payload={'headers':['日期\\时间',*[interval_label(t) for t in range(144)],'全天购电量','全天购电费'],
                 'plan':[],'adjusted':[],'battery':[],'emergency':[],'mode':mode,'experiment':name}
        timeline=[];omitted=0.
        for i,day in enumerate(a['day_indices']):
            date=str(ds.dates[day]);q,r,c,d,e,u,s=[a[k][i] for k in ('Q','R','C','D','E','U','S')];p=prices[day]
            cash=float(daily[i]['cash_cost_yuan']);plan=float(p@q)
            payload['plan'].append([date,*q.tolist(),float(q.sum()),cash if mode=='q42' else plan])
            payload['adjusted'].append([date,*r.tolist(),float(r.sum()),cash])
            battery=[[clock_text(240*j)+'-'+clock_text(240*(j+1)),float(c[j*24:(j+1)*24].sum()),
                      float(d[j*24:(j+1)*24].sum()),'00:00' if j==0 else '24:00' if j==1 else None,
                      float(s[0]) if j==0 else float(s[-1]) if j==1 else None] for j in range(6)]
            payload['battery'].extend([[date if j==0 else None,*row] for j,row in enumerate(battery)])
            seg,small=segments(e);omitted+=small
            payload['emergency'].extend([[date if j==0 else None,*row] for j,row in enumerate(seg or [['无',0.]])])
            if date in DATES:
                selected.append({'mode':mode,**{k:v if k=='date' else float(v) for k,v in daily[i].items()},
                    'purchases':[[interval_label(6*h),float(q[6*h]),float(r[6*h])] for h in (10,12,14,16,18,20)],
                    'battery':battery,'emergency_segments':seg})
            for t in range(144):
                normal=float(p[t]*(q[t]+1.5*max(r[t]-q[t],0)-.5*max(q[t]-r[t],0))) if mode=='q43' else float(p[t]*q[t])
                timeline.append({'date':date,'interval':interval_label(t),'load_kw':float(ds.load_kw[day,t]),
                    'pv_kw':float(ds.pv_kw[day,t]),'price_yuan_per_kwh':float(p[t]),
                    **{k+'_kwh':float(a[k][i,t]) for k in ('Q','R','C','D','E','U')},
                    'S_start_kwh':float(s[t]),'S_end_kwh':float(s[t+1]),'normal_cost_yuan':normal,
                    'emergency_cost_yuan':float(5*p[t]*e[t]),'cash_cost_yuan':normal+float(5*p[t]*e[t])})
        assert omitted<1e-4
        payload['emergency_omitted_kwh']=omitted
        write_json(run/(mode+'_workbook_payload.json'),payload)
        write_csv(out/(mode+'_每日账单.csv'),daily)
        write_csv(out/(mode+'_10分钟完整轨迹.csv'),timeline)
    write_json(out/'指定日期结果.json',selected)
    text=['# 第四问题目指定日期表格','','Q为零点原计划，R为最终正常购电量；R不是调整增量。电量单位kWh，费用单位元。','']
    for row in selected:
        text.extend([f"## {row['mode'].upper()} · {row['date']}",'','|时间段|零点计划 Q|最终正常购电 R|','|---|---:|---:|'])
        text.extend(f'|{t}|{q:.2f}|{r:.2f}|' for t,q,r in row['purchases'])
        text.extend(['',f"全天实际总费 **{row['cash_cost_yuan']:,.2f}**；原计划费 {row['planned_cost_yuan']:,.2f}，调整净费用 {row['adjustment_cost_yuan']:,.2f}，紧急费 {row['emergency_cost_yuan']:,.2f}。",
            f"计划购电 {row['planned_kwh']:,.2f}；最终正常购电 {row['executed_normal_kwh']:,.2f}；紧急购电 {row['emergency_kwh']:,.2f}。",
            f"0:00库存 {row['initial_kwh']:,.2f}；24:00库存 {row['terminal_kwh']:,.2f}。",'',
            '|时间段|充电量|放电量|','|---|---:|---:|'])
        text.extend(f'|{b[0]}|{b[1]:.2f}|{b[2]:.2f}|' for b in row['battery'])
        text.extend(['','|紧急购电时间段|购电量|','|---|---:|'])
        text.extend(f'|{t}|{e:.2f}|' for t,e in (row['emergency_segments'] or [['无',0.]]));text.append('')
    (out/'题目指定日期表格.md').write_text('\n'.join(text)+'\n',encoding='utf-8')
    print('Prepared two workbook payloads and full trajectories',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--run',default='full_q4_20260913');a=p.parse_args();prepare(ROOT/'outputs'/a.run)
