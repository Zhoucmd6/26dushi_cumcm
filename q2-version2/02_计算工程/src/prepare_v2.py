from pathlib import Path
import sys
import csv
import json
import numpy as np
from forecasting import read_dataset, interval_label
from reporting_v2 import build_payload
from run_v2 import write_json, write_csv


def main(run):
    root = Path(__file__).resolve().parents[1]
    dataset = read_dataset(root/'data/processed')
    config = json.loads((run/'run_manifest.json').read_text(encoding='utf-8'))['config']
    selection = json.loads((run/'selection.json').read_text(encoding='utf-8'))
    summaries = json.loads((run/'experiment_summaries.json').read_text(encoding='utf-8'))
    note = f"version2：零点固定Q/C/D；1月验证选择{selection['chosen_model']}；256次整日配对残差抽样；1月连续预热；年末储量6000。"
    payload = build_payload(run,dataset,config,selection['main_experiment'],note)
    write_json(run/'workbook_payload.json',payload)
    out = run/'deliverables'
    out.mkdir(exist_ok=True)
    write_csv(out/'第二问每日费用与电量.csv',payload['q2']['daily'])
    write_json(out/'指定日期结果.json',payload['q2']['selected_dates'])
    # NpzFile按键访问会重新解压整块数组；导出逐点表前只加载一次。
    with np.load(run/selection['main_experiment']/'trajectories.npz') as saved:
        a = {key:saved[key] for key in saved.files}
    trajectory = []
    for i,day in enumerate(a['day_indices']):
        for t in range(144):
            trajectory.append({'date':dataset.dates[day].isoformat(),'interval':interval_label(t),
                'price_yuan_per_kwh':float(dataset.prices[t]),'load_kw':float(dataset.load_kw[day,t]),
                'pv_kw':float(dataset.pv_kw[day,t]),**{k+'_kwh':float(a[k][i,t]) for k in ('Q','C','D','E','U')},
                'S_start_kwh':float(a['S'][i,t]),'S_end_kwh':float(a['S'][i,t+1])})
    write_csv(out/'第二问完整10分钟轨迹.csv',trajectory)
    fields = ['name','model','total_cash_cost_yuan','total_planned_cost_yuan','total_emergency_cost_yuan',
              'total_emergency_purchase_kwh','total_unused_kwh','initial_kwh','terminal_kwh','day_end_at_min_count']
    write_csv(out/'第二问全部方案对比.csv',[{k:s[k] for k in fields} for s in summaries])
    write_csv(out/'一月验证与选择.csv',[{'model':s['model'],'terminal_multiplier':s['terminal_multiplier'],
        'cash_cost_yuan':s['total_cash_cost_yuan'],**{k:v for k,v in s['forecast_metrics'].items() if k!='daylight_definition'}} for s in selection['validation']])
    write_csv(out/'全年预测精度.csv',[{'model':s['model'],**{k:v for k,v in s['forecast_metrics'].items() if k!='daylight_definition'}}
        for s in summaries if s['name'].startswith('candidate_')])
    lines = ['# 题目指定时段与指定日期结果','', '电量单位为 kWh，费用单位为元。所有时段采用区间起点 00:00 至末端 24:00。','',
        '## 第一问','', '| 时段 | 计划购电量 |','|---|---:|']
    lines += [f'| {t} | {v:.6f} |' for t,v in payload['q1']['requested_intervals']]
    lines += ['',f"全天购电量 {payload['q1']['purchase_kwh']:.6f}，费用 {payload['q1']['cost_yuan']:.6f}。",'']
    for item in payload['q2']['selected_dates']:
        lines += [f"## {item['date']}",'','| 时段 | 计划购电量 |','|---|---:|']
        lines += [f'| {t} | {v:.6f} |' for t,v in item['purchases_at_requested_intervals']]
        lines += ['','| 四小时时段 | 充电量 | 放电量 |','|---|---:|---:|']
        lines += [f'| {r[0]} | {r[1]:.6f} | {r[2]:.6f} |' for r in item['battery_four_hour']]
        lines += ['',f"初始储量 {item['initial_kwh']:.6f}，末端储量 {item['terminal_kwh']:.6f}，实际费用 {item['actual_total_cost_yuan']:.6f}。",'',
                  '| 紧急购电区间 | 电量 |','|---|---:|']
        lines += [f"| {x['interval']} | {x['energy_kwh']:.6f} |" for x in item['emergency_segments']]
        lines += ['']
    (out/'题目指定时段与日期表格.md').write_text('\n'.join(lines),encoding='utf-8')
    print(json.dumps({'main':selection['main_experiment'],'dates':len(payload['q2']['purchases']),
                      'emergency_segments':len(payload['q2']['emergency'])},ensure_ascii=False))


if __name__ == '__main__':
    main(Path(sys.argv[1]).resolve())
