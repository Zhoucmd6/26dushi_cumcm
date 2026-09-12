"""从保存结果按公式独立复算，不调用优化矩阵或场景生成函数。"""
from pathlib import Path
from datetime import date
import sys
import csv
import json
import numpy as np
from forecasting import read_dataset
from run_v2 import write_json


def audit(run):
    root = Path(__file__).resolve().parents[1]
    dataset = read_dataset(root/'data/processed')
    config = json.loads((run/'run_manifest.json').read_text(encoding='utf-8'))['config']
    selection = json.loads((run/'selection.json').read_text(encoding='utf-8'))
    records = []
    for folder in sorted(p.parent for p in run.rglob('summary.json')):
        spec = json.loads((folder/'specification.json').read_text(encoding='utf-8'))
        summary = json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        info = json.loads((folder/'information_audit.json').read_text(encoding='utf-8'))
        a = np.load(folder/'trajectories.npz')
        days = a['day_indices'].tolist()
        if days != spec['days']:
            raise ValueError('Saved day indices differ')
        n = len(days)
        q,c,d,s,e,u = [a[k] for k in ('Q','C','D','S','E','U')]
        if any(x.shape != (n,144) for x in (q,c,d,e,u)) or s.shape != (n,145):
            raise ValueError(f'Bad shape: {folder}')
        if not all(np.isfinite(x).all() for x in (q,c,d,s,e,u)):
            raise ValueError('Nonfinite results')
        b = spec['battery']
        net = dataset.load_kw[days]/6 + c - q - dataset.pv_kw[days]/6 - d
        checks = {
            'flow_nonnegative_kwh': max(0., -min(x.min() for x in (q,c,d,e,u))),
            'state_equation_kwh': float(np.max(abs(np.diff(s,axis=1)-b['eta_charge']*c+d/b['eta_discharge']))),
            'energy_balance_kwh': float(np.max(abs(q+e+dataset.pv_kw[days]/6+d-dataset.load_kw[days]/6-c-u))),
            'emergency_replay_kwh': float(np.max(abs(e-np.maximum(net,0)))),
            'unused_replay_kwh': float(np.max(abs(u-np.maximum(-net,0)))),
            'simultaneous_per_day_kwh': float(np.max(np.minimum(np.maximum(c,0),np.maximum(d,0)).sum(axis=1))),
            'state_bounds_kwh': max(0.,b['min_kwh']-s.min(),s.max()-b['max_kwh']),
            'power_bounds_kwh': max(0.,c.max()-b['max_power_kw']/6,d.max()-b['max_power_kw']/6),
            'cross_day_kwh': float(np.max(abs(s[1:,0]-s[:-1,-1]))) if n>1 else 0.,
            'initial_kwh': float(abs(s[0,0]-spec['initial_kwh'])),
            'cash_total_yuan': float(abs((q+5*e)@dataset.prices @ np.ones(n)-summary['total_cash_cost_yuan']))}
        if spec['final_target'] is not None and not spec['no_storage']:
            checks['final_kwh'] = float(abs(s[-1,-1]-spec['final_target']))
        with np.load(run/'forecasts'/f"{spec['model']}.npz") as saved_forecasts:
            forecasts = {key:saved_forecasts[key] for key in saved_forecasts.files}
        with (folder/'daily_results.csv').open(encoding='utf-8-sig',newline='') as stream:
            daily = list(csv.DictReader(stream))
        objective_max = 0.
        for i, (day, meta) in enumerate(zip(days,info)):
            origin = dataset.dates[day]
            if meta['origin'] != origin.isoformat() or (meta['training_end'] and date.fromisoformat(meta['training_end']) >= origin):
                raise ValueError('Training information leak')
            if meta.get('residual_end') and date.fromisoformat(meta['residual_end']) >= origin:
                raise ValueError('V1 residual information leak')
            if meta.get('residual_dates'):
                ids = [(date.fromisoformat(x)-dataset.dates[0]).days for x in meta['residual_dates']]
                if any(j >= day for j in ids):
                    raise ValueError('Bootstrap information leak')
                counts = np.asarray(meta['residual_counts'])
                if spec['mode'] == 'bootstrap' and counts.sum() != config['bootstrap_draws']:
                    raise ValueError('Bootstrap count mismatch')
                if spec['mode'] == 'bootstrap':
                    pool = np.arange(max(0,day-config['residual_days']),day)
                    sampled = np.random.default_rng(np.random.SeedSequence([spec['seed'],day])).choice(pool,config['bootstrap_draws'],replace=True)
                    expected_ids, expected_counts = np.unique(sampled, return_counts=True)
                    if not np.array_equal(ids, expected_ids) or not np.array_equal(counts, expected_counts):
                        raise ValueError('Bootstrap seed reconstruction failed')
            elif spec['mode'] == 'v1_enumerate':
                ids = list(range(max(7,day-config['residual_days']),day))
                counts = np.ones(len(ids))
            else:
                ids, counts = [], np.ones(1)
            pl, pg = forecasts['load_kw'][day], forecasts['pv_kw'][day]
            if ids:
                l = np.maximum(pl+spec['residual_scale']*(dataset.load_kw[ids]-forecasts['load_kw'][ids]),0)/6
                g = np.maximum(pg+spec['residual_scale']*(dataset.pv_kw[ids]-forecasts['pv_kw'][ids]),0)/6
                if spec['mode'] == 'v1_enumerate':
                    support = dataset.pv_kw[max(0,day-28):day].max(axis=0)>0
                    g[:,~support] = 0
            else:
                l,g = np.maximum(pl[None,:],0)/6,np.maximum(pg[None,:],0)/6
            expected = float(dataset.prices@q[i] + (counts/counts.sum())@(np.maximum(l+c[i]-q[i]-g-d[i],0)@(5*dataset.prices))
                             + spec['terminal_coefficient']*s[i,-1])
            objective_max = max(objective_max,abs(expected-float(daily[i]['planning_objective_yuan'])))
        checks['scenario_objective_reconstruction_yuan'] = objective_max
        if any(not np.isfinite(v) or v > 1e-5 for v in checks.values()):
            raise ValueError(f'{folder.name}: {checks}')
        records.append({'name':str(folder.relative_to(run)),'days':n,'checks':checks})
    chosen = min(selection['validation'], key=lambda r:r['total_cash_cost_yuan'])
    if chosen['model'] != selection['chosen_model'] or chosen['terminal_multiplier'] != selection['chosen_terminal_multiplier']:
        raise ValueError('Selection does not match January validation')
    report = {'audited_experiments':len(records), 'audited_days':sum(r['days'] for r in records),
              'maximum_numerical_residual':max(v for r in records for v in r['checks'].values()),
              'selection_from_january_only':True, 'records':records}
    write_json(run/'independent_audit.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='records'},ensure_ascii=False,indent=2))


if __name__ == '__main__':
    audit(Path(sys.argv[1]).resolve())
