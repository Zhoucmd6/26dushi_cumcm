"""前两问 version2：PDF候选比较、1月预热与选择、334天封闭回测。

运行：python src/run_v2.py --run v2_20260912 [--pilot]
同名完整实验可恢复；输入、配置或数值源码变化时拒绝混用旧结果。
"""
from dataclasses import asdict
from datetime import date
from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform
import shutil
import time
import traceback
import numpy as np
import scipy
import sklearn
import statsmodels
from forecasting import read_dataset, no_storage_purchase, make_scenarios
from forecasting_v2 import MODELS, build_cache, scenarios
from microgrid_core import Battery, solve_q1, simulate_fixed_plan, require_checks
from stochastic_dispatch import solve_q2_fixed

ROOT = Path(__file__).resolve().parents[1]


def validate_config(config):
    fixed = {'time_interpretation': 'source_label_is_interval_end',
        'january_policy': 'common_predeclared_P1_continuous_dispatch_from_Jan1',
        'january_first_day_forecast': 'attachment1_prior_as_per_PDF_3.9.4',
        'initial_january_kwh': 6000, 'year_end_kwh': 6000,
        'selection_dates': ['2025-01-21','2025-01-31'], 'selection_terminal_kwh': 6000,
        'selection_primary_metric': 'actual_cash_cost_yuan',
        'terminal_rule': 'Phi(s)=-multiplier*min(price)/eta_charge*s',
        'output_start': '2025-02-01', 'output_end': '2025-12-31',
        'candidates': list(MODELS), 'main_solver': 'extensive'}
    for k, v in fixed.items():
        if config.get(k) != v:
            raise ValueError(f'Unsupported modeling rule: {k}; update implementation explicitly')
    Battery(config['eta_charge'], config['eta_discharge'])
    for key in ('bootstrap_draws','residual_days','seed'):
        if not isinstance(config[key], int) or config[key] < 1:
            raise ValueError(f'Invalid {key}')
    if not config['terminal_multipliers'] or any(not np.isfinite(x) or x < 0 for x in config['terminal_multipliers']):
        raise ValueError('Invalid terminal candidates')


def write_json(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def forecast_metrics(dataset, cache, days):
    el = cache['load_kw'][days]-dataset.load_kw[days]
    eg = cache['pv_kw'][days]-dataset.pv_kw[days]
    active = dataset.pv_kw[days] > 0
    return {'load_mae_kw': float(abs(el).mean()), 'load_rmse_kw': float(np.sqrt(np.mean(el**2))),
            'pv_mae_kw': float(abs(eg).mean()), 'pv_rmse_kw': float(np.sqrt(np.mean(eg**2))),
            'pv_daylight_mae_kw': float(abs(eg[active]).mean()),
            'pv_daylight_rmse_kw': float(np.sqrt(np.mean(eg[active]**2))),
            'daylight_definition': 'actual PV > 0, used only in retrospective scoring'}


def experiment(dataset, cache, config, folder, days, initial, *, terminal_multiplier=1.,
               residual_scale=1., seed=None, mode='bootstrap', battery=None,
               final_target=None, no_storage=False):
    folder = Path(folder)
    battery = battery or Battery(config['eta_charge'], config['eta_discharge'])
    seed = config['seed'] if seed is None else seed
    coefficient = -terminal_multiplier*float(dataset.prices.min())/battery.eta_charge
    specification = {'model': cache['model'], 'days': list(days), 'initial_kwh': initial,
        'terminal_multiplier': terminal_multiplier, 'terminal_coefficient': coefficient,
        'residual_scale': residual_scale, 'seed': seed, 'mode': mode,
        'no_storage': no_storage, 'battery': asdict(battery), 'final_target': final_target}
    if (folder/'summary.json').exists():
        if json.loads((folder/'specification.json').read_text(encoding='utf-8')) != specification:
            raise ValueError(f'Resume specification mismatch: {folder}')
        return json.loads((folder/'summary.json').read_text(encoding='utf-8'))
    folder.mkdir(parents=True, exist_ok=True)
    write_json(folder/'specification.json', specification)
    state = initial
    rows, audits = [], []
    trajectories = {k: [] for k in ('Q', 'C', 'D', 'S', 'E', 'U')}
    maxima = {}
    start = time.perf_counter()
    with (folder/'progress.jsonl').open('w', encoding='utf-8') as log:
        for count, day in enumerate(days, 1):
            if mode == 'v1_enumerate':
                l, g, prob, info = make_scenarios(dataset, {**cache, 'minimum_history_days': 7}, day,
                    residual_days=config['residual_days'], residual_scale=residual_scale)
            else:
                l, g, prob, info = scenarios(dataset, cache, day, seed=seed,
                    draws=config['bootstrap_draws'], residual_days=config['residual_days'],
                    residual_scale=residual_scale, mode=mode)
            terminal = final_target if count == len(days) else None
            if no_storage:
                q = no_storage_purchase(l, g, prob)
                c, d = np.zeros(144), np.zeros(144)
                plan_s = np.full(145, state)
                gap, runtime = 0., 0.
                objective = float(dataset.prices@q + prob@(np.maximum(l-g-q, 0)@(5*dataset.prices)) + coefficient*state)
                checks = {}
            else:
                result = solve_q2_fixed(dataset.prices, l, g, prob, battery=battery,
                    initial_kwh=state, terminal_kwh=terminal, terminal_cost_per_kwh=coefficient,
                    method=config['main_solver'], time_limit_s=config['time_limit_s'],
                    absolute_gap=config['absolute_gap_yuan'], relative_gap=config['relative_gap'])
                q, c, d, plan_s = [np.asarray(result['plan'][k]) for k in ('Q', 'C', 'D', 'S')]
                gap, runtime = result['absolute_gap_yuan'], result['runtime_s']
                objective, checks = result['objective_yuan'], dict(result['checks'])
            actual = simulate_fixed_plan(dataset.prices, dataset.load_kw[day]/6, dataset.pv_kw[day]/6,
                q, c, d, battery=battery, initial_kwh=state, emergency_multiplier=5.)
            s, e, u = [np.asarray(actual[k]) for k in ('S', 'E', 'U_total_unused')]
            checks.update(actual['checks'])
            checks['plan_state_reconstruction_kwh'] = float(np.max(abs(s-plan_s)))
            if trajectories['S']:
                checks['cross_day_kwh'] = float(abs(s[0]-trajectories['S'][-1][-1]))
            if terminal is not None and not no_storage:
                checks['final_target_kwh'] = float(abs(s[-1]-terminal))
            require_checks(checks)
            for k, v in checks.items():
                maxima[k] = max(maxima.get(k, 0.), v)
            row = {'date': dataset.dates[day].isoformat(),
                   'planned_cost_yuan': actual['planned_cost_yuan'],
                   'emergency_cost_yuan': actual['emergency_cost_yuan'],
                   'cash_cost_yuan': actual['cash_cost_yuan'], 'planned_purchase_kwh': float(q.sum()),
                   'emergency_purchase_kwh': float(e.sum()), 'total_unused_kwh': float(u.sum()),
                   'charge_kwh': float(c.sum()), 'discharge_kwh': float(d.sum()),
                   'initial_kwh': float(s[0]), 'terminal_kwh': float(s[-1]),
                   'solver_gap_yuan': gap, 'solver_runtime_s': runtime,
                   'planning_objective_yuan': objective, 'scenario_count': len(prob)}
            rows.append(row)
            audits.append(info)
            for k, v in zip(trajectories, (q, c, d, s, e, u)):
                trajectories[k].append(v)
            state = float(s[-1])
            log.write(json.dumps(row, ensure_ascii=False)+'\n')
            log.flush()
            if count == 1 or count % 60 == 0 or count == len(days):
                print(f'{folder.name}: {count}/{len(days)}, elapsed {time.perf_counter()-start:.1f}s', flush=True)
    np.savez_compressed(folder/'trajectories.npz', day_indices=np.array(days),
                        **{k: np.asarray(v) for k, v in trajectories.items()})
    write_csv(folder/'daily_results.csv', rows)
    write_json(folder/'information_audit.json', audits)
    summary = {'name': folder.name, 'model': cache['model'], 'days': len(days),
               'initial_kwh': initial, 'terminal_kwh': state,
               'total_cash_cost_yuan': sum(r['cash_cost_yuan'] for r in rows),
               'total_planned_cost_yuan': sum(r['planned_cost_yuan'] for r in rows),
               'total_emergency_cost_yuan': sum(r['emergency_cost_yuan'] for r in rows),
               'total_emergency_purchase_kwh': sum(r['emergency_purchase_kwh'] for r in rows),
               'total_unused_kwh': sum(r['total_unused_kwh'] for r in rows),
               'day_end_at_min_count': sum(abs(r['terminal_kwh']-battery.min_kwh) < 1e-5 for r in rows),
               'max_solver_gap_yuan': max(r['solver_gap_yuan'] for r in rows),
               'checks_maxima': maxima, 'forecast_metrics': forecast_metrics(dataset, cache, list(days)),
               'runtime_s': time.perf_counter()-start}
    write_json(folder/'summary.json', summary)
    return summary


def load_cache(dataset, model, config, folder, stop):
    file = folder/f'{model}.npz'
    if file.exists():
        arr = np.load(file)
        return {'model': model, 'load_kw': arr['load_kw'], 'pv_kw': arr['pv_kw'],
            'metadata': {int(k): v for k, v in json.loads(file.with_suffix('.json').read_text(encoding='utf-8')).items()}}
    result = build_cache(dataset, model, config['seed'], stop=stop, progress=lambda s: print(s, flush=True))
    np.savez_compressed(file, load_kw=result['load_kw'], pv_kw=result['pv_kw'])
    write_json(file.with_suffix('.json'), result['metadata'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', default='v2_20260912')
    parser.add_argument('--pilot', action='store_true')
    args = parser.parse_args()
    config = json.loads((ROOT/'configs/q1_q2_v2.json').read_text(encoding='utf-8'))
    validate_config(config)
    dataset = read_dataset(ROOT/'data/processed')
    out = ROOT/'outputs'/args.run
    out.mkdir(parents=True, exist_ok=True)
    numeric_files = ['run_v2.py', 'forecasting_v2.py', 'forecasting.py', 'microgrid_core.py', 'stochastic_dispatch.py']
    manifest = {'config': config, 'pilot': args.pilot, 'inputs': dataset.sources,
        'code': {n: hashlib.sha256((ROOT/'src'/n).read_bytes()).hexdigest() for n in numeric_files}}
    if (out/'run_manifest.json').exists():
        if json.loads((out/'run_manifest.json').read_text(encoding='utf-8')) != manifest:
            raise ValueError('Numeric source/config/input changed: create a new run id')
    else:
        write_json(out/'run_manifest.json', manifest)
        (out/'code_snapshot').mkdir(exist_ok=True)
        for n in numeric_files:
            shutil.copy2(ROOT/'src'/n, out/'code_snapshot'/n)
    write_json(out/'environment.json', {'python': platform.python_version(), 'numpy': np.__version__,
        'scipy': scipy.__version__, 'sklearn': sklearn.__version__, 'statsmodels': statsmodels.__version__})
    battery = Battery(config['eta_charge'], config['eta_discharge'])
    q1 = {m: solve_q1(dataset.prices, dataset.baseline_load_kw/6, dataset.baseline_pv_kw/6,
                    battery=battery, initial_kwh=6000, method=m) for m in ('lp', 'milp')}
    q1['roundtrip90_sensitivity'] = solve_q1(dataset.prices, dataset.baseline_load_kw/6,
        dataset.baseline_pv_kw/6, battery=Battery(np.sqrt(.9), np.sqrt(.9)), initial_kwh=6000, method='milp')
    write_json(out/'q1.json', q1)
    (out/'forecasts').mkdir(exist_ok=True)
    caches = {m: load_cache(dataset, m, config, out/'forecasts', 34 if args.pilot else 365) for m in MODELS}
    warm = experiment(dataset, caches[MODELS[0]], config, out/'warmup', range(31), 6000.)
    warm_s = np.load(out/'warmup/trajectories.npz')['S']
    validation = []
    for model in MODELS:
        for mult in config['terminal_multipliers']:
            summary = experiment(dataset, caches[model], config,
                out/'validation'/f'{model}_terminal_{mult}', range(20, 31), float(warm_s[19, -1]),
                terminal_multiplier=mult, final_target=6000.)
            validation.append({'model': model, 'terminal_multiplier': mult, **summary})
    chosen = min(validation, key=lambda r: r['total_cash_cost_yuan'])
    model, mult = chosen['model'], chosen['terminal_multiplier']
    selection = {'chosen_model': model, 'chosen_terminal_multiplier': mult,
        'chosen_validation_cost_yuan': chosen['total_cash_cost_yuan'],
        'selection_information_cutoff': '2025-01-31T24:00',
        'selection_start': '2025-01-21', 'selection_end': '2025-01-31',
        'validation_common_initial_kwh': float(warm_s[19,-1]), 'validation_common_terminal_kwh': 6000.,
        'formal_common_initial_kwh': warm['terminal_kwh'], 'validation': validation,
        'main_experiment': f'candidate_{model}',
        'scope': 'historical validation among preregistered candidates, not a global optimum guarantee'}
    write_json(out/'selection.json', selection)
    print('JANUARY SELECTION '+json.dumps({k: selection[k] for k in ('chosen_model','chosen_terminal_multiplier','formal_common_initial_kwh')}, ensure_ascii=False), flush=True)
    days = list(range(31, 34 if args.pilot else 365))
    final = None if args.pilot else 6000.
    initial = warm['terminal_kwh']
    specifications = []
    for m in MODELS:
        best = min((r for r in validation if r['model'] == m), key=lambda r: r['total_cash_cost_yuan'])
        specifications.append((f'candidate_{m}', m, {'terminal_multiplier': best['terminal_multiplier']}))
    specifications += [('point_selected', model, {'mode': 'point', 'terminal_multiplier': mult}),
        ('no_storage_selected', model, {'no_storage': True, 'terminal_multiplier': mult}),
        ('enumerate_selected', model, {'mode': 'enumerate', 'terminal_multiplier': mult}),
        ('v1_mechanism_matched_initial', MODELS[0], {'mode': 'v1_enumerate', 'terminal_multiplier': 1.}),
        ('terminal_0_8', model, {'terminal_multiplier': mult*.8}),
        ('terminal_1_2', model, {'terminal_multiplier': mult*1.2}),
        ('residual_0_8', model, {'terminal_multiplier': mult, 'residual_scale': .8}),
        ('residual_1_2', model, {'terminal_multiplier': mult, 'residual_scale': 1.2}),
        ('seed_20260913', model, {'terminal_multiplier': mult, 'seed': 20260913}),
        ('seed_20260914', model, {'terminal_multiplier': mult, 'seed': 20260914})]
    summaries = []
    for name, m, kwargs in specifications:
        # 无储能基线不消耗起始库存，也不要求补足无法充入的年度储量。
        # 与储能策略的库存差额须单列，禁止隐含为同一终端条件。
        end = None if kwargs.get('no_storage') else final
        summary = experiment(dataset, caches[m], config, out/name, days, initial, final_target=end, **kwargs)
        summaries.append(summary)
        write_json(out/'experiment_summaries.json', summaries)
    alternate = Battery(np.sqrt(.9), np.sqrt(.9))
    aw = experiment(dataset, caches[MODELS[0]], config, out/'warmup_roundtrip90', range(31), 6000., battery=alternate)
    summaries.append(experiment(dataset, caches[model], config, out/'roundtrip90', days, aw['terminal_kwh'],
        battery=alternate, terminal_multiplier=mult, final_target=final))
    write_json(out/'experiment_summaries.json', summaries)
    write_json(out/'completion.json', {'status': 'numerical_results_complete', 'formal_days': len(days),
        'experiments': len(summaries), 'validation_experiments': len(validation), 'pilot': args.pilot,
        'exports_verified': False})
    print('COMPLETE '+str(out), flush=True)


if __name__ == '__main__':
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
