"""PDF主模型B3、B0—B4对照与严格历史验证；独立进程只写各自实验目录。"""
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor,as_completed
from itertools import product
from contextlib import redirect_stdout
import argparse,json,hashlib,shutil
import numpy as np
from run_q3 import ROOT,run_policy,write_json,write_csv,score_modes
from forecasting import read_dataset
from q3_forecasting import build_cache,read_official,MODES

GRID=[{'window':w,'shrink':s,'bin_hours':b,'floor':25.,'interpretation':'linear_nodes'}
      for w,s,b in product([14,28],[3.,7.],[3,6])]

def config_id(c):
    return f"w{c['window']}_s{c['shrink']:g}_b{c['bin_hours']}"+('_mean' if c.get('interpretation')=='hourly_mean' else '')

def get_cache(run,c,ds=None):
    path=Path(run)/'caches'/f'{config_id(c)}.npz'
    if path.exists():return dict(np.load(path))
    ds=read_dataset(ROOT/'data/processed') if ds is None else ds
    official=read_official(ROOT/'data/processed/pv_forecasts.csv',ds.dates)
    cache=build_cache(ds,official,**c);path.parent.mkdir(exist_ok=True)
    np.savez_compressed(path,**cache);return cache

def validation_job(run,c,initial,gammas):
    run=Path(run);name=config_id(c);folder=run/'validation'/name;folder.mkdir(parents=True,exist_ok=True)
    ds=read_dataset(ROOT/'data/processed');cache=get_cache(run,c,ds)
    scores=score_modes(ds,cache,20,31);write_csv(folder/'probability_scores.csv',scores)
    crps={m:sum(r['crps']*r['points'] for r in scores if r['mode']==m and r['subset']=='daylight')/
             sum(r['points'] for r in scores if r['mode']==m and r['subset']=='daylight') for m in MODES}
    candidates=[]
    with (folder/'progress.log').open('a',encoding='utf-8') as log,redirect_stdout(log):
        for mode in MODES:
            for j,gamma in enumerate(gammas):
                policy={'mode':mode,'gamma':gamma,'mask':7,'calibration':c}
                result=run_policy(ds,cache,folder/f'{mode}_gamma{j}',20,31,initial,policy)
                candidates.append({'config_id':name,'calibration':c,'mode':mode,'gamma':gamma,
                    'total_fee_yuan':result['total_fee_yuan'],'daylight_crps_kw':crps[mode],
                    'exponential_eligible':crps['exponential']<=min(crps['pooled'],crps['binned'])})
    write_json(folder/'candidates.json',candidates);return candidates

def experiment_job(run,name,policy,initial,start=31,end=365):
    run=Path(run);ds=read_dataset(ROOT/'data/processed');cache=get_cache(run,policy['calibration'],ds)
    folder=run/('experiments' if start==31 else 'risk_validation')/name;folder.mkdir(parents=True,exist_ok=True)
    with (folder/'progress.log').open('a',encoding='utf-8') as log,redirect_stdout(log):
        return name,run_policy(ds,cache,folder,start,end,initial,policy)

def build_experiments(selection,gamma0):
    selected=selection['selected']
    base={k:selected[k] for k in ['mode','gamma','calibration']};base['mask']=7
    experiments={'main':base,'point_official':{**base,'point':True},
                 'scale_pooled':{**base,'mode':'pooled'}}
    experiments.update({f'info_mask{m}':{**base,'mask':m} for m in range(7)})
    for mode in ('binned','exponential'):
        if mode!=base['mode']:experiments['scale_'+mode]={**base,'mode':mode}
    for alpha in (.90,.95):
        for lam in (.05,.1):
            name=f'cvar{round(alpha*100)}'+('_lambda005' if lam==.05 else '')
            experiments[name]={**base,'risk_lambda':lam,'alpha':alpha}
    if base['gamma']!=0:experiments['terminal_zero']={**base,'gamma':0.}
    else:experiments['terminal_soft']={**base,'gamma':gamma0}
    experiments['terminal_hard']={**base,'gamma':0.,'daily_terminal':True}
    experiments['settlement_alternative']={**base,'settlement':'no_refund_penalty'}
    experiments['efficiency_roundtrip90']={**base,'eta':float(np.sqrt(.9))}
    experiments['hourly_mean']={**base,'calibration':{**base['calibration'],'interpretation':'hourly_mean'}}
    for key in ['probability_selected','global_cash_selected']:
        p={k:selection[key][k] for k in ['mode','gamma','calibration']};p['mask']=7
        if p!=base:experiments[key]=p
    return experiments

def run_jobs(jobs,workers,function):
    with ProcessPoolExecutor(max_workers=workers) as pool:
        pending={pool.submit(function,*args):name for name,args in jobs}
        for future in as_completed(pending):
            value=future.result();print('COMPLETED '+pending[future],flush=True);yield value

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',required=True)
    parser.add_argument('--workers',type=int,default=3);parser.add_argument('--phase',choices=['all','calibrate','evaluate'],default='all')
    parser.add_argument('--only',help='Comma-separated evaluation experiments')
    args=parser.parse_args();run=Path(args.run).resolve();run.mkdir(parents=True,exist_ok=True)
    ds=read_dataset(ROOT/'data/processed');gamma0=float(ds.prices.min()/.9);gammas=[0.,gamma0,2*gamma0]
    inputs={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'data').rglob('*') if p.is_file() and p.suffix in ('.xlsx','.csv')}
    numeric={n:hashlib.sha256((ROOT/'src'/n).read_bytes()).hexdigest() for n in
             ['q3_dispatch.py','q3_forecasting.py','run_q3.py','q3_workflow.py','forecasting.py','microgrid_core.py']}
    cfg={'input_sha256':inputs,'numerical_code_sha256':numeric,'calibration_grid':GRID,'gamma_candidates':gammas,
         'validation_dates':['2025-01-21','2025-01-31'],'evaluation_dates':['2025-02-01','2025-12-31'],
         'draws':64,'seed':20260912,'sigma_floor_kw':25.,'battery_eta_charge':.9,'battery_eta_discharge':.9,
         'primary_model':'B3 conditional lead scale, risk neutral; B4 only adds CVaR to exactly this B3',
         'selection_rule':'Reject exponential per configuration if its January daylight CRPS exceeds pooled or binned; select conditional model by January cash. Also report unrestricted cash selection and conditional CRPS selection.',
         'risk_grid':{'alpha':[.9,.95],'lambda':[0.,.05,.1]},
         'warmup_policy':'Jan1 idle battery and zero Q; Jan2-31 predeclared binned window28 shrink7 bin6 gamma=min_price/eta, causal continuous state.',
         'evaluation_terminal_kwh':6000,'settlement':'p*q+1.5*p*positive(r-q)-0.5*p*positive(q-r)'}
    configpath=run/'run_config.json'
    if configpath.exists():
        old=json.loads(configpath.read_text(encoding='utf-8'))
        if old!=cfg:raise ValueError('Changed numerical configuration; use a new run directory')
    else:write_json(configpath,cfg);shutil.copytree(ROOT/'src',run/'code_snapshot',dirs_exist_ok=True)
    if args.phase in ('all','calibrate'):
        warm_c=GRID[-1];cache=get_cache(run,warm_c,ds)
        warm=run_policy(ds,cache,run/'warmup_jan2_31',1,31,6000.,{'mode':'binned','gamma':gamma0,'mask':7,'calibration':warm_c},end_target=False)
        write_json(run/'jan1_cold_start.json',{'initial_kwh':6000,'terminal_kwh':6000,'planned_kwh':0.,
              'emergency_fee_yuan':float(np.maximum(ds.load_kw[0]-ds.pv_kw[0],0)/6@(5*ds.prices)),
              'reason':'No pre-2025 observed load history was supplied.'})
        initial=float(dict(np.load(run/'warmup_jan2_31/dispatch.npz'))['S'][18,-1])
        candidates=[]
        jobs=[(config_id(c),(str(run),c,initial,gammas)) for c in GRID]
        for result in run_jobs(jobs,args.workers,validation_job):candidates.extend(result)
        candidates.sort(key=lambda x:(x['config_id'],x['mode'],x['gamma']))
        eligible=[x for x in candidates if x['mode']!='exponential' or x['exponential_eligible']]
        conditional=[x for x in eligible if x['mode']!='pooled']
        selected=min(conditional,key=lambda x:x['total_fee_yuan'])
        # 同一gamma下比较概率评分选参和决策费用选参，避免混入终端偏好变化。
        prob=min([x for x in conditional if x['gamma']==selected['gamma']],key=lambda x:(x['daylight_crps_kw'],x['config_id']))
        selection={'selected':selected,'global_cash_selected':min(eligible,key=lambda x:x['total_fee_yuan']),
                   'probability_selected':prob,'all_candidates':candidates,'initial_validation_kwh':initial,
                   'feb1_initial_kwh':warm['terminal_kwh'],'selection_date':'2025-01-31 after 24:00'}
        write_json(run/'selection.json',selection)
        write_csv(run/'january_validation_grid.csv',[{k:v for k,v in x.items() if k!='calibration'} for x in candidates])
        print('SELECTED '+json.dumps(selected),flush=True)
    selection=json.loads((run/'selection.json').read_text(encoding='utf-8'))
    experiments=build_experiments(selection,gamma0);write_json(run/'experiment_registry.json',experiments)
    if args.phase in ('all','calibrate'):
        jobs=[(name,(str(run),name,p,selection['initial_validation_kwh'],20,31)) for name,p in experiments.items() if name.startswith('cvar')]
        risk=[]
        for name,result in run_jobs(jobs,args.workers,experiment_job):risk.append({'name':name,**result})
        write_json(run/'january_risk_validation.json',risk)
    if args.phase in ('all','evaluate'):
        if args.only:
            names=args.only.split(',')
            if set(names)-set(experiments):raise ValueError('Unknown experiment')
            experiments={n:experiments[n] for n in names}
        # Build unique caches once before independent experiments start.
        for p in experiments.values():get_cache(run,p['calibration'],ds)
        main_cache=get_cache(run,selection['selected']['calibration'],ds)
        np.savez_compressed(run/'forecast_cache.npz',**main_cache)
        write_csv(run/'annual_probability_scores.csv',score_modes(ds,main_cache,31,365))
        jobs=[(name,(str(run),name,p,selection['feb1_initial_kwh'])) for name,p in experiments.items()]
        for name,result in run_jobs(jobs,args.workers,experiment_job):print(f"{name}: {result['total_fee_yuan']:.6f} yuan",flush=True)
    print('Q3 WORKFLOW COMPLETED '+str(run),flush=True)

if __name__=='__main__':main()
