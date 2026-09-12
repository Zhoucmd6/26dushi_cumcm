"""第三问：连续热身、1月验证、2—12月逐日滚动回测；可断点恢复。"""
from pathlib import Path
from dataclasses import asdict
import argparse, csv, hashlib, json, shutil, time
import numpy as np
from forecasting import read_dataset
from microgrid_core import Battery, dispatch_checks, require_checks
from q3_dispatch import solve_roll, settlement, weighted_cvar
from q3_forecasting import read_official, build_cache, make_scenarios, probability_scores, MODES

ROOT=Path(__file__).resolve().parents[1]
BATTERY=Battery(.9,.9)

def write_json(path,value):
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')

def write_csv(path,rows):
    if not rows:return
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)

def run_policy(ds,cache,folder,start,end,initial,policy,*,end_target=True):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    eta=policy.get('eta',.9);battery=Battery(eta,eta)
    if (folder/'summary.json').exists():
        saved=json.loads((folder/'summary.json').read_text(encoding='utf-8'))
        if saved['policy']!=policy or saved['start']!=str(ds.dates[start]) or saved['end']!=str(ds.dates[end-1]) or abs(saved['initial_kwh']-initial)>1e-7:
            raise ValueError('Existing result belongs to different settings; use a new run directory')
        return saved
    write_json(folder/'policy.json',policy)
    days=end-start;n=144
    arrays={k:np.zeros((days,n)) for k in ['Q','R','C','D','E','U','normal_fee']}
    arrays['S']=np.zeros((days,145))
    for k in ['R_versions','C_versions','D_versions']:arrays[k]=np.full((days,4,n),np.nan)
    daily=[];updates=[];state=float(initial);started=time.perf_counter();completed=0
    checkpoint=folder/'checkpoint.npz';metadata=folder/'checkpoint.json'
    if checkpoint.exists() and metadata.exists():
        saved=json.loads(metadata.read_text(encoding='utf-8'));completed=saved['completed_days']
        if saved['policy']!=policy or saved['start']!=start or saved['end']!=end or saved['initial']!=initial:raise ValueError('Incompatible checkpoint')
        arrays.update(dict(np.load(checkpoint)));daily=saved['daily'];updates=saved['updates'];state=saved['state']
    for di,day in enumerate(range(start,end)):
        if di<completed:continue
        S=arrays['S'][di];S[0]=state
        for issue,hour in enumerate([0,6,12,18]):
            t=hour*6;sl=slice(t,t+36)
            active=[0]+[h for j,h in enumerate([6,12,18]) if policy.get('mask',7)&(1<<j) and h<=hour]
            source=max(active)
            L,G,prob,audit=make_scenarios(ds,cache,day,hour,source,mode=policy['mode'],point=policy.get('point',False))
            terminal=6000. if policy.get('daily_terminal',False) or (end_target and day==end-1) else None
            out=solve_roll(ds.prices[t:],L,G,prob,battery=battery,initial_kwh=state,
                           original_q=None if issue==0 else arrays['Q'][di,t:],
                           risk_lambda=policy.get('risk_lambda',0.),alpha=policy.get('alpha',.9),
                           terminal_gamma=policy['gamma'],terminal_kwh=terminal,
                           settlement_mode=policy.get('settlement','net_refund'))
            plan=out['plan']
            if issue==0:arrays['Q'][di]=plan['R']
            for k in ['R','C','D']:
                arrays[k][di,sl]=plan[k][:36];arrays[k+'_versions'][di,issue,t:]=plan[k]
            # 独立按已执行充放电递推实际状态，下一轮接入这个状态。
            c,d=arrays['C'][di,sl],arrays['D'][di,sl]
            S[t+1:t+37]=state+np.cumsum(eta*c-d/eta);state=float(S[t+36])
            deficit=(ds.load_kw[day,sl]-ds.pv_kw[day,sl])/6+c-d-arrays['R'][di,sl]
            arrays['E'][di,sl]=np.maximum(deficit,0);arrays['U'][di,sl]=np.maximum(-deficit,0)
            audit.update(objective_yuan=out['objective_yuan'],scenario_cvar_yuan=out['scenario_cvar_yuan'],
                         terminal_penalty_yuan=out['terminal_penalty_yuan'],solver_gap_yuan=out['gap_yuan'],
                         runtime_s=out['runtime_s'],max_model_residual=max(out['checks'].values()),
                         initial_kwh=float(S[t]),executed_end_kwh=state,
                         solver_route=out['solver_route'],solver_attempts=out['attempts'])
            updates.append(audit)
        Q,R,C,D,E,U=[arrays[k][di] for k in ['Q','R','C','D','E','U']]
        fee=settlement(ds.prices,Q,R,policy.get('settlement','net_refund'));arrays['normal_fee'][di]=fee
        checks=dispatch_checks(C,D,S,battery,float(S[0]))
        if policy.get('daily_terminal'):checks['daily_terminal']=abs(S[-1]-6000)
        checks['balance']=float(abs(R+E+ds.pv_kw[day]/6+D-ds.load_kw[day]/6-C-U).max())
        checks['midnight_commitment']=float(abs(R[:36]-Q[:36]).max())
        checks['negative_purchase']=float(max(0,-Q.min(),-R.min()))
        require_checks(checks,1e-4)
        em=float(E@(5*ds.prices));qcost=float(Q@ds.prices)
        daily.append({'date':ds.dates[day].isoformat(),'q_kwh':float(Q.sum()),'r_kwh':float(R.sum()),
                      'q_fee_yuan':qcost,'up_fee_yuan':float(1.5*ds.prices@np.maximum(R-Q,0)),
                      'down_half_fee_yuan':float(.5*ds.prices@np.maximum(Q-R,0)),
                      'normal_fee_yuan':float(fee.sum()),'emergency_fee_yuan':em,'total_fee_yuan':float(fee.sum()+em),
                      'emergency_kwh':float(E.sum()),'unused_kwh':float(U.sum()),'charge_kwh':float(C.sum()),
                      'discharge_kwh':float(D.sum()),'throughput_kwh':float(C.sum()+D.sum()),
                      'adjust_up_kwh':float(np.maximum(R-Q,0).sum()),'adjust_down_kwh':float(np.maximum(Q-R,0).sum()),
                      'adjusted_intervals':int((abs(R-Q)>1e-6).sum()),
                      'initial_kwh':float(S[0]),'terminal_kwh':state,
                      'max_physical_residual':max(checks.values())})
        if (di+1)%14==0 or day==end-1:
            print(f'{folder.name}: {di+1}/{days}, {ds.dates[day]}, {time.perf_counter()-started:.1f}s',flush=True)
            write_json(folder/'progress.json',{'completed_days':di+1,'days':days,'date':str(ds.dates[day])})
            np.savez_compressed(checkpoint,**arrays)
            write_json(metadata,{'completed_days':di+1,'policy':policy,'start':start,'end':end,'initial':initial,
                                'state':state,'daily':daily,'updates':updates})
    np.savez_compressed(folder/'dispatch.npz',day_indices=np.arange(start,end),**arrays)
    write_csv(folder/'daily.csv',daily);write_json(folder/'updates.json',updates)
    losses=np.array([r['emergency_fee_yuan'] for r in daily]);total=sum(r['total_fee_yuan'] for r in daily)
    summary={'policy':policy,'start':str(ds.dates[start]),'end':str(ds.dates[end-1]),'days':days,
             'total_fee_yuan':total,'normal_fee_yuan':sum(r['normal_fee_yuan'] for r in daily),
             'emergency_fee_yuan':float(losses.sum()),'emergency_kwh':float(arrays['E'].sum()),
             'unused_kwh':float(arrays['U'].sum()),'daily_emergency_cvar90_yuan':weighted_cvar(losses,np.ones(days)/days,.9),
             'daily_emergency_cvar95_yuan':weighted_cvar(losses,np.ones(days)/days,.95),
             'emergency_days':int(np.sum(arrays['E'].sum(axis=1)>1e-6)),
             'initial_kwh':initial,'terminal_kwh':state,'runtime_s':time.perf_counter()-started,
             'max_physical_residual':max(r['max_physical_residual'] for r in daily),
             'max_solver_gap_yuan':max(u['solver_gap_yuan'] for u in updates)}
    for key in ['charge_kwh','discharge_kwh','throughput_kwh','adjust_up_kwh','adjust_down_kwh','adjusted_intervals']:
        summary[key]=sum(r[key] for r in daily)
    summary['solver_runtime_total_s']=sum(u['runtime_s'] for u in updates)
    summary['solver_runtime_max_s']=max(u['runtime_s'] for u in updates)
    write_json(folder/'summary.json',summary);return summary

def score_modes(ds,cache,start,end):
    rows=[]
    for mode in MODES:
        for hour in [0,6,12,18]:
            bags={key:[] for key in ['crps','coverage','width','interval_score','mae','squared_error']};active=[]
            for day in range(start,end):
                _,g,p,_=make_scenarios(ds,cache,day,hour,hour,mode=mode)
                actual=ds.pv_kw[day,hour*6:];scores=probability_scores(g*6,p,actual)
                active.append(actual>0)
                for key in bags:bags[key].append(scores[key])
            mask=np.concatenate(active)
            for subset,sel in [('all',np.ones(len(mask),bool)),('daylight',mask)]:
                if not sel.any():continue
                row={'mode':mode,'issue_hour':hour,'subset':subset,'points':int(sel.sum()),
                     **{key:float(np.concatenate(value)[sel].mean()) for key,value in bags.items()}}
                row['rmse']=float(np.sqrt(row.pop('squared_error')));rows.append(row)
    return rows

if __name__=='__main__':
    from q3_workflow import main
    main()
