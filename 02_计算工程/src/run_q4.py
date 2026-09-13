"""Preregistered historical selection, continuous replay, and Q4 ablations."""
from pathlib import Path
from dataclasses import asdict
import argparse
import csv
import hashlib
import json
import time
import numpy as np
from q4_dispatch import Battery, solve_dispatch, bill, physical_checks
from q4_forecasting import read_inputs, scenarios, realized_signal, classify_signal

ROOT=Path(__file__).resolve().parents[1]


def write_json(path,value):
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    def default(obj):
        if isinstance(obj,np.ndarray):return obj.tolist()
        if isinstance(obj,np.generic):return obj.item()
        raise TypeError(type(obj).__name__)
    Path(path).write_text(json.dumps(value,ensure_ascii=False,indent=2,default=default)+'\n',encoding='utf-8')


def write_csv(path,rows):
    with Path(path).open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def load_cache(name):
    with np.load(ROOT/f'outputs/prepared/{name}.npz') as data:return {k:data[k] for k in data.files}


def prediction_score(values,prob,actual):
    means=prob@values
    order=np.argsort(values,axis=0);v=np.take_along_axis(values,order,axis=0)
    w=np.take_along_axis(np.broadcast_to(prob[:,None],values.shape),order,axis=0)
    cw=np.cumsum(w,axis=0)
    crps=(prob@np.abs(values-actual)-(w*v*(2*cw-w-1)).sum(axis=0))
    low=np.take_along_axis(v,(cw>=.1).argmax(axis=0)[None,:],axis=0)[0]
    high=np.take_along_axis(v,(cw>=.9).argmax(axis=0)[None,:],axis=0)[0]
    return {'abs_error':float(np.abs(means-actual).sum()),'squared_error':float(((means-actual)**2).sum()),
            'crps':float(crps.sum()),'coverage80':int(((actual>=low)&(actual<=high)).sum()),'points':len(actual)}


def experiment(ds,prices,cache,folder,days,initial,*,mode,rho=1.,gamma=.5,k=1,
               point=False,price_correction=True,lookahead=False,terminal=False,
               settlement='refund',method='lp',progress=True):
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    days=list(days)
    config={'mode':mode,'rho':rho,'gamma':gamma,'k':k,'point':point,'price_correction':price_correction,
            'lookahead':lookahead,'days':days,'initial':initial,'terminal':terminal,'settlement':settlement,'method':method}
    if (folder/'summary.json').exists():
        if json.loads((folder/'config.json').read_text('utf-8'))!=config:
            raise ValueError('Refusing to resume an experiment with different settings')
        return json.loads((folder/'summary.json').read_text('utf-8'))
    write_json(folder/'config.json',config)
    arrays={key:np.zeros((len(days),145 if key=='S' else 144)) for key in ('Q','R','C','D','S','E','U','forecast_price')}
    arrays['day_indices']=np.array(days)
    rows=[];logs=[];maxima={};state=float(initial);battery=Battery()
    total_runtime=0.;scores=dict(abs_error=0.,squared_error=0.,crps=0.,coverage80=0,points=0)
    groups_count={};started=time.perf_counter();optimizations=0
    for ix,day in enumerate(days):
        q=None;before=None;day_initial=state;planned_cost=adjustment_cost=emergency_cost=0.
        arrays['S'][ix,0]=state
        issues=(0,) if mode=='q42' else (0,1,2,3)
        day_solver=0.
        for issue in issues:
            t=issue*36;count=144 if mode=='q42' else 36
            observed_group=None
            if before is not None and 'grouping' in before and 'centers' in before['grouping']:
                observed_group=classify_signal(realized_signal(ds,prices,cache,day,issue-1,price_correction),before['grouping'])
            net,price,prob,group,meta=scenarios(ds,prices,cache,day,issue,mode=mode,point=point,
                price_correction=price_correction,groups=k if lookahead else 1)
            final_min=6000. if terminal and day==days[-1] else None
            solved=solve_dispatch(net,price,prob,state,mode=mode,original_q=None if q is None else q[t:],
                groups=group if lookahead else None,prefix=36,rho=rho,gamma=gamma,
                terminal_min=final_min,battery=battery,method=method,settlement=settlement)
            if q is None:q=solved['Q'].copy();arrays['Q'][ix]=q
            node=solved['nodes'][0]
            r,c,d=[node[key][:count] for key in ('R','C','D')]
            s=np.r_[state,state+np.cumsum(battery.eta_c*c-d/battery.eta_d)]
            actual_net=(ds.load_kw[day,t:t+count]-ds.pv_kw[day,t:t+count])/6
            actual_price=prices[day,t:t+count]
            paid=bill(actual_price,actual_net,q[t:t+count],r,c,d,adjustment=mode=='q43',settlement=settlement)
            for key,values in [('R',r),('C',c),('D',d),('E',paid['E']),('U',paid['U']),('forecast_price',prob@price[:,:count])]:
                arrays[key][ix,t:t+count]=values
            arrays['S'][ix,t:t+count+1]=s
            check=physical_checks(actual_net,r,c,d,s,state,battery,final_min if t+count==144 else None)
            check['planned_execution_state']=float(np.max(np.abs(s-node['S'][:count+1])))
            for key,value in {**solved['checks'],**check}.items():maxima[key]=max(maxima.get(key,0.),value)
            if max(check.values())>2e-5:raise RuntimeError(f'Physical replay violation: {check}')
            block_score=prediction_score(price[:,:count],prob,actual_price)
            for key in scores:scores[key]+=block_score[key]
            state=float(s[-1]);planned_cost+=paid['planned_cost'];adjustment_cost+=paid['adjustment_cost'];emergency_cost+=paid['emergency_cost']
            day_solver+=solved['runtime_s'];optimizations+=1
            groups_count[str(solved['branches'])]=groups_count.get(str(solved['branches']),0)+1
            # Future group actions are recorded for audit, never executed at this update.
            logs.append({**meta,'observed_previous_group':observed_group,'initial_stock':float(s[0]),
                'executed_terminal_stock':state,'objective':solved['objective'],'expected_cash':solved['expected_cash'],
                'terminal_penalty':solved['terminal_cost'],'solver_runtime_s':solved['runtime_s'],
                'solver_method':method,'mip_gap':solved['mip_gap'],'purified_charge_kwh':solved['purified_charge_kwh'],
                'branch_nodes':[{key:n[key] for key in ('start','stop','ids','prob','R','C','D','S')} for n in solved['nodes']],
                'checks':solved['checks']})
            before=meta
        total_runtime+=day_solver
        rows.append({'date':ds.dates[day].isoformat(),'planned_kwh':float(q.sum()),'executed_normal_kwh':float(arrays['R'][ix].sum()),
            'planned_cost_yuan':planned_cost,'adjustment_cost_yuan':adjustment_cost,'emergency_cost_yuan':emergency_cost,
            'cash_cost_yuan':planned_cost+adjustment_cost+emergency_cost,'emergency_kwh':float(arrays['E'][ix].sum()),
            'unused_kwh':float(arrays['U'][ix].sum()),'charge_kwh':float(arrays['C'][ix].sum()),'discharge_kwh':float(arrays['D'][ix].sum()),
            'up_kwh':float(np.maximum(arrays['R'][ix]-q,0.).sum()),'down_kwh':float(np.maximum(q-arrays['R'][ix],0.).sum()),
            'initial_kwh':day_initial,'terminal_kwh':state,'solver_runtime_s':day_solver})
        if progress and (ix==0 or (ix+1)%30==0 or ix+1==len(days)):
            print(f'{folder.name}: {ix+1}/{len(days)}, cash={sum(r["cash_cost_yuan"] for r in rows):.2f}, {time.perf_counter()-started:.1f}s',flush=True)
    np.savez_compressed(folder/'trajectory.npz',**arrays)
    write_csv(folder/'daily.csv',rows)
    write_json(folder/'planning_log.json',logs)
    summary={'name':folder.name,'days':len(days),'initial_kwh':initial,'terminal_kwh':state,**config,
        'cash_cost_yuan':sum(r['cash_cost_yuan'] for r in rows),
        **{key:sum(r[key] for r in rows) for key in ('planned_cost_yuan','adjustment_cost_yuan','emergency_cost_yuan','emergency_kwh','unused_kwh','charge_kwh','discharge_kwh','up_kwh','down_kwh')},
        'price_mae':scores['abs_error']/scores['points'],'price_rmse':np.sqrt(scores['squared_error']/scores['points']),
        'price_crps':scores['crps']/scores['points'],'price_coverage80':scores['coverage80']/scores['points'],
        'optimization_count':optimizations,'solver_runtime_s':total_runtime,'wall_runtime_s':time.perf_counter()-started,
        'checks_maxima':maxima,'group_counts':groups_count,
        'end_at_min':sum(abs(r['terminal_kwh']-1200)<1e-4 for r in rows),
        'end_at_max':sum(abs(r['terminal_kwh']-10800)<1e-4 for r in rows)}
    # Config contains the exact day indices; expose the count separately.
    summary['days']=len(days)
    write_json(folder/'summary.json',summary)
    return summary


def select_parameters(ds,prices,folder):
    warmcache,cache=load_cache('warmup'),load_cache('formal')
    warms={}
    for mode in ('q42','q43'):
        warms[mode]=experiment(ds,prices,warmcache,folder/('warmup_'+mode),range(31),6000.,
                               mode=mode,rho=0. if mode=='q42' else 1.,gamma=.5)
    choices={'selection_cutoff':'2025-01-31T24:00','relative_tie_tolerance':1e-4,'warmup':warms}
    validation=[]
    for mode in ('q42','q43'):
        with np.load(folder/('warmup_'+mode)/'trajectory.npz') as z:initial=float(z['S'][19,-1])
        terminal_tests=[]
        for gamma in (0.,.25,.5,1.):
            name=f'validate_{mode}_gamma_{gamma:g}'
            s=experiment(ds,prices,cache,folder/name,range(20,31),initial,mode=mode,gamma=gamma,terminal=True)
            terminal_tests.append(s);validation.append(s)
        gamma=min(terminal_tests,key=lambda x:(x['cash_cost_yuan'],x['gamma']))['gamma']
        candidates=[]
        for value in ((0.,.25,.5,.75,1.) if mode=='q42' else (1,2,3)):
            key='rho' if mode=='q42' else 'k'
            kw={'rho':value} if mode=='q42' else {'k':value,'lookahead':True}
            s=experiment(ds,prices,cache,folder/f'validate_{mode}_{key}_{value:g}',range(20,31),initial,
                          mode=mode,gamma=gamma,terminal=True,**kw)
            candidates.append(s);validation.append(s)
        minimum=min(s['cash_cost_yuan'] for s in candidates)
        key='rho' if mode=='q42' else 'k'
        best=min((s for s in candidates if s['cash_cost_yuan']<=minimum+abs(minimum)*1e-4),key=lambda s:s[key])
        choices[mode]={'gamma':gamma,key:best[key],'validation_cash_yuan':best['cash_cost_yuan'],
                       'validation_initial_kwh':initial,'formal_initial_kwh':warms[mode]['terminal_kwh']}
    choices['validation']=validation
    choices['q42_main']='q42_selected'
    choices['q43_main']='q43_common' if choices['q43']['k']==1 else f'q43_lookahead_k{choices["q43"]["k"]}'
    write_json(folder/'selection.json',choices)
    return choices


def definitions(selection):
    g2,g3=selection['q42']['gamma'],selection['q43']['gamma']
    return {
        'q42_point':dict(mode='q42',gamma=g2,point=True,rho=0.),
        'q42_pair':dict(mode='q42',gamma=g2,rho=1.),
        'q42_independent':dict(mode='q42',gamma=g2,rho=0.),
        'q42_selected':dict(mode='q42',gamma=g2,rho=selection['q42']['rho']),
        'q42_gamma0':dict(mode='q42',gamma=0.,rho=selection['q42']['rho']),
        'q43_point':dict(mode='q43',gamma=g3,point=True),
        'q43_no_price_correction':dict(mode='q43',gamma=g3,price_correction=False),
        'q43_common':dict(mode='q43',gamma=g3),
        'q43_lookahead_k2':dict(mode='q43',gamma=g3,k=2,lookahead=True),
        'q43_lookahead_k3':dict(mode='q43',gamma=g3,k=3,lookahead=True),
        'q43_gamma0':dict(mode='q43',gamma=0.,k=selection['q43']['k'],lookahead=selection['q43']['k']>1),
    }


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='full_q4_20260913')
    parser.add_argument('--selection-only',action='store_true');parser.add_argument('--experiment')
    parser.add_argument('--pilot',action='store_true');args=parser.parse_args()
    folder=ROOT/'outputs'/args.run;folder.mkdir(exist_ok=True)
    ds,prices,official=read_inputs(ROOT/'data/processed')
    manifest={'inputs':ds.sources,'model':hashlib.sha256((ROOT/'model/model-q4.tex').read_bytes()).hexdigest(),
              'numeric_code':{n:hashlib.sha256((ROOT/'src'/n).read_bytes()).hexdigest() for n in ('q4_dispatch.py','q4_forecasting.py','run_q4.py')},
              'pilot':args.pilot,'risk_weight':0,'settlement':'delivery_real_price_original_commitment_once',
              'battery':asdict(Battery()),'residual_window':28,'residual_first_day_index':7,
              'gamma_candidates':[0,.25,.5,1.],'rho_candidates':[0,.25,.5,.75,1.],'K_candidates':[1,2,3]}
    if (folder/'manifest.json').exists() and json.loads((folder/'manifest.json').read_text('utf-8'))!=manifest:
        raise ValueError('Run source or inputs changed; start a new run name')
    write_json(folder/'manifest.json',manifest)
    selection=json.loads((folder/'selection.json').read_text('utf-8')) if (folder/'selection.json').exists() else select_parameters(ds,prices,folder)
    if args.selection_only:return
    cache=load_cache('formal');experiments=definitions(selection)
    selected=[args.experiment] if args.experiment else list(experiments)
    for name in selected:
        spec=experiments[name];mode=spec['mode']
        experiment(ds,prices,cache,folder/name,range(31,34 if args.pilot else 365),
                   selection[mode]['formal_initial_kwh'],terminal=not args.pilot,**spec)
    print('COMPLETE',args.run,','.join(selected),flush=True)


if __name__=='__main__':main()
