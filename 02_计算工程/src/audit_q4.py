"""Independent algebraic replay of stored decisions and all conditional plans.

Does not call optimizer validation or billing helpers. Scenario generators are
reused only to reconstruct the exact archived optimization information set.
"""
from pathlib import Path
import argparse,csv,json,hashlib
import numpy as np
from q4_forecasting import read_inputs,scenarios
from run_q4 import ROOT,load_cache,write_json

TOL=2e-5

def audit_experiment(folder,ds,prices,cache):
    config=json.loads((folder/'config.json').read_text('utf-8'))
    summary=json.loads((folder/'summary.json').read_text('utf-8'))
    with np.load(folder/'trajectory.npz') as z:a={k:z[k] for k in z.files}
    daily=list(csv.DictReader((folder/'daily.csv').open(encoding='utf-8-sig')))
    logs=json.loads((folder/'planning_log.json').read_text('utf-8'))
    days=a['day_indices'];mode=config['mode'];maxima={}
    def check(key,error):
        error=float(np.max(np.abs(error)))
        maxima[key]=max(maxima.get(key,0.),error)
        if error>TOL:raise AssertionError(f'{folder.name} {key}: {error}')
    assert days.tolist()==config['days'] and len(daily)==len(days)
    for key in ('Q','R','C','D','E','U'):assert a[key].shape==(len(days),144) and np.isfinite(a[key]).all()
    assert a['S'].shape==(len(days),145) and np.isfinite(a['S']).all()
    check('cross_day_stock',a['S'][1:,0]-a['S'][:-1,-1] if len(days)>1 else 0)
    check('initial_stock',a['S'][0,0]-config['initial'])
    check('stock_recurrence',np.diff(a['S'],axis=1)-.9*a['C']+a['D']/.9)
    check('stock_bounds',np.maximum(np.maximum(1200-a['S'],a['S']-10800),0))
    check('power_bounds',np.maximum(np.maximum(a['C'],a['D'])-5000/6,0))
    check('nonnegative',np.maximum(-np.stack([a[k] for k in ('Q','R','C','D','E','U')]),0))
    check('charge_discharge_exclusive',np.minimum(a['C'],a['D']))
    if config['terminal']:check('year_end_stock',max(6000-a['S'][-1,-1],0))
    net=(ds.load_kw[days]-ds.pv_kw[days])/6;p=prices[days]
    emergency=np.maximum(net+a['C']-a['D']-a['R'],0)
    unused=np.maximum(a['R']+a['D']-a['C']-net,0)
    check('emergency_definition',a['E']-emergency);check('unused_definition',a['U']-unused)
    check('power_balance',a['R']+a['E']+a['D']-a['C']-a['U']-net)
    up=np.maximum(a['R']-a['Q'],0);down=np.maximum(a['Q']-a['R'],0)
    if mode=='q42':check('fixed_daily_commitment',a['R']-a['Q'])
    else:check('midnight_executed_commitment',a['R'][:,:36]-a['Q'][:,:36])
    adjustment=np.zeros_like(p) if mode=='q42' else p*(1.5*up+(-.5 if config['settlement']=='refund' else .5)*down)
    costs={'planned_cost_yuan':(p*a['Q']).sum(axis=1),'adjustment_cost_yuan':adjustment.sum(axis=1),
           'emergency_cost_yuan':(5*p*emergency).sum(axis=1)}
    costs['cash_cost_yuan']=sum(costs.values())
    for key,v in costs.items():
        check('daily_'+key,v-np.array([float(r[key]) for r in daily]))
        check('total_'+key,v.sum()-summary[key])
    for key,values in [('emergency_kwh',emergency),('unused_kwh',unused),('charge_kwh',a['C']),('discharge_kwh',a['D']),('up_kwh',up),('down_kwh',down)]:
        check('total_'+key,values.sum()-summary[key])
    expected_issues=1 if mode=='q42' else 4
    assert len(logs)==expected_issues*len(days)
    branch_count=0;scenario_points=0
    for j,log in enumerate(logs):
        row=j//expected_issues;issue=j%expected_issues;day=int(days[row]);t=36*issue;n=144-t
        count=144 if mode=='q42' else 36
        assert log['day']==ds.dates[day].isoformat() and log['hour']==6*issue
        assert all(h<day for h in log['history_days']) and len(set(log['history_days']))==len(log['history_days'])
        assert log['price_prefix_end_index']==t and log['official_visible_issue']==(6*issue if mode=='q43' else None)
        nn,pp,prob,groups,meta=scenarios(ds,prices,cache,day,issue,mode=mode,point=config['point'],
              price_correction=config['price_correction'],groups=config['k'] if config['lookahead'] else 1)
        assert meta['history_days']==log['history_days'] and meta['scenario_count']==log['scenario_count']
        if 'grouping' in meta:assert meta['grouping']==log['grouping']
        shape=nn.shape;rr=np.zeros(shape);cc=np.zeros(shape);dd=np.zeros(shape);assigned=np.zeros(shape,int)
        branch_terminal=0;root=log['branch_nodes'][0]
        check('recorded_initial',root['S'][0]-a['S'][row,t])
        for key in ('R','C','D'):check('executed_'+key,np.asarray(root[key])[:count]-a[key][row,t:t+count])
        check('executed_stock',np.asarray(root['S'])[:count+1]-a['S'][row,t:t+count+1])
        for k,node in enumerate(log['branch_nodes']):
            ids=np.array(node['ids'],int);b,e=node['start'],node['stop'];s=np.asarray(node['S'])
            r,c,d=[np.asarray(node[key]) for key in ('R','C','D')]
            assert len(s)==e-b+1 and len(r)==e-b
            check('branch_stock_recurrence',np.diff(s)-.9*c+d/.9)
            check('branch_stock_bounds',np.maximum(np.maximum(1200-s,s-10800),0))
            check('branch_power_bounds',np.maximum(np.maximum(c,d)-5000/6,0))
            check('branch_exclusivity',np.minimum(c,d))
            check('branch_nonnegative',np.maximum(-np.r_[r,c,d],0))
            check('branch_probability',prob[ids].sum()-node['prob'])
            if k:
                check('shared_bridge',s[0]-root['S'][-1])
                assert groups is not None and len(set(groups[ids].tolist()))==1
                assert len(ids)>=5 or len(set(groups.tolist()))==1
            rr[ids,b:e]=r;cc[ids,b:e]=c;dd[ids,b:e]=d;assigned[ids,b:e]+=1
            if e==n:
                branch_terminal+=config['gamma']*prob[ids].sum()*max(6000-s[-1],0)
                if config['terminal'] and row==len(days)-1:check('branch_final_minimum',max(6000-s[-1],0))
            branch_count+=1
        assert (assigned==1).all()
        q=a['Q'][row,t:]
        if mode=='q42':check('plan_fixed_q',rr-q)
        normal=rr if mode=='q42' else q+1.5*np.maximum(rr-q,0)+(-.5 if config['settlement']=='refund' else .5)*np.maximum(q-rr,0)
        weighted=prob[:,None]*(config['rho']*pp+(1-config['rho'])*(prob@pp))
        expected=float((prob[:,None]*pp*normal).sum()+5*(weighted*np.maximum(nn+cc-dd-rr,0)).sum())
        check('expected_cash_objective',expected-log['expected_cash'])
        check('terminal_penalty_objective',branch_terminal-log['terminal_penalty'])
        check('total_objective',expected+branch_terminal-log['objective'])
        scenario_points+=nn.size
    return {'experiment':folder.name,'days':len(days),'optimizations':len(logs),'conditional_nodes':branch_count,
            'scenario_time_pairs':scenario_points,'tolerance':TOL,'maximum_errors':maxima,
            'maximum_error':max(maxima.values()),'cash_cost_yuan':float(costs['cash_cost_yuan'].sum()),
            'trajectory_sha256':hashlib.sha256((folder/'trajectory.npz').read_bytes()).hexdigest()}

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--run',default='full_q4_20260913');a=parser.parse_args()
    run=ROOT/'outputs'/a.run;ds,prices,_=read_inputs(ROOT/'data/processed')
    caches={name:load_cache(name) for name in ('formal','warmup')};results=[]
    for folder in sorted(run.iterdir()):
        if not (folder/'summary.json').is_file():continue
        result=audit_experiment(folder,ds,prices,caches['warmup' if folder.name.startswith('warmup_') else 'formal'])
        results.append(result);print(folder.name,result['maximum_error'],flush=True)
    write_json(run/'audit.json',{'experiments':results,'experiment_count':len(results),
        'days_checked':sum(r['days'] for r in results),'optimizations_checked':sum(r['optimizations'] for r in results),
        'maximum_error':max(r['maximum_error'] for r in results),'tolerance':TOL})

if __name__=='__main__':main()
