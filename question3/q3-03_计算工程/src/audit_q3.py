"""独立复算落地轨迹、账单与信息时序；不调用优化器。"""
import json
from pathlib import Path
import numpy as np
from forecasting import read_dataset
from microgrid_core import Battery, dispatch_checks, require_checks
from run_q3 import ROOT,write_json

def audit_arrays(ds,a,policy,summary,updates):
    ids=a['day_indices'];n=len(ids)
    if list(ids)!=list(range(31,365)):raise ValueError('Missing or reordered evaluation dates')
    if len(updates)!=n*4:raise ValueError('Missing rolling decision audit')
    for key in ['Q','R','C','D','E','U','normal_fee']:
        if a[key].shape!=(n,144) or not np.isfinite(a[key]).all():raise ValueError('Invalid '+key)
    checks={};eta=policy.get('eta',.9);battery=Battery(eta,eta)
    for i,day in enumerate(ids):
        physical=dispatch_checks(a['C'][i],a['D'][i],a['S'][i],battery,float(a['S'][i,0]))
        for key,value in physical.items():checks[key]=max(checks.get(key,0),value)
        for j,hour in enumerate([0,6,12,18]):
            u=updates[i*4+j];t=hour*6
            allowed=[0]+[h for bit,h in enumerate([6,12,18]) if policy.get('mask',7)&(1<<bit) and h<=hour]
            if u['official_issue_hour']!=max(allowed) or u['date']!=str(ds.dates[day]) or u['decision_hour']!=hour:
                raise ValueError('Future or disallowed official forecast')
            calibration=policy.get('calibration',{})
            if (u['window_days']!=calibration.get('window',28) or u['bin_hours']!=calibration.get('bin_hours',6)
                or u['shrink_days']!=calibration.get('shrink',7.) or u['scale_mode']!=policy['mode']):
                raise ValueError('Calibration differs from registered policy')
            if u['calibration_end']!=str(ds.dates[day-1]) or any(x>=day or x<max(7,day-u['window_days']) for x in u['sampled_day_indices']):
                raise ValueError('Historical sample look-ahead')
            if abs(u['first_lead_hours']-(hour-u['official_issue_hour']+1/6))>1e-10:
                raise ValueError('Forecast lead reset')
            for key in ['R','C','D']:
                v=a[key+'_versions'][i,j]
                if np.isfinite(v[:t]).any() or not np.isfinite(v[t:]).all():raise ValueError('Invalid remaining horizon archive')
                checks['executed_version']=max(checks.get('executed_version',0),float(abs(a[key][i,t:t+36]-v[t:t+36]).max()))
        checks['original_commitment']=max(checks.get('original_commitment',0),float(abs(a['Q'][i]-a['R_versions'][i,0]).max()))
        if i:
            checks['previous_state_used']=max(checks.get('previous_state_used',0),abs(a['S'][i,0]-a['S'][i-1,-1]))
    # 独立结算：主口径使用等价公式 p*r + 0.5*p*abs(r-q)。
    p=ds.prices[None,:];q,r=a['Q'],a['R']
    normal=p*r+.5*p*abs(r-q)
    if policy.get('settlement')=='no_refund_penalty':normal+=p*np.maximum(q-r,0)
    deficit=(ds.load_kw[ids]-ds.pv_kw[ids])/6+a['C']-a['D']-r
    checks.update(fee_formula=float(abs(normal-a['normal_fee']).max()),
        emergency_replay=float(abs(np.maximum(deficit,0)-a['E']).max()),
        unused_replay=float(abs(np.maximum(-deficit,0)-a['U']).max()),
        annual_fee=abs(float((normal+5*p*a['E']).sum())-summary['total_fee_yuan']),
        across_day=float(abs(a['S'][1:,0]-a['S'][:-1,-1]).max()),
        initial=abs(a['S'][0,0]-summary['initial_kwh']),year_end=abs(a['S'][-1,-1]-6000),
        nonnegative_purchase=float(max(0,-q.min(),-r.min())),
        first_six_hours=float(abs(q[:,:36]-r[:,:36]).max()))
    if policy.get('daily_terminal'):checks['daily_terminal']=float(abs(a['S'][:,-1]-6000).max())
    require_checks(checks,1e-4)
    return {'days':n,'updates':len(updates),'ten_minute_intervals':n*144,'checks':checks,
            'max_residual':max(checks.values()),'scenario_draws':64,
            'available_independent_days_min':min(u['available_independent_days'] for u in updates),
            'available_independent_days_max':max(u['available_independent_days'] for u in updates),
            'unique_sampled_days_min':min(u['unique_sampled_days'] for u in updates),
            'unique_sampled_days_max':max(u['unique_sampled_days'] for u in updates)}

def audit_run(run):
    run=Path(run);ds=read_dataset(ROOT/'data/processed');records={}
    for folder in sorted((run/'experiments').iterdir()):
        if not (folder/'summary.json').exists():continue
        j=lambda name:json.loads((folder/name).read_text(encoding='utf-8'))
        records[folder.name]=audit_arrays(ds,dict(np.load(folder/'dispatch.npz')),j('policy.json'),j('summary.json'),j('updates.json'))
    write_json(run/'independent_audit.json',records);return records

if __name__=='__main__':
    import sys
    result=audit_run(sys.argv[1]);print(json.dumps({k:v['max_residual'] for k,v in result.items()}))
