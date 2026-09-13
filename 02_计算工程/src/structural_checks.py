"""Real-data LP/MILP equivalence and same-information lookahead containment."""
import argparse,json
import numpy as np
from q4_dispatch import solve_dispatch
from q4_forecasting import read_inputs,scenarios
from run_q4 import ROOT,load_cache,write_json

DATES=('2025-03-20','2025-06-21','2025-09-23','2025-12-21')

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default='full_q4_20260913');arg=p.parse_args()
    run=ROOT/'outputs'/arg.run;selection=json.loads((run/'selection.json').read_text('utf-8'))
    ds,prices,_=read_inputs(ROOT/'data/processed');cache=load_cache('formal');rows=[]
    for mode in ('q42','q43'):
        with np.load(run/selection[mode+'_main']/'trajectory.npz') as z:a={k:z[k] for k in z.files}
        for date in DATES:
            day=[str(d) for d in ds.dates].index(date);row=day-31
            for issue in ((0,) if mode=='q42' else (0,1,2,3)):
                t=36*issue;initial=float(a['S'][row,t]);original=None if not issue else a['Q'][row,t:]
                net,price,prob,groups,_=scenarios(ds,prices,cache,day,issue,mode=mode,groups=2)
                kw=dict(mode=mode,original_q=original,gamma=selection[mode]['gamma'],rho=selection[mode].get('rho',1.))
                lp=solve_dispatch(net,price,prob,initial,**kw)
                # Physical MILP checks the same ungrouped model at all four issue times.
                mi=solve_dispatch(net,price,prob,initial,method='milp',**kw)
                delta=abs(lp['objective']-mi['objective'])
                if delta>2e-5:raise AssertionError(('LP/MILP',date,issue,delta))
                result={'mode':mode,'date':date,'hour':6*issue,'lp_objective':lp['objective'],
                        'milp_objective':mi['objective'],'lp_milp_error':delta,'milp_gap':mi['mip_gap']}
                if mode=='q43' and issue<3:
                    one=solve_dispatch(net,price,prob,initial,groups=np.zeros(len(prob),int),**kw)
                    two=solve_dispatch(net,price,prob,initial,groups=groups,**kw)
                    if abs(one['objective']-lp['objective'])>2e-5 or two['objective']>lp['objective']+2e-5:
                        raise AssertionError(('containment',date,issue))
                    # Also verify the actual branched physical formulation, not just its baseline.
                    two_mi=solve_dispatch(net,price,prob,initial,groups=groups,method='milp',**kw)
                    if abs(two_mi['objective']-two['objective'])>2e-5:raise AssertionError('branched LP/MILP')
                    result.update(k1_error=abs(one['objective']-lp['objective']),k2_groups=two['branches'],
                        k2_objective=two['objective'],k2_predicted_reduction=lp['objective']-two['objective'],
                        k2_milp_error=abs(two_mi['objective']-two['objective']))
                rows.append(result);print(mode,date,6*issue,delta,flush=True)
    write_json(run/'structural_checks.json',{'cases':rows,'tolerance':2e-5,
        'lp_milp_cases':len(rows)+sum('k2_milp_error' in r for r in rows),
        'maximum_lp_milp_error':max(max(r['lp_milp_error'],r.get('k2_milp_error',0)) for r in rows)})

if __name__=='__main__':main()
