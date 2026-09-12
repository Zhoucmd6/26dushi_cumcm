"""PDF第4、5、12节：R、分组点/概率误差和场景截断影响。评价不反馈给决策。"""
from pathlib import Path
from collections import defaultdict
import csv,json
import numpy as np
from forecasting import read_dataset
from run_q3 import ROOT,write_csv,write_json
from q3_forecasting import MODES,make_scenarios,probability_scores,masks,scale_from_reliability

def diagnostics(run):
    run=Path(run);out=run/'deliverables';out.mkdir(exist_ok=True)
    ds=read_dataset(ROOT/'data/processed');cache=dict(np.load(run/'forecast_cache.npz'))
    selection=json.loads((run/'selection.json').read_text(encoding='utf-8'));selected=selection['selected']['mode']
    totals={};moment_rows=[];trace=[];inverse_error=0.
    for day in range(31,365):
        for hour in (0,6,12,18):
            t=hour*6;issue=hour//6;y=ds.pv_kw[day,t:];lead=(np.arange(t,144)+1)/6-hour
            raw=cache['raw_g'][day,issue,t:];mu=cache['mu_g'][day,issue,t:]
            for mode in MODES:
                le,ge,p,audit=make_scenarios(ds,cache,day,hour,hour,mode=mode);l,g=le*6,ge*6
                scores=probability_scores(g,p,y)
                arrays={**scores,'raw_mae':abs(raw-y),'raw_mse':(raw-y)**2,'center_mae':abs(mu-y),'center_mse':(mu-y)**2}
                for b in [0,*range(1,25-hour)]:
                    leadmask=np.ones(len(y),bool) if b==0 else np.ceil(lead)==b
                    for subset,mask in [('all',leadmask),('daylight',leadmask&(y>0)),('zero_pv',leadmask&(y==0))]:
                        if not mask.any():continue
                        key=(mode,hour,b,subset)
                        if key not in totals:totals[key]={'points':0,'independent_days':0,**{k:0. for k in arrays}}
                        acc=totals[key];acc['points']+=int(mask.sum());acc['independent_days']+=1
                        for k,v in arrays.items():acc[k]+=float(v[mask].sum())
                unique=np.unique(audit['sampled_day_indices'])
                zg=(ds.pv_kw[unique,t:]-cache['mu_g'][unique,issue,t:])/cache['sigma_g_'+mode][unique,issue,t:]
                zl=(ds.load_kw[unique,t:]-cache['mu_l'][unique,t:])/cache['sigma_l'][unique,t:]
                unclipped=mu+cache['sigma_g_'+mode][day,issue,t:]*zg
                ml=p@l;mg=p@g;vl=p@((l-ml)**2);vg=p@((g-mg)**2);cov=p@((l-ml)*(g-mg))
                corr=np.divide(cov,np.sqrt(vl*vg),out=np.full_like(vg,np.nan),where=vl*vg>1e-16)
                raw_l=cache['mu_l'][day,t:]+cache['sigma_l'][day,t:]*zl
                uml=p@raw_l;umg=p@unclipped;uvl=p@((raw_l-uml)**2);uvg=p@((unclipped-umg)**2)
                ucov=p@((raw_l-uml)*(unclipped-umg))
                ucorr=np.divide(ucov,np.sqrt(uvl*uvg),out=np.full_like(uvg,np.nan),where=uvl*uvg>1e-16)
                for subset,mask in [('all',np.ones(len(y),bool)),('daylight',y>0),('zero_pv',y==0)]:
                    if not mask.any():continue
                    finite=mask&np.isfinite(corr)&np.isfinite(ucorr)
                    moment_rows.append({'date':str(ds.dates[day]),'mode':mode,'issue_hour':hour,'subset':subset,'points':int(mask.sum()),
                        'clipped_probability_mean':float((p@(unclipped<0)) [mask].mean()),
                        'center_to_scenario_mean_abs_kw':float(abs(mg-mu)[mask].mean()),
                        'scale_to_scenario_std_abs_kw':float(abs(np.sqrt(vg)-cache['sigma_g_'+mode][day,issue,t:])[mask].mean()),
                        'clipping_mean_increase_kw':float((mg-umg)[mask].mean()),
                        'clipping_variance_change_kw2':float((vg-uvg)[mask].mean()),
                        'joint_covariance_before_kw2':float(ucov[mask].mean()),'joint_covariance_after_kw2':float(cov[mask].mean()),
                        'correlation_defined_points':int(finite.sum()),
                        'joint_correlation_before':float(ucorr[finite].mean()) if finite.any() else None,
                        'joint_correlation_after':float(corr[finite].mean()) if finite.any() else None})
            for offset,h in enumerate(lead):
                row={'date':str(ds.dates[day]),'issue_hour':hour,'valid_interval_end_hour':float((t+offset+1)/6),
                     'original_lead_hours':float(h),'raw_pv_kw':float(raw[offset]),'calibrated_center_kw':float(mu[offset]),
                     's_ref_kw':float(cache['sref'][day]),'beta_per_hour':float(cache['beta'][day]),
                     'historical_days':min(day,int(cache['window_days']))}
                for mode in MODES:
                    s=float(cache['sigma_g_'+mode][day,issue,t+offset]);r=float(cache['reliability_g_'+mode][day,issue,t+offset])
                    row['sigma_'+mode+'_kw']=s;row['R_'+mode]=r
                    inverse_error=max(inverse_error,abs(float(scale_from_reliability(r,row['s_ref_kw']))-s))
                trace.append(row)
    rows=[]
    for (mode,hour,b,subset),acc in sorted(totals.items()):
        n=acc['points'];row={'mode':mode,'issue_hour':hour,'lead_hour_bin':b,'subset':subset,
            'points':n,'independent_days':acc['independent_days']}
        for key in ('crps','coverage','width','interval_score','mae','raw_mae','center_mae'):row[key]=acc[key]/n
        row.update(rmse=np.sqrt(acc['squared_error']/n),raw_rmse=np.sqrt(acc['raw_mse']/n),center_rmse=np.sqrt(acc['center_mse']/n))
        rows.append(row)
    write_csv(out/'提前量与发布时间分组误差.csv',rows)
    write_csv(out/'非负截断与联合场景诊断.csv',moment_rows)
    write_csv(out/'预测尺度与可信度R完整记录.csv',trace)
    write_json(run/'probability_diagnostics_audit.json',{'R_to_sigma_max_error_kw':inverse_error,'reliability_rows':len(trace),
        'score_rows':len(rows),'scene_moment_rows':len(moment_rows),'evaluation_days':334,
        'zero_pv_definition':'Actual PV equals zero; this post-hoc evaluation group includes night and potentially zero-generation daytime.',
        'selected_mode':selected,'grouping':'lead_hour_bin=0 means all leads; scores in kW; repeated releases are not independent observations.',
        'correlation_definition':'Scenario-probability-weighted L/G covariance and correlation at each time, then averaged over defined time points; not causal independence.'})
    print('DIAGNOSTICS '+json.dumps({'R_inverse_error_kw':inverse_error,'rows':len(trace)}),flush=True)

if __name__=='__main__':
    import sys
    diagnostics(sys.argv[1])
