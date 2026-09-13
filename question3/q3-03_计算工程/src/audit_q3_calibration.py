"""独立按PDF式10—16重算主缓存，不复用分组矩/尺度拟合函数。"""
from pathlib import Path
import json,sys
import numpy as np
from run_q3 import ROOT,write_json
from forecasting import read_dataset

def audit_calibration(run):
    run=Path(run);c=dict(np.load(run/'forecast_cache.npz'));ds=read_dataset(ROOT/'data/processed')
    window=int(c['window_days']);nu=float(c['shrink_days']);width=int(c['bin_hours'])
    lead=(np.arange(144)[None,:]+1)/6-np.array([0,6,12,18])[:,None]
    clock=np.broadcast_to(np.arange(144)//36,lead.shape);bins=np.ceil(lead/width).astype(int)-1;valid=lead>0
    grouping=[(g,b,(clock==g)&(bins==b)&valid) for g in range(4) for b in range(24//width) if np.any((clock==g)&(bins==b)&valid)]
    maximum={'bias_kw':0.,'center_kw':0.,'binned_sigma_kw':0.,'pooled_sigma_kw':0.,'beta_per_hour':0.,'sigma0_kw':0.,'exponential_sigma_kw':0.}
    count=0
    for day in range(31,365):
        start=max(0,day-window);n=day-start
        rawerr=ds.pv_kw[start:day,None,:]-c['raw_g'][start:day]
        err=ds.pv_kw[start:day,None,:]-c['mu_g'][start:day]
        poolbias=rawerr[:,valid].mean(axis=1).mean()
        poolmean=err[:,valid].mean(axis=1).mean();poolvar=max(0,(err[:,valid]**2).mean(axis=1).mean()-poolmean**2)
        x=[];y=[];groups=[]
        for g,b,mask in grouping:
            bias=(n*rawerr[:,mask].mean(axis=1).mean()+nu*poolbias)/(n+nu)
            maximum['bias_kw']=max(maximum['bias_kw'],abs(bias-c['bias'][day,g,b]))
            center=np.maximum(0,c['raw_g'][day][mask]+bias)
            maximum['center_kw']=max(maximum['center_kw'],float(abs(center-c['mu_g'][day][mask]).max()))
            sample=err[:,mask];v=max(0,(sample**2).mean(axis=1).mean()-sample.mean(axis=1).mean()**2)
            sigma=np.sqrt(max(25**2,(n*v+nu*poolvar)/(n+nu)))
            maximum['binned_sigma_kw']=max(maximum['binned_sigma_kw'],float(abs(sigma-c['sigma_g_binned'][day][mask]).max()))
            maximum['pooled_sigma_kw']=max(maximum['pooled_sigma_kw'],float(abs(max(25,np.sqrt(poolvar))-c['sigma_g_pooled'][day][mask]).max()))
            x.append(float(lead[mask].mean()));y.append(np.log(sigma/c['sref'][day]));groups.append(g);count+=int(mask.sum())
        x=np.array(x);y=np.array(y);groups=np.array(groups)
        design=np.c_[np.eye(4)[groups],x]
        estimate=np.linalg.lstsq(np.sqrt(n)*design,np.sqrt(n)*y,rcond=None)[0]
        beta=max(0,estimate[-1])
        intercept=np.array([(y[groups==g]-beta*x[groups==g]).mean() for g in range(4)])
        sigma0=c['sref'][day]*np.exp(intercept)
        maximum['beta_per_hour']=max(maximum['beta_per_hour'],abs(beta-c['beta'][day]))
        maximum['sigma0_kw']=max(maximum['sigma0_kw'],float(abs(sigma0-c['sigma0'][day]).max()))
        expected=sigma0[clock]*np.exp(beta*lead)
        maximum['exponential_sigma_kw']=max(maximum['exponential_sigma_kw'],float(abs(expected[valid]-c['sigma_g_exponential'][day][valid]).max()))
    if max(maximum.values())>1e-6:raise ValueError(maximum)
    result={'checked_evaluation_days':334,'checked_release_target_points':count,'independent_formula_max_errors':maximum,
            'passed':True,'tolerance':1e-6,'variance_convention':'Equal complete-day weights, within-day population second moments.'}
    write_json(run/'independent_calibration_audit.json',result);print(json.dumps(result))
    return result

if __name__=='__main__':audit_calibration(sys.argv[1])
