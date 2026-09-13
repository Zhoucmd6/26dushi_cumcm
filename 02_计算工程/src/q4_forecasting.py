"""Causal price/net-load forecasts and next-release signal scenarios."""
from pathlib import Path
import csv
import hashlib
import json
import numpy as np
from sklearn.cluster import KMeans
from forecasting import read_dataset, clock_text
from forecasting_v2 import predict
from q3_forecasting import read_official, integrate_hourly


def read_inputs(root):
    root=Path(root)
    ds=read_dataset(root)
    rows=list(csv.DictReader((root/'price_points.csv').open(encoding='utf-8-sig',newline='')))
    if len(rows)!=365*144:raise ValueError('Incomplete price grid')
    prices=np.empty((365,144))
    for ix,row in enumerate(rows):
        day,t=divmod(ix,144)
        if row['date']!=ds.dates[day].isoformat() or int(row['point_index'])!=t+1 or row['time_label']!=clock_text((t+1)*10):
            raise ValueError('Price date/time mapping mismatch')
        prices[day,t]=float(row['price_yuan_per_kwh'])
    if not np.isfinite(prices).all() or (prices<=0).any():raise ValueError('Positive finite actual prices required')
    official=read_official(root/'pv_forecasts.csv',ds.dates)
    ds.sources.update({name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in ('price_points.csv','pv_forecasts.csv')})
    return ds,prices,official


def price_predictions(prices,prior,method,alpha=0.,decay=.1):
    """The day-zero price prior is attachment 1; all other inputs are past."""
    prices=np.asarray(prices,float);days=len(prices);logp=np.log(prices)
    result=np.full((days,4,144),np.nan)
    for i in range(days):
        if not i:base=np.log(prior)
        elif method=='yesterday':base=logp[i-1].copy()
        else:
            window=int(method.split('_')[1]);begin=max(0,i-window)
            if method.startswith('mean_'):
                base=np.log(prices[begin:i].mean(axis=0))
            else:
                weights=2.**(-np.arange(i-begin-1,-1,-1)/7.);weights/=weights.sum()
                base=weights@logp[begin:i]
        delta=0.
        for issue in range(4):
            t=issue*36
            if issue:
                for j in range(t-36,t):delta=(1-alpha)*delta+alpha*(logp[i,j]-base[j])
            lead=(np.arange(t,144)+1)/6-issue*6
            result[i,issue,t:]=base[t:]+np.exp(-decay*lead)*delta
    return result


def select_price(prices,prior,start=20,end=31):
    methods=['yesterday','mean_7','mean_14','mean_28','weighted_7','weighted_14','weighted_28']
    candidates=[]
    for method in methods:
        forecast=price_predictions(prices[:end],prior,method)
        error=np.exp(forecast[start:end,0])-prices[start:end]
        candidates.append({'method':method,'mae':float(np.abs(error).mean()),'rmse':float(np.sqrt(np.mean(error**2)))})
    chosen=min(candidates,key=lambda x:(x['mae'],methods.index(x['method'])))['method']
    corrections=[]
    for alpha in (0.,.2,.5):
        for decay in ((0.,) if alpha==0 else (0.,.1,.5)):
            forecast=price_predictions(prices[:end],prior,chosen,alpha,decay)
            errors=np.concatenate([(np.exp(forecast[start:end,k,k*36:])-prices[start:end,k*36:]).ravel() for k in (1,2,3)])
            corrections.append({'alpha':alpha,'decay_per_hour':decay,'mae':float(np.abs(errors).mean()),
                                'rmse':float(np.sqrt(np.mean(errors**2)))})
    best=min(corrections,key=lambda x:(x['mae'],x['alpha'],x['decay_per_hour']))
    return {'method':chosen,**best,'price_candidates':candidates,'correction_candidates':corrections,
            'validation_start':'2025-01-21','validation_end':'2025-01-31','criterion':'historical MAE',
            'formal_frozen_at':'2025-02-01T00:00'}


def supply_predictions(ds,official,stop=365,progress=None):
    load=np.empty((stop,144));pv=np.empty_like(load)
    p1_load=np.empty_like(load);p1_pv=np.empty_like(load)
    raw=np.full((stop,4,144),np.nan)
    for i in range(stop):
        for model,l,g in [('P3_random_forest',load,pv),('P1_weighted28',p1_load,p1_pv)]:
            l[i],g[i],_=predict(ds.load_kw[:i],ds.pv_kw[:i],ds.dates[:i],ds.dates[i],model,
                               ds.baseline_load_kw,ds.baseline_pv_kw,20260912)
        for issue in range(4):
            t=issue*36
            anchor=ds.pv_kw[i,t-1] if t else (ds.pv_kw[i-1,-1] if i else 0.)
            raw[i,issue,t:]=integrate_hourly(official[i,issue],float(anchor),issue*6)/6
        if progress and (i==0 or (i+1)%60==0):progress(f'supply forecasts {i+1}/{stop}')
    return {'load':load/6,'pv':pv/6,'p1_load':p1_load/6,'p1_pv':p1_pv/6,'official':raw}


def make_cache(ds,prices,supply,selection,*,warmup=False):
    # During actual January warmup the method is preregistered, not selected on January's future.
    days=len(supply['load'])
    method='weighted_7' if warmup else selection['method']
    alpha=0. if warmup else selection['alpha'];decay=0. if warmup else selection['decay_per_hour']
    y=price_predictions(prices[:days],ds.prices,method,alpha,decay)
    y0=price_predictions(prices[:days],ds.prices,method,0.,0.)
    load=supply['p1_load' if warmup else 'load']
    pv=supply['p1_pv' if warmup else 'pv']
    return {**supply,'active_load':load,'net_q42':load-pv,'net_q43':load[:,None,:]-supply['official'],
            'log_price':y,'log_price_uncorrected':y0}


def information_groups(signals,k,minimum_days=5,seed=20260913):
    signals=np.asarray(signals,float)
    sigma=signals.std(axis=0)
    active=sigma>1e-10
    metadata={'requested_k':int(k),'minimum_distinct_days':int(minimum_days),
              'scale':sigma.tolist(),'active_dimensions':active.tolist(),'centers':[]}
    if k==1 or len(signals)<k*minimum_days or not active.any():
        metadata.update(actual_k=1,reason='requested one group or insufficient signal samples')
        return np.zeros(len(signals),int),metadata
    z=signals[:,active]/sigma[active]
    model=KMeans(n_clusters=k,random_state=seed,n_init=10).fit(z)
    count=np.bincount(model.labels_,minlength=k)
    if count.min()<minimum_days:
        metadata.update(actual_k=1,reason='group has fewer than five distinct historical days')
        return np.zeros(len(signals),int),metadata
    metadata.update(actual_k=int(k),centers=model.cluster_centers_.tolist(),counts=count.tolist(),reason='historical paired signals')
    return model.labels_,metadata


def classify_signal(observed,metadata):
    if metadata['actual_k']==1:return 0
    active=np.array(metadata['active_dimensions'],bool)
    z=np.asarray(observed)[active]/np.asarray(metadata['scale'])[active]
    return int(np.argmin(((np.asarray(metadata['centers'])-z)**2).sum(axis=1)))


def scenarios(ds,prices,cache,day,issue,*,mode='q42',window=28,point=False,price_correction=True,
              groups=1,price_mean_only=False):
    """Only whole historical days enter residuals and future-release scenarios."""
    if mode=='q42' and issue!=0:raise ValueError('Q4-2 only plans at midnight')
    t=issue*36
    predicted_net=cache['net_q42'][day] if mode=='q42' else cache['net_q43'][day,issue]
    log_forecast=cache['log_price' if price_correction else 'log_price_uncorrected']
    historical=np.arange(max(7,day-window),day)
    if point or not len(historical):
        net=predicted_net[None,t:].copy();p=np.exp(log_forecast[day,issue,t:])[None,:]
        ids=np.array([],int);prob=np.ones(1)
    else:
        ids=historical
        old_prediction=cache['net_q42'][ids,t:] if mode=='q42' else cache['net_q43'][ids,issue,t:]
        actual_net=(ds.load_kw[ids,t:]-ds.pv_kw[ids,t:])/6
        net=predicted_net[None,t:]+actual_net-old_prediction
        price_error=np.log(prices[ids,t:])-log_forecast[ids,issue,t:]
        p=np.exp(log_forecast[day,issue,t:]+price_error)
        prob=np.full(len(ids),1/len(ids))
    if price_mean_only:p=np.broadcast_to(prob@p,p.shape).copy()
    if not np.isfinite(net).all() or not np.isfinite(p).all() or (p<=0).any():raise ValueError('Invalid generated scenario')
    grouping=None
    meta={'day':ds.dates[day].isoformat(),'hour':int(issue*6),'history_days':ids.tolist(),
          'latest_history':ds.dates[day-1].isoformat() if day else None,'point':bool(point),
          'price_prefix_end_index':t,'official_visible_issue':int(issue*6) if mode=='q43' else None,
          'scenario_count':len(prob),'negative_net_values':int((net<0).sum())}
    if mode=='q43' and issue<3:
        b=t+36
        if len(ids):
            # Simulated next release is current official forecast + a historical paired revision.
            now=cache['official'][day,issue,b:b+36]
            revision=cache['official'][ids,issue+1,b:b+36]-cache['official'][ids,issue,b:b+36]
            hypothetical=np.maximum(0,now+revision)
            zg=(hypothetical-now).sum(axis=1)
            zp=(np.log(p[:,30:36])-log_forecast[day,issue,b-6:b]).mean(axis=1)
            signal=np.c_[zg,zp]
            grouping,group_meta=information_groups(signal,groups)
            group_meta['next_forecast_negative_clipped']=int((now+revision<0).sum())
            group_meta['signal_scenarios']=signal.tolist()
        else:
            grouping=np.zeros(1,int);group_meta={'requested_k':groups,'actual_k':1,'reason':'deterministic cold start'}
        meta['grouping']=group_meta
    return net,p,prob,grouping,meta


def realized_signal(ds,prices,cache,day,issue,price_correction=True):
    """Called at the NEXT update only, after its observations are released."""
    b=(issue+1)*36
    if issue>=3:raise ValueError('No within-day next update')
    zg=float((cache['official'][day,issue+1,b:b+36]-cache['official'][day,issue,b:b+36]).sum())
    y=cache['log_price' if price_correction else 'log_price_uncorrected']
    zp=float((np.log(prices[day,b-6:b])-y[day,issue,b-6:b]).mean())
    return np.array([zg,zp])
