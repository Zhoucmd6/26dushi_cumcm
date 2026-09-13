"""附件3按发布时间对齐、因果偏差/尺度校准及成对历史标准残差场景。"""
from datetime import datetime,timedelta
from pathlib import Path
import csv
import hashlib
import numpy as np
from forecasting import point_forecast

MODES=('pooled','exponential','binned')
HOURS=np.array([0,6,12,18])


def read_official(path,dates):
    out=np.full((len(dates),4,24),np.nan);seen=set()
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            issued=datetime.fromisoformat(row['issued_at']);lead=int(row['lead_hours'])
            valid=datetime.fromisoformat(row['valid_at']);i=(issued.date()-dates[0]).days
            if issued.minute or issued.second or issued.hour not in HOURS or not 0<=i<len(dates) or not 1<=lead<=24:
                raise ValueError('Invalid official forecast issue/lead')
            if valid!=issued+timedelta(hours=lead):raise ValueError('Forecast valid time mismatch')
            key=(i,issued.hour//6,lead-1)
            if key in seen:raise ValueError('Duplicate official forecast')
            seen.add(key);out[key]=float(row['pv_forecast_kw'])
    if len(seen)!=len(dates)*4*24 or not np.isfinite(out).all() or (out<0).any():
        raise ValueError('Incomplete/invalid official forecast grid')
    return out


def integrate_hourly(hourly,anchor,issue_hour):
    """整点值作瞬时节点，逐10分钟区间精确积分分段线性曲线。"""
    hourly=np.asarray(hourly,dtype=float)
    if hourly.shape!=(24,) or issue_hour not in HOURS or not np.isfinite(hourly).all() or np.any(hourly<0) or anchor<0:
        raise ValueError('Invalid hourly forecast input')
    n=(24-issue_hour)*6;x=np.arange(n+1)/6
    values=np.interp(x,np.arange(25),np.r_[anchor,hourly])
    return (values[:-1]+values[1:])/2


def masks(bin_hours=6):
    if bin_hours not in (3,6):raise ValueError('Lead bins must be 3 or 6 hours')
    g=np.broadcast_to(np.arange(144)//36,(4,144))
    h=(np.arange(144)[None,:]+1)/6-HOURS[:,None]
    ell=np.ceil(h/bin_hours).astype(int)-1
    valid=h>0
    groups={(a,b):(g==a)&(ell==b)&valid for a in range(4) for b in range(24//bin_hours) if np.any((g==a)&(ell==b)&valid)}
    return g,h,ell,valid,groups


def day_moments(values,groups):
    n=len(values);mean=np.full((n,4,max(b for _,b in groups)+1),np.nan);second=mean.copy()
    for (g,l),mask in groups.items():
        a=values[:,mask]
        mean[:,g,l]=a.mean(axis=1);second[:,g,l]=(a*a).mean(axis=1)
    return mean,second


def fit_scale(mean,second,start,end,floor,shrink,sref,groups,h):
    if end<=start:
        return np.full(mean.shape[1:],sref),np.full(4,sref),sref,0.,np.full(4,sref)
    all_mean=mean[start:end];all_second=second[start:end]
    pool_mean=float(np.nanmean(all_mean));pool_var=max(0,float(np.nanmean(all_second))-pool_mean**2)
    pool_sigma=max(floor,np.sqrt(pool_var));bins=np.full(mean.shape[1:],pool_sigma)
    n=end-start
    for (g,l) in groups:
        var=max(0,float(all_second[:,g,l].mean())-float(all_mean[:,g,l].mean())**2)
        bins[g,l]=np.sqrt(max(floor**2,(n*var+shrink*pool_var)/(n+shrink)))
    # 组固定截距、共享非负提前量斜率的加权最小二乘；每个分箱按独立日数加权。
    xs={g:[] for g in range(4)};ys={g:[] for g in range(4)}
    for (g,l),mask in groups.items():
        xs[g].append(float(h[mask].mean()));ys[g].append(float(np.log(bins[g,l]/sref)))
    numerator=denominator=0.
    for g in range(4):
        x=np.array(xs[g]);y=np.array(ys[g]);numerator+=n*float((x-x.mean())@(y-y.mean()));denominator+=n*float((x-x.mean())@(x-x.mean()))
    beta=max(0,numerator/denominator) if denominator>0 else 0.
    sigma0=np.array([sref*np.exp(np.mean(ys[g])-beta*np.mean(xs[g])) for g in range(4)])
    group_constant=np.array([sref*np.exp(np.mean(ys[g])) for g in range(4)])
    return bins,group_constant,pool_sigma,beta,sigma0


def build_cache(ds,official,window=28,shrink=7.,floor=25.,bin_hours=6,interpretation='linear_nodes'):
    if window<7 or shrink<=0 or floor<=0:raise ValueError('Invalid calibration settings')
    if interpretation not in ('linear_nodes','hourly_mean'):raise ValueError('Invalid forecast interpretation')
    days=len(ds.dates);g,h,ell,valid,groups=masks(bin_hours)
    raw=np.full((days,4,144),np.nan);mu_g=raw.copy();mu_l=np.zeros((days,144))
    for i in range(days):
        if i:
            mu_l[i]=point_forecast(ds.load_kw[:i],ds.pv_kw[:i],ds.dates[:i],ds.dates[i],'weighted28')[0]
        for issue,k in enumerate(HOURS):
            t=k*6;anchor=float(ds.pv_kw[i,t-1]) if t else (float(ds.pv_kw[i-1,-1]) if i else 0.)
            raw[i,issue,t:]=(integrate_hourly(official[i,issue],anchor,int(k)) if interpretation=='linear_nodes'
                              else np.repeat(official[i,issue,:24-k],6))
    raw_error=ds.pv_kw[:,None,:]-raw
    raw_mean,raw_second=day_moments(raw_error,groups)
    cal_mean=np.full_like(raw_mean,np.nan);cal_second=np.full_like(raw_mean,np.nan)
    scales_g={mode:np.full_like(raw,np.nan) for mode in MODES}
    scales_l=np.full((days,144),floor);betas=np.zeros(days);sigma0s=np.zeros((days,4));srefs=np.zeros(days)
    bias_all=np.zeros(raw_mean.shape);bins_all=np.zeros(raw_mean.shape)
    for i in range(days):
        start=max(0,i-window);n=i-start
        sref=1000.
        if i>=7:
            # 评价期开始前固定参考；不倒灌到前七天的决策。
            daymask=np.broadcast_to(ds.pv_kw[:7,None,:]>0,raw[:7].shape)&np.broadcast_to(valid,raw[:7].shape)
            sref=max(floor,float(np.sqrt(np.mean(raw_error[:7][daymask]**2))))
        srefs[i]=sref
        pool=float(np.nanmean(raw_mean[start:i])) if n else 0.
        for (group,lead),mask in groups.items():
            bias=(n*float(raw_mean[start:i,group,lead].mean())+shrink*pool)/(n+shrink) if n else 0.
            bias_all[i,group,lead]=bias
            mu_g[i][mask]=np.maximum(0,raw[i][mask]+bias)
        bins,constant,pool_sigma,beta,sigma0=fit_scale(cal_mean,cal_second,start,i,floor,shrink,sref,groups,h)
        betas[i]=beta;sigma0s[i]=sigma0;bins_all[i]=bins
        for (group,lead),mask in groups.items():
            scales_g['pooled'][i][mask]=max(floor,pool_sigma)
            scales_g['binned'][i][mask]=bins[group,lead]
            scales_g['exponential'][i][mask]=sigma0[group]*np.exp(beta*h[mask])
        # 当天校准后的误差进入档案；只在后续日期使用。
        err=ds.pv_kw[i,None,:]-mu_g[i]
        for (group,lead),mask in groups.items():
            cal_mean[i,group,lead]=err[mask].mean();cal_second[i,group,lead]=(err[mask]**2).mean()
        hist_start=max(1,start)
        if i>hist_start:
            error=ds.load_kw[hist_start:i]-mu_l[hist_start:i]
            pool_var=float(error.var())
            for group in range(4):
                sl=slice(group*36,(group+1)*36);var=float(error[:,sl].var());count=len(error)
                scales_l[i,sl]=np.sqrt(max(floor**2,(count*var+shrink*pool_var)/(count+shrink)))
        else:scales_l[i]=sref
    return {'raw_g':raw,'mu_g':mu_g,'mu_l':mu_l,'sigma_l':scales_l,
            **{'sigma_g_'+key:value for key,value in scales_g.items()},
            **{'reliability_g_'+key:reliability(value,srefs[:,None,None]) for key,value in scales_g.items()},
            'window_days':np.array(window),'bin_hours':np.array(bin_hours),'shrink_days':np.array(shrink),
            'interpretation':np.array(interpretation),
            'beta':betas,'sigma0':sigma0s,'sref':srefs,'bias':bias_all,'binned_scale':bins_all}


def reliability(sigma,sref):
    """PDF式(16)：无量纲尺度指标，不是正确概率。"""
    return 1/(1+(np.asarray(sigma)/sref)**2)


def scale_from_reliability(r,sref):
    return sref*np.sqrt(1/np.asarray(r)-1)


def make_scenarios(ds,cache,day,current_hour,source_hour,*,mode='binned',draws=64,window=None,seed=20260912,point=False):
    if current_hour not in HOURS or source_hour not in HOURS or source_hour>current_hour or mode not in MODES:
        raise ValueError('Invalid forecast issue visibility or scale mode')
    t=current_hour*6;issue=source_hour//6
    window=int(cache.get('window_days',28)) if window is None else window
    if day<0 or day>=len(ds.dates) or draws<1:raise ValueError('Invalid scenario date/count')
    ml=cache['mu_l'][day,t:];mg=cache['raw_g' if point else 'mu_g'][day,issue,t:]
    history=np.arange(max(7,day-window),day)
    if point or len(history)==0:
        load=ml[None,:];pv=mg[None,:];prob=np.ones(1);sampled=[];unique=[]
    else:
        rng=np.random.default_rng(np.random.SeedSequence([seed,day,source_hour]))
        sampled=rng.choice(history,size=draws,replace=True)
        unique,counts=np.unique(sampled,return_counts=True);prob=counts/draws
        z_l=(ds.load_kw[unique,t:]-cache['mu_l'][unique,t:])/cache['sigma_l'][unique,t:]
        z_g=(ds.pv_kw[unique,t:]-cache['mu_g'][unique,issue,t:])/cache['sigma_g_'+mode][unique,issue,t:]
        load=np.maximum(0,ml+cache['sigma_l'][day,t:]*z_l)
        pv=np.maximum(0,mg+cache['sigma_g_'+mode][day,issue,t:]*z_g)
    if not np.isfinite(load).all() or not np.isfinite(pv).all():raise ValueError('Nonfinite scenario')
    audit={'date':ds.dates[day].isoformat(),'decision_hour':int(current_hour),'official_issue_hour':int(source_hour),
           'forecast_age_hours':int(current_hour-source_hour),'first_lead_hours':float(current_hour-source_hour+1/6),
           'last_lead_hours':float(24-source_hour),'calibration_end':ds.dates[day-1].isoformat() if day else None,
           'available_independent_days':len(history),'unique_sampled_days':len(unique),'requested_draws':draws,
           'sampled_day_indices':[int(x) for x in sampled],'scale_mode':mode,'point_forecast':point,
           'window_days':int(window),'bin_hours':int(cache.get('bin_hours',6)),
           'shrink_days':float(cache.get('shrink_days',7))}
    return load/6,pv/6,prob,audit


def probability_scores(scenarios,prob,y,delta=.1):
    x=np.asarray(scenarios,dtype=float);p=np.asarray(prob,dtype=float);y=np.asarray(y,dtype=float)
    if x.ndim!=2 or p.shape!=(len(x),) or y.shape!=(x.shape[1],):raise ValueError('Invalid score dimensions')
    order=np.argsort(x,axis=0);values=np.take_along_axis(x,order,axis=0)
    weights=np.broadcast_to(p[:,None],x.shape);weights=np.take_along_axis(weights,order,axis=0)
    cumulative=weights.cumsum(axis=0)
    quant=lambda a:values[np.argmax(cumulative>=a,axis=0),np.arange(x.shape[1])]
    low,high=quant(delta/2),quant(1-delta/2)
    crps=(p[:,None]*abs(x-y)).sum(axis=0)-(weights*values*(2*cumulative-weights-1)).sum(axis=0)
    return {'crps':crps,'coverage':((y>=low)&(y<=high)).astype(float),'width':high-low,
            'interval_score':high-low+2/delta*(np.maximum(low-y,0)+np.maximum(y-high,0)),
            'mae':abs((p[:,None]*x).sum(axis=0)-y),
            'squared_error':((p[:,None]*x).sum(axis=0)-y)**2}
