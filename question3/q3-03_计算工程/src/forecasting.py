"""只使用目标日前已完成日期的预测与历史样本外残差场景。

候选模型是预先定义的历史均值/加权均值，不使用全年数据调参。
枚举最近历史联合残差，形成离散经验分布，不额外随机重抽样。
"""
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
import csv
import hashlib

import numpy as np


def clock_text(minute):
    if not isinstance(minute,int) or not 0<=minute<=1440:
        raise ValueError("Minute must be an integer from 0 to 1440")
    return f"{minute//60:02d}:{minute%60:02d}"


def interval_label(index):
    if not isinstance(index,int) or not 0<=index<144:
        raise ValueError("Interval index must be 0..143")
    return clock_text(index*10)+"-"+clock_text((index+1)*10)


@dataclass
class Dataset:
    dates: list
    load_kw: np.ndarray
    pv_kw: np.ndarray
    prices: np.ndarray
    baseline_load_kw: np.ndarray
    baseline_pv_kw: np.ndarray
    sources: dict


def read_dataset(processed):
    processed=Path(processed)
    actual_path=processed/"actual_points.csv"
    baseline_path=processed/"baseline_points.csv"
    with actual_path.open(encoding="utf-8-sig",newline="") as stream:
        rows=list(csv.DictReader(stream))
    expected=[date(2025,1,1)+timedelta(days=i) for i in range(365)]
    if len(rows)!=365*144:
        raise ValueError("Expected 365 complete days, each with 144 records")
    load,pv=np.empty((365,144)),np.empty((365,144))
    for row_id,row in enumerate(rows):
        day,t=divmod(row_id,144)
        if row["date"]!=expected[day].isoformat() or int(row["point_index"])!=t+1 or row["time_label"]!=clock_text((t+1)*10):
            raise ValueError(f"Unexpected date/time/point mapping at input row {row_id+2}")
        load[day,t],pv[day,t]=float(row["load_kw"]),float(row["pv_actual_kw"])
    with baseline_path.open(encoding="utf-8-sig",newline="") as stream:
        baseline=list(csv.DictReader(stream))
    if len(baseline)!=144:
        raise ValueError("Invalid baseline length")
    for t,row in enumerate(baseline):
        if int(row["point_index"])!=t+1 or row["time_label"]!=clock_text((t+1)*10):
            raise ValueError("Baseline time mapping differs from actual data")
    p,bl,bg=[np.array([float(row[field]) for row in baseline]) for field in ("price_yuan_per_kwh","load_kw","pv_forecast_kw")]
    for arr in (load,pv,p,bl,bg):
        if not np.isfinite(arr).all() or (arr<0).any():
            raise ValueError("Nonfinite or negative input")
    if (p<=0).any():
        raise ValueError("Strictly positive electricity prices are required")
    return Dataset(expected,load,pv,p,bl,bg,{x.name:hashlib.sha256(x.read_bytes()).hexdigest() for x in (actual_path,baseline_path)})


def point_forecast(history_load, history_pv, history_dates, target_date, model):
    """历史只能包含target_date之前的完整日。模型参数预先指定。"""
    load,pv=np.asarray(history_load,dtype=float),np.asarray(history_pv,dtype=float)
    if load.ndim!=2 or load.shape!=pv.shape or len(load)!=len(history_dates) or len(load)==0:
        raise ValueError("Invalid history shape")
    if any(d>=target_date for d in history_dates) or any(b-a!=timedelta(days=1) for a,b in zip(history_dates,history_dates[1:])):
        raise ValueError("History must contain only consecutive completed dates before the forecast origin")
    if history_dates[-1]!=target_date-timedelta(days=1):
        raise ValueError("Latest completed historical date is missing")
    if not np.isfinite(load).all() or not np.isfinite(pv).all() or (load<0).any() or (pv<0).any():
        raise ValueError("Invalid historical power")
    if model=="mean7":
        window=min(7,len(load)); weights=np.ones(window)
    elif model=="weighted28":
        window=min(28,len(load))
        ages=np.arange(window,0,-1)
        weights=np.exp(-np.log(2)*ages/7)
        weights*=np.array([2. if d.weekday()==target_date.weekday() else 1. for d in history_dates[-window:]])
    else:
        raise ValueError("Unknown forecast model")
    weights/=weights.sum()
    return weights@load[-window:],weights@pv[-window:],{
        "model":model,"training_start":history_dates[-window].isoformat(),
        "training_end":history_dates[-1].isoformat(),"origin":target_date.isoformat(),
        "history_days":window,"weights":weights.tolist()}


def causal_forecast_cache(dataset, model, minimum_history_days=7):
    """离线重建每个历史日当时能得到的预测；调用时只传入其历史前缀。"""
    if minimum_history_days<1 or minimum_history_days>=len(dataset.dates):
        raise ValueError("Invalid historical warmup")
    forecast_load=np.full_like(dataset.load_kw,np.nan,dtype=float)
    forecast_pv=np.full_like(dataset.pv_kw,np.nan,dtype=float)
    metadata={}
    for day in range(minimum_history_days,len(dataset.dates)):
        l,g,info=point_forecast(dataset.load_kw[:day],dataset.pv_kw[:day],dataset.dates[:day],dataset.dates[day],model)
        forecast_load[day],forecast_pv[day]=l,g
        metadata[day]=info
    return {"model":model,"load_kw":forecast_load,"pv_kw":forecast_pv,"metadata":metadata,
            "minimum_history_days":minimum_history_days}


def make_scenarios(dataset, cache, day, *, residual_days=28, residual_scale=1., deterministic=False):
    """整日负荷/光伏残差配对；截取day之前的样本外残差，转换为kWh。

PV支撑范围只由最近28个已观测日确定，完全未观测到发电的时段置零。
这不预知日出时间，也不能保证未来不会出现新的发电时段。
"""
    if day<cache["minimum_history_days"]+1 or day>=len(dataset.dates):
        raise ValueError("Insufficient historical out-of-sample residuals")
    if not isinstance(residual_days,int) or residual_days<1 or not np.isfinite(residual_scale) or residual_scale<0:
        raise ValueError("Invalid residual configuration")
    point_load=cache["load_kw"][day]
    point_pv=cache["pv_kw"][day]
    begin=max(cache["minimum_history_days"],day-residual_days)
    ids=np.arange(begin,day)
    if deterministic:
        raw_load=point_load[None,:]
        raw_pv=point_pv[None,:]
    else:
        # 只从已完成的历史日期减去其当时可用预测，不使用目标日误差。
        residual_load=dataset.load_kw[ids]-cache["load_kw"][ids]
        residual_pv=dataset.pv_kw[ids]-cache["pv_kw"][ids]
        raw_load=point_load+residual_scale*residual_load
        raw_pv=point_pv+residual_scale*residual_pv
    support=dataset.pv_kw[max(0,day-28):day].max(axis=0)>0
    scenario_load=np.maximum(raw_load,0)/6
    scenario_pv=np.where(support,np.maximum(raw_pv,0),0)/6
    if not np.isfinite(scenario_load).all() or not np.isfinite(scenario_pv).all():
        raise ValueError("Nonfinite forecast/scenario; inspect warmup and history")
    count=len(scenario_load)
    audit={**cache["metadata"][day],"scenario_count":count,
           "residual_start":None if deterministic else dataset.dates[begin].isoformat(),
           "residual_end":None if deterministic else dataset.dates[day-1].isoformat(),
           "scenario_construction":"point_forecast" if deterministic else "enumerated_paired_daily_out_of_sample_residuals",
           "residual_scale":residual_scale,"negative_load_values_clipped":int((raw_load<0).sum()),
           "negative_pv_values_clipped":int((raw_pv<0).sum()),
           "pv_values_masked_outside_recent_support":int(((raw_pv>0)&~support).sum())}
    return scenario_load,scenario_pv,np.full(count,1/count),audit


def no_storage_purchase(load_scenarios,pv_scenarios,probabilities,quantile=.8):
    net=load_scenarios-pv_scenarios
    if not 0<quantile<1:
        raise ValueError("Quantile must be in (0,1)")
    order=np.argsort(net,axis=0)
    sorted_net=np.take_along_axis(net,order,axis=0)
    cdf=np.cumsum(np.asarray(probabilities)[order],axis=0)
    index=np.argmax(cdf>=quantile-1e-12,axis=0)
    return np.maximum(sorted_net[index,np.arange(net.shape[1])],0)
