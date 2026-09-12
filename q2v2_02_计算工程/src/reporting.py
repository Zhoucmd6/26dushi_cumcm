"""由完整10分钟轨迹生成论文表格、模板填充值与独立核对材料。"""
from pathlib import Path
import json

import numpy as np

from forecasting import interval_label, clock_text


def emergency_segments(energy,threshold=1e-6):
    e=np.asarray(energy,dtype=float)
    if not np.isfinite(threshold) or threshold<0:
        raise ValueError("Emergency reporting threshold must be finite and nonnegative")
    if e.shape!=(144,) or not np.isfinite(e).all() or (e<0).any():
        raise ValueError("Expected 144 nonnegative emergency energy values")
    mask=e>threshold
    segments=[]
    t=0
    while t<144:
        if not mask[t]:
            t+=1;continue
        start=t
        while t<144 and mask[t]:
            t+=1
        segments.append({"start_index":start,"end_index_exclusive":t,
                         "interval":clock_text(start*10)+"-"+clock_text(t*10),
                         "energy_kwh":float(e[start:t].sum())})
    return segments,float(e[~mask].sum())


def four_hour_rows(charge,discharge,state):
    c,d,s=np.asarray(charge),np.asarray(discharge),np.asarray(state)
    if c.shape!=(144,) or d.shape!=(144,) or s.shape!=(145,):
        raise ValueError("Four-hour aggregation requires 144 intervals and 145 states")
    return [[clock_text(j*240)+"-"+clock_text((j+1)*240),float(c[j*24:(j+1)*24].sum()),
             float(d[j*24:(j+1)*24].sum()),"00:00" if j==0 else "24:00" if j==1 else None,
             float(s[0]) if j==0 else float(s[-1]) if j==1 else None] for j in range(6)]


def build_payload(run_path,dataset,config):
    run=Path(run_path)
    q1=json.loads((run/"q1.json").read_text(encoding="utf-8"))["milp"]
    data=np.load(run/"main_weighted28/trajectories.npz")
    ids=data["day_indices"].tolist()
    if ids!=list(range(31,365)):
        raise ValueError("Formal template payload requires all 334 dates")
    dates=[dataset.dates[day].isoformat() for day in ids]
    plan_rows=[];battery_rows=[];emergency_rows=[];daily=[];segments_by_day={}
    excluded=0.
    for i,day in enumerate(ids):
        q,c,d,s,e,u=[data[key][i] for key in ("Q","C","D","S","E","U")]
        planned_cost=float(dataset.prices@q)
        emergency_cost=float(5*dataset.prices@e)
        total_cost=planned_cost+emergency_cost
        plan_rows.append([dates[i],*q.tolist(),float(q.sum()),total_cost])
        for j,row in enumerate(four_hour_rows(c,d,s)):
            battery_rows.append([dates[i] if j==0 else None,*row])
        segments,omitted=emergency_segments(e,config["emergency_reporting_threshold_kwh"])
        excluded+=omitted;segments_by_day[dates[i]]=segments
        for j,segment in enumerate(segments):
            emergency_rows.append([dates[i] if j==0 else None,segment["interval"],segment["energy_kwh"]])
        daily.append({"date":dates[i],"planned_purchase_kwh":float(q.sum()),"emergency_purchase_kwh":float(e.sum()),
                      "planned_plus_emergency_kwh":float(q.sum()+e.sum()),"planned_cost_yuan":planned_cost,
                      "emergency_cost_yuan":emergency_cost,"actual_total_cost_yuan":total_cost,
                      "initial_kwh":float(s[0]),"terminal_kwh":float(s[-1]),"total_unused_kwh":float(u.sum())})
    if excluded>1e-4:
        raise ValueError(f"Emergency reporting threshold suppresses material energy: {excluded} kWh")
    selected=[]
    for date_text in ("2025-03-20","2025-06-21","2025-09-23","2025-12-21"):
        i=dates.index(date_text)
        selected.append({**daily[i],"purchases_at_requested_intervals":[[interval_label(hour*6),float(data["Q"][i,hour*6])] for hour in (10,12,14,16,18,20)],
                         "battery_four_hour":four_hour_rows(data["C"][i],data["D"][i],data["S"][i]),
                         "emergency_segments":segments_by_day[date_text]})
    a=q1["plan"]
    payload={"q1":{"purchases":[[interval_label(t),a["Q"][t]] for t in range(144)],
                   "battery":four_hour_rows(a["C"],a["D"],a["S"]),
                   "cost_yuan":q1["cost_yuan"],"purchase_kwh":q1["purchase_kwh"],
                   "requested_intervals":[[interval_label(hour*6),a["Q"][hour*6]] for hour in (10,12,14,16,18,20)]},
             "q2":{"headers":["日期\\时间",*[interval_label(t) for t in range(144)],"全天购电量","全天购电费"],
                   "purchases":plan_rows,"battery":battery_rows,"emergency":emergency_rows,
                   "daily":daily,"selected_dates":selected,"emergency_omitted_below_threshold_kwh":excluded},
             "notes":{"time_mapping":"源标签为区间终点；输出模板改为00:00-00:10至23:50-24:00，点序和数值不循环平移。",
                      "q2_total_columns":"计划购电量表EP列为计划Q的日合计；EQ列为计划费用加实际紧急购电费用。紧急电量在单独工作表。",
                      "energy_units":"各电量kWh，费用元。C、D为交流母线侧电量，S为电池内部储量。",
                      "sources":"C题题面、附件1、附件2、附件5及前两问修订模型。",
                      "method":"零点固定全天电池计划；28日加权历史预测、最近历史样本外配对残差经验场景；一月闲置、年末6000。"}}
    return payload
