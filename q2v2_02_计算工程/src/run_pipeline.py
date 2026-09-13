"""前两问完整基线：因果预测、场景规划、连续回放及可追踪数值输出。

本程序输出用于Excel导出及论文图表的结果包，Excel由独立模板导出器生成。
"""
from dataclasses import asdict
from datetime import datetime, timezone, date
from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform
import time
import traceback
import uuid

import numpy as np
import scipy

from forecasting import read_dataset, causal_forecast_cache, make_scenarios, no_storage_purchase, interval_label
from microgrid_core import Battery, solve_q1, simulate_fixed_plan, require_checks
from stochastic_dispatch import solve_q2_fixed

ROOT=Path(__file__).resolve().parents[1]


def validate_config(config):
    expected={"time_interpretation":"source_label_is_interval_end","power_integration":"piecewise_constant_10_minutes",
              "january_battery_policy":"idle_no_self_discharge","q2_policy":"fixed_day_ahead_battery",
              "terminal_value_rule":"minus_min_daily_price_divided_by_charge_efficiency_times_stored_kwh",
              "main_forecast":"weighted28",
              "scenario_method":"enumerate_recent_paired_out_of_sample_daily_residuals_equal_probability",
              "template_time_labels":"correct_to_00_00_through_24_00_in_output_copies"}
    if any(config[k]!=v for k,v in expected.items()):
        raise ValueError("An unsupported modeling rule was requested; do not silently ignore it")
    if config["forecast_candidates"]!=["mean7","weighted28"]:
        raise ValueError("This baseline entry point has two predefined forecast candidates")
    if config["initial_february_kwh"]!=6000:
        raise ValueError("The chosen January idle policy implies February initial storage 6000")
    if config["output_start"]!="2025-02-01" or config["output_end"]!="2025-12-31":
        raise ValueError("This full-year entry point expects the contest output date range")


def write_json(path,data):
    Path(path).write_text(json.dumps(data,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def write_csv(path,rows):
    with Path(path).open("w",encoding="utf-8-sig",newline="") as stream:
        if rows:
            writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
            writer.writeheader();writer.writerows(rows)


def scenario_experiment(dataset,cache,config,battery,days,*,name,mode="stochastic",
                        terminal_multiplier=1.,residual_scale=1.,residual_days=28,folder):
    folder.mkdir(parents=True,exist_ok=False)
    state=config["initial_february_kwh"]
    minimum_price=float(dataset.prices.min())
    terminal_coefficient=-terminal_multiplier*minimum_price/battery.eta_charge
    records=[]
    audits=[]
    trajectories={key:[] for key in ("Q","C","D","S","E","U")}
    maximum_checks={}
    started=time.perf_counter()
    experiment_config={"name":name,"model":cache["model"],"mode":mode,
                       "terminal_coefficient_yuan_per_internal_kwh":terminal_coefficient,
                       "residual_days":residual_days,"residual_scale":residual_scale,
                       "start":dataset.dates[days[0]].isoformat(),"end":dataset.dates[days[-1]].isoformat(),
                       "initial_kwh":state,"year_end_kwh":config["year_end_kwh"],
                       "date_count":len(days),"main_solver":config["main_solver"]}
    write_json(folder/"config.json",experiment_config)
    log=(folder/"run.log").open("w",encoding="utf-8")
    try:
        for count,day in enumerate(days,1):
            day_date=dataset.dates[day]
            L,G,prob,audit=make_scenarios(dataset,cache,day,residual_days=residual_days,
                                         residual_scale=residual_scale,deterministic=mode=="point")
            terminal=config["year_end_kwh"] if day_date==date(2025,12,31) else None
            if mode=="no_storage":
                q=no_storage_purchase(L,G,prob)
                c,d=np.zeros(144),np.zeros(144)
                plan_state=np.full(145,state)
                solver={"runtime_s":0.,"absolute_gap_yuan":0.,"iterations":0,"checks":{},
                        "objective_yuan":float(dataset.prices@q+prob@(np.maximum(L-G-q,0)@(5*dataset.prices))+terminal_coefficient*state)}
            else:
                solver=solve_q2_fixed(dataset.prices,L,G,prob,battery=battery,initial_kwh=state,
                                     terminal_kwh=terminal,terminal_cost_per_kwh=terminal_coefficient,
                                     method=config["main_solver"],time_limit_s=config["solver_time_limit_s"],
                                     absolute_gap=config["solver_absolute_gap_yuan"],relative_gap=config["solver_relative_gap"])
                q,c,d,plan_state=[np.asarray(solver["plan"][key]) for key in ("Q","C","D","S")]
            actual=simulate_fixed_plan(dataset.prices,dataset.load_kw[day]/6,dataset.pv_kw[day]/6,q,c,d,
                                       battery=battery,initial_kwh=state,emergency_multiplier=5)
            s=np.asarray(actual["S"])
            checks={**actual["checks"],**solver["checks"],"plan_vs_actual_state_kwh":float(np.abs(s-plan_state).max())}
            if trajectories["S"]:
                checks["cross_day_difference_kwh"]=float(abs(s[0]-trajectories["S"][-1][-1]))
            if terminal is not None:
                checks["year_end_difference_kwh"]=float(abs(s[-1]-terminal))
            require_checks(checks)
            for key,value in checks.items():
                maximum_checks[key]=max(maximum_checks.get(key,0),value)
            record={"date":day_date.isoformat(),"planned_cost_yuan":actual["planned_cost_yuan"],
                    "emergency_cost_yuan":actual["emergency_cost_yuan"],"cash_cost_yuan":actual["cash_cost_yuan"],
                    "planned_purchase_kwh":float(q.sum()),"emergency_purchase_kwh":float(np.sum(actual["E"])),
                    "total_unused_kwh":float(np.sum(actual["U_total_unused"])),
                    "charge_kwh":float(c.sum()),"discharge_kwh":float(d.sum()),
                    "initial_kwh":float(s[0]),"terminal_kwh":float(s[-1]),
                    "load_mae_kw":float(np.abs(cache["load_kw"][day]-dataset.load_kw[day]).mean()),
                    "pv_mae_kw":float(np.abs(cache["pv_kw"][day]-dataset.pv_kw[day]).mean()),
                    "solver_runtime_s":solver["runtime_s"],"solver_absolute_gap_yuan":solver["absolute_gap_yuan"],
                    "scenario_count":len(prob),"planning_objective_yuan":solver["objective_yuan"]}
            records.append(record);audits.append(audit)
            for key,value in zip(trajectories,(q,c,d,s,actual["E"],actual["U_total_unused"])):
                trajectories[key].append(value)
            state=actual["terminal_kwh"]
            log.write(json.dumps({"date":record["date"],"cash_cost_yuan":record["cash_cost_yuan"],
                                  "state_end_kwh":state,"maximum_residual":max(checks.values())},ensure_ascii=False)+"\n")
            log.flush()
            if count==1 or count%30==0 or count==len(days):
                print(f"{name}: {count}/{len(days)} days; last={day_date}; elapsed={time.perf_counter()-started:.1f}s",flush=True)
        arrays={key:np.asarray(value) for key,value in trajectories.items()}
        arrays["day_indices"]=np.array(days)
        np.savez_compressed(folder/"trajectories.npz",**arrays)
        write_csv(folder/"daily_results.csv",records)
        write_json(folder/"information_audit.json",audits)
        prediction_load=cache["load_kw"][days]-dataset.load_kw[days]
        prediction_pv=cache["pv_kw"][days]-dataset.pv_kw[days]
        active=dataset.pv_kw[days]>0
        summary={"name":name,"days":len(days),"start":records[0]["date"],"end":records[-1]["date"],
                 "total_cash_cost_yuan":sum(r["cash_cost_yuan"] for r in records),
                 "total_planned_cost_yuan":sum(r["planned_cost_yuan"] for r in records),
                 "total_emergency_cost_yuan":sum(r["emergency_cost_yuan"] for r in records),
                 "total_planned_purchase_kwh":sum(r["planned_purchase_kwh"] for r in records),
                 "total_emergency_purchase_kwh":sum(r["emergency_purchase_kwh"] for r in records),
                 "total_unused_kwh":sum(r["total_unused_kwh"] for r in records),
                 "initial_kwh":records[0]["initial_kwh"],"terminal_kwh":state,
                 "forecast_load_mae_kw":float(np.abs(prediction_load).mean()),
                 "forecast_load_rmse_kw":float(np.sqrt((prediction_load**2).mean())),
                 "forecast_pv_mae_kw":float(np.abs(prediction_pv).mean()),
                 "forecast_pv_rmse_kw":float(np.sqrt((prediction_pv**2).mean())),
                 "forecast_pv_active_mae_kw":float(np.abs(prediction_pv[active]).mean()),
                 "runtime_s":time.perf_counter()-started,"checks_maxima":maximum_checks,
                 "max_solver_absolute_gap_yuan":max(r["solver_absolute_gap_yuan"] for r in records)}
        write_json(folder/"summary.json",summary)
        return summary
    except Exception as exc:
        write_json(folder/"failure.json",{"date":dataset.dates[day].isoformat(),"error":str(exc),"completed_days":len(records)})
        write_csv(folder/"partial_daily_results.csv",records)
        log.write(traceback.format_exc())
        raise
    finally:
        log.close()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/q1_q2_baseline.json")
    parser.add_argument("--pilot-days",type=int,default=0)
    parser.add_argument("--sensitivity",action="store_true")
    args=parser.parse_args()
    config=json.loads(args.config.read_text(encoding="utf-8"))
    validate_config(config)
    if args.pilot_days<0:
        raise ValueError("pilot-days must be nonnegative")
    dataset=read_dataset(ROOT/"data/processed")
    battery=Battery(config["eta_charge"],config["eta_discharge"])
    cache={m:causal_forecast_cache(dataset,m,config["minimum_history_days"]) for m in config["forecast_candidates"]}
    days=list(range(31,365))
    if args.pilot_days:
        days=days[:args.pilot_days]
    run_id=("pilot" if args.pilot_days else "full")+"_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:8]
    out=ROOT/"outputs/runs"/run_id
    out.mkdir(parents=True,exist_ok=False)
    metadata={"run_id":run_id,"config":config,"battery":asdict(battery),"python":platform.python_version(),
              "scipy":scipy.__version__,"input_sha256":dataset.sources,
              "code_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/"src").glob("*.py")}}
    write_json(out/"config.json",metadata)
    write_json(out/"time_mapping.json",[{"point_index":t+1,"source_time_label":interval_label(t).split("-")[1],
                                         "interval":interval_label(t),"minute_start":t*10,"minute_end":(t+1)*10} for t in range(144)])
    try:
        q1={m:solve_q1(dataset.prices,dataset.baseline_load_kw/6,dataset.baseline_pv_kw/6,battery=battery,
                       initial_kwh=6000,method=m) for m in ("lp","milp")}
        write_json(out/"q1.json",q1)
        np.savez_compressed(out/"forecasts.npz",**{f"{m}_{kind}":c[kind+"_kw"] for m,c in cache.items() for kind in ("load","pv")})
        specifications=[("main_weighted28","weighted28","stochastic",1.,1.),
                        ("comparison_mean7","mean7","stochastic",1.,1.),
                        ("comparison_point","weighted28","point",1.,1.),
                        ("comparison_no_storage","weighted28","no_storage",1.,1.)]
        if args.sensitivity:
            specifications.extend([("sensitivity_terminal_0_8","weighted28","stochastic",.8,1.),
                                   ("sensitivity_terminal_1_2","weighted28","stochastic",1.2,1.),
                                   ("sensitivity_residual_0_8","weighted28","stochastic",1.,.8),
                                   ("sensitivity_residual_1_2","weighted28","stochastic",1.,1.2)])
        summaries=[]
        for name,model,mode,terminal_multiplier,residual_scale in specifications:
            summaries.append(scenario_experiment(dataset,cache[model],config,battery,days,name=name,mode=mode,
                                                 terminal_multiplier=terminal_multiplier*config["terminal_value_multiplier"],
                                                 residual_scale=residual_scale*config["residual_scale"],
                                                 residual_days=config["residual_days"],folder=out/name))
            write_json(out/"experiment_summaries.json",summaries)
        write_json(out/"completion.json",{"status":"numerical_pipeline_completed","date_count":len(days),
                                          "experiments":len(summaries),"formal_workbooks_exported":False})
        print(json.dumps({"output_directory":str(out),"experiments":[{"name":s["name"],"cost":s["total_cash_cost_yuan"]} for s in summaries]},ensure_ascii=False,indent=2))
    except Exception as exc:
        write_json(out/"failure.json",{"error":str(exc),"traceback":traceback.format_exc()})
        raise


if __name__=="__main__":
    main()
