"""在真实144时段历史场景上比较整体求解与Benders；失败也保存。"""
from pathlib import Path
from datetime import date
import argparse
import json
import time

from forecasting import read_dataset, causal_forecast_cache, make_scenarios
from microgrid_core import Battery
from stochastic_dispatch import solve_q2_fixed

ROOT=Path(__file__).resolve().parents[1]


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("run",type=Path)
    args=parser.parse_args()
    ds=read_dataset(ROOT/"data/processed")
    cfg=json.loads((args.run/"config.json").read_text(encoding="utf-8"))["config"]
    cache=causal_forecast_cache(ds,"weighted28")
    battery=Battery(cfg["eta_charge"],cfg["eta_discharge"])
    coefficient=-float(ds.prices.min())/battery.eta_charge
    cases=[("2025-02-01",7),("2025-02-01",14),("2025-02-01",28),("2025-06-21",28)]
    results=[]
    for case_index,(day_text,k) in enumerate(cases):
        day=ds.dates.index(date.fromisoformat(day_text))
        L,G,prob,_=make_scenarios(ds,cache,day,residual_days=k)
        row={"date":day_text,"periods":144,"scenarios":len(prob),"initial_kwh":6000,
             "time_limit_per_method_s":20,"results":{}}
        # 交替方法顺序减轻固定先后启动影响；这些仍是少量单次基准，非普遍性能结论。
        methods=("extensive","benders") if case_index%2==0 else ("benders","extensive")
        for method in methods:
            started=time.perf_counter()
            try:
                r=solve_q2_fixed(ds.prices,L,G,prob,battery=battery,initial_kwh=6000,terminal_kwh=None,
                                 terminal_cost_per_kwh=coefficient,method=method,time_limit_s=20,max_iterations=200)
                row["results"][method]={key:r[key] for key in ("objective_yuan","runtime_s","iterations","cuts","absolute_gap_yuan")}
                row["results"][method]["status"]="converged"
            except RuntimeError as exc:
                message=str(exc)
                limited=any(text in message for text in ("time limit","Time limit","iteration limit","MILP status=1"))
                row["results"][method]={"status":"not_converged_within_limits" if limited else "solver_or_validation_failure",
                                         "error":message,"runtime_s":time.perf_counter()-started}
                if not limited:
                    (args.run/"solver_benchmark_failure.json").write_text(json.dumps(row,ensure_ascii=False,indent=2),encoding="utf-8")
                    raise
            print(day_text,len(prob),method,row["results"][method],flush=True)
        if all(x["status"]=="converged" for x in row["results"].values()):
            difference=abs(row["results"]["extensive"]["objective_yuan"]-row["results"]["benders"]["objective_yuan"])
            row["objective_difference_yuan"]=difference
            if difference>1e-3:
                raise AssertionError("Solver formulations disagree beyond tolerance")
        results.append(row)
        (args.run/"solver_benchmark.json").write_text(json.dumps(results,ensure_ascii=False,indent=2),encoding="utf-8")


if __name__=="__main__":
    main()
