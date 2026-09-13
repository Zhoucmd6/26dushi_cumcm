"""可复现第一问诊断入口，输出原始点序调度，不导出正式比赛模板。"""
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import argparse
import csv
import hashlib
import json
import platform
import traceback
import uuid

import scipy

from microgrid_core import Battery, load_baseline_csv, solve_q1

ROOT = Path(__file__).resolve().parents[1]


def write_json(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/q1_diagnostic.json")
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    if config["run_status"] != "diagnostic_pending_model_confirmation" or config["source_time_interpretation"] != "record_order_only":
        raise ValueError("This entry point currently supports diagnostic record-order runs only")
    if config["power_integration"] != "piecewise_constant_over_10_minutes":
        raise ValueError("Unsupported power integration rule")
    methods = config["methods"]
    if not methods or len(set(methods)) != len(methods) or any(m not in ("lp","milp") for m in methods):
        raise ValueError("Invalid methods configuration")
    run_id = "q1_"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")+"_"+uuid.uuid4().hex[:8]
    out = ROOT/"outputs/runs"/run_id
    out.mkdir(parents=True,exist_ok=False)
    source = ROOT/"data/processed/baseline_points.csv"
    try:
        battery = Battery(eta_charge=config["eta_charge"],eta_discharge=config["eta_discharge"])
        metadata = {"run_id":run_id,"config":config,"battery":asdict(battery),
                    "python":platform.python_version(),"scipy":scipy.__version__,
                    "input_path":str(source),"input_sha256":hashlib.sha256(source.read_bytes()).hexdigest(),
                    "code_sha256":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (Path(__file__),ROOT/"src/microgrid_core.py")}}
        write_json(out/"config.json",metadata)
        data = load_baseline_csv(source,step_hours=battery.step_hours)
        results = {}
        for method in methods:
            result = solve_q1(data["price"],data["load_kwh"],data["pv_kwh"],battery=battery,
                              initial_kwh=config["initial_kwh"],method=method,
                              time_limit_s=config["solver_time_limit_s"],mip_rel_gap=config["mip_rel_gap"])
            results[method] = result
            write_json(out/(method+".json"),result)
            with (out/(method+"_plan_by_source_point.csv")).open("w",encoding="utf-8-sig",newline="") as stream:
                writer=csv.writer(stream)
                writer.writerow(["point_index","source_time_label","Q_kwh","C_bus_kwh","D_bus_kwh","W_pv_kwh","S_start_kwh","S_end_kwh","cost_yuan"])
                a=result["plan"]
                for t in range(144):
                    writer.writerow([t+1,data["source_time_label"][t],a["Q"][t],a["C"][t],a["D"][t],a["W"][t],a["S"][t],a["S"][t+1],float(data["price"][t]*a["Q"][t])])
        summary = {"status":"completed_diagnostic","run_id":run_id,
                   "scope":"按源记录次序试算，时间区间未定稿；未填写result1.xlsx。",
                   "results":{m:{k:v for k,v in r.items() if k != "plan"} for m,r in results.items()}}
        if "lp" in results and "milp" in results:
            difference = results["milp"]["cost_yuan"]-results["lp"]["cost_yuan"]
            if difference < -1e-5:
                raise RuntimeError("MILP cost is below the LP lower bound")
            summary["milp_minus_lp_yuan"]=difference
        write_json(out/"summary.json",summary)
        (out/"run.log").write_text("Completed diagnostic. See config.json, summary.json and plans.\n",encoding="utf-8")
        print(json.dumps({"output_directory":str(out),"cost_yuan":{m:r["cost_yuan"] for m,r in results.items()}},ensure_ascii=False,indent=2))
    except Exception as exc:
        write_json(out/"failure.json",{"status":"failed","error":str(exc)})
        (out/"run.log").write_text(traceback.format_exc(),encoding="utf-8")
        raise


if __name__ == "__main__":
    main()
