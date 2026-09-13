"""微网通用计算模块。所有调度流量为kWh，功率上限为交流母线侧kW。

函数不解释Excel时间标签、不推断初末条件、不选择预测模型。
第一问首尾相等由solve_q1显式实现；固定计划回放不重置跨日状态。
"""
from dataclasses import asdict, dataclass
from pathlib import Path
import csv
from datetime import date
import math
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import lil_matrix


@dataclass(frozen=True)
class Battery:
    eta_charge: float
    eta_discharge: float
    min_kwh: float = 1200.0
    max_kwh: float = 10800.0
    max_power_kw: float = 5000.0
    step_hours: float = 1/6

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("Battery parameters must be finite")
        if not 0 < self.eta_charge <= 1 or not 0 < self.eta_discharge <= 1:
            raise ValueError("Charge and discharge efficiencies must be in (0, 1]")
        if not 0 <= self.min_kwh < self.max_kwh:
            raise ValueError("Invalid battery capacity bounds")
        if self.max_power_kw <= 0 or self.step_hours <= 0:
            raise ValueError("Power and time step must be positive")

    @property
    def limit_kwh(self):
        return self.max_power_kw * self.step_hours

    def validate_initial(self, value):
        # 跨日独立重算可产生1e-10级边界误差，允许数值容差，不重置或截断储量。
        if not math.isfinite(value) or not self.min_kwh-1e-7 <= value <= self.max_kwh+1e-7:
            raise ValueError("Initial stored energy is outside the permitted range")


def vector(values, name, length=None, nonnegative=True):
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or len(a) == 0 or (length is not None and len(a) != length):
        raise ValueError(f"{name}: expected a nonempty vector of length {length}")
    if not np.isfinite(a).all() or (nonnegative and (a < 0).any()):
        raise ValueError(f"{name}: invalid numeric values")
    return a


def dispatch_checks(charge, discharge, state, battery, initial):
    """独立按物理公式计算残差，输出单位为kWh。"""
    c = vector(charge, "charge", nonnegative=False)
    d = vector(discharge, "discharge", len(c), nonnegative=False)
    s = vector(state, "state", len(c)+1)
    return {
        "nonnegative_dispatch_violation_kwh": float(max(0, -c.min(), -d.min())),
        "state_transition_residual_kwh": float(np.abs(np.diff(s)-battery.eta_charge*c+d/battery.eta_discharge).max()),
        "state_bound_violation_kwh": float(max(0, battery.min_kwh-s.min(), s.max()-battery.max_kwh)),
        "power_bound_violation_kwh": float(max(0, c.max()-battery.limit_kwh, d.max()-battery.limit_kwh)),
        "simultaneous_charge_discharge_kwh": float(np.minimum(np.maximum(c,0), np.maximum(d,0)).sum()),
        "initial_state_difference_kwh": float(abs(s[0]-initial)),
    }


def require_checks(checks, tolerance=1e-5):
    failed = {k:v for k,v in checks.items() if not math.isfinite(v) or v > tolerance}
    if failed:
        raise RuntimeError(f"Independent verification failed: {failed}")


def solve_q1(price, load_kwh, pv_kwh, *, battery, initial_kwh,
             method="milp", time_limit_s=60., mip_rel_gap=1e-9):
    """求解第一问，首尾储量均为initial_kwh；LP同时充放电另行标记。

失败或超时不会作为已验证最优结果返回。输入的日期/时间含义由调用方提供。
"""
    p = vector(price, "price")
    n = len(p)
    load, pv = vector(load_kwh, "load", n), vector(pv_kwh, "pv", n)
    if (p <= 0).any():
        raise ValueError("This implementation requires strictly positive prices")
    if method not in ("lp", "milp"):
        raise ValueError("method must be lp or milp")
    if not math.isfinite(time_limit_s) or time_limit_s <= 0 or not 0 <= mip_rel_gap < 1:
        raise ValueError("Invalid solver limits")
    battery.validate_initial(initial_kwh)
    integer = method == "milp"
    size = (6 if integer else 5)*n+1
    q,c,d,w = [np.arange(k*n,(k+1)*n) for k in range(4)]
    s,z = np.arange(4*n,5*n+1), np.arange(5*n+1,6*n+1)
    objective = np.zeros(size)
    objective[q] = p
    low, high = np.zeros(size), np.full(size,np.inf)
    high[c] = high[d] = battery.limit_kwh
    high[w] = pv
    low[s], high[s] = battery.min_kwh,battery.max_kwh
    low[s[0]] = high[s[0]] = initial_kwh
    low[s[-1]] = high[s[-1]] = initial_kwh
    eq = lil_matrix((2*n,size))
    rhs = np.r_[load-pv,np.zeros(n)]
    for t in range(n):
        eq[t,q[t]],eq[t,c[t]],eq[t,d[t]],eq[t,w[t]]=1,-1,1,-1
        eq[n+t,s[t+1]],eq[n+t,s[t]]=1,-1
        eq[n+t,c[t]],eq[n+t,d[t]]=-battery.eta_charge,1/battery.eta_discharge
    started = time.perf_counter()
    if integer:
        high[z] = 1
        integrality = np.zeros(size)
        integrality[z] = 1
        ub = lil_matrix((2*n,size))
        for t in range(n):
            ub[t,c[t]],ub[t,z[t]]=1,-battery.limit_kwh
            ub[n+t,d[t]],ub[n+t,z[t]]=1,battery.limit_kwh
        result = milp(objective,integrality=integrality,bounds=Bounds(low,high),
                      constraints=[LinearConstraint(eq.tocsr(),rhs,rhs),
                                   LinearConstraint(ub.tocsr(),-np.inf,np.r_[np.zeros(n),np.full(n,battery.limit_kwh)])],
                      options={"time_limit":time_limit_s,"mip_rel_gap":mip_rel_gap})
    else:
        result = linprog(objective,A_eq=eq.tocsr(),b_eq=rhs,bounds=list(zip(low,high)),
                         method="highs",options={"time_limit":time_limit_s})
    runtime = time.perf_counter()-started
    if not result.success:
        raise RuntimeError(f"{method} not solved to requested tolerance; status={result.status}: {result.message}")
    Q,C,D,W,S = (result.x[i] for i in (q,c,d,w,s))
    # 浮点求解器可能返回极小负数，保留原值并按容差检查，不修改调度来通过检查。
    negative = float(max(0,-result.x.min()))
    if negative > 1e-5:
        raise RuntimeError("Solver returned negative energy outside tolerance")
    checks = {
        "bus_balance_residual_kwh":float(np.abs(Q+pv+D-load-C-W).max()),
        "state_transition_residual_kwh":float(np.abs(np.diff(S)-battery.eta_charge*C+D/battery.eta_discharge).max()),
        "state_bound_violation_kwh":float(max(0,battery.min_kwh-S.min(),S.max()-battery.max_kwh)),
        "power_bound_violation_kwh":float(max(0,C.max()-battery.limit_kwh,D.max()-battery.limit_kwh)),
        "pv_spill_bound_violation_kwh":float(max(0,(W-pv).max(),-W.min())),
        "nonnegative_violation_kwh":negative,
        "initial_state_difference_kwh":float(abs(S[0]-initial_kwh)),
        "terminal_state_difference_kwh":float(abs(S[-1]-initial_kwh)),
        "objective_difference_yuan":float(abs(p@Q-result.fun)),
    }
    require_checks(checks)
    simultaneous = float(np.minimum(np.maximum(C,0),np.maximum(D,0)).sum())
    if integer:
        require_checks({"simultaneous_charge_discharge_kwh":simultaneous})
    return {
        "method":method,"status":int(result.status),"message":str(result.message),
        "runtime_s":runtime,"mip_gap":float(result.mip_gap) if integer else None,
        "cost_yuan":float(p@Q),"purchase_kwh":float(Q.sum()),
        "simultaneous_charge_discharge_kwh":simultaneous,
        "physically_feasible":simultaneous <= 1e-5,"checks":checks,
        "plan":{name:values.tolist() for name,values in zip(("Q","C","D","W","S"),(Q,C,D,W,S))},
    }


def simulate_fixed_plan(price, load_kwh, pv_kwh, purchase_kwh, charge_kwh, discharge_kwh,
                        *, battery, initial_kwh, emergency_multiplier):
    """第二问固定电池计划的逐时回放；实际曲线仅用于结算，不改变日前计划。

总富余可能来自光伏、计划购电或电池，不能全部称作已付费浪费。
调用方必须把返回的终值作为下一日初值；本函数没有每日重置逻辑。
"""
    p = vector(price,"price")
    n = len(p)
    load,pv = vector(load_kwh,"load",n),vector(pv_kwh,"pv",n)
    q,c,d = [vector(v,name,n,nonnegative=False) for v,name in zip(
        (purchase_kwh,charge_kwh,discharge_kwh),("Q","C","D"))]
    require_checks({"negative_plan_violation_kwh":float(max(0,-q.min(),-c.min(),-d.min()))})
    battery.validate_initial(initial_kwh)
    if (p <= 0).any() or not math.isfinite(emergency_multiplier) or emergency_multiplier <= 0:
        raise ValueError("Prices and emergency multiplier must be positive")
    s = np.r_[initial_kwh,initial_kwh+np.cumsum(battery.eta_charge*c-d/battery.eta_discharge)]
    deficit = load+c-q-pv-d
    e,u = np.maximum(deficit,0),np.maximum(-deficit,0)
    checks = dispatch_checks(c,d,s,battery,initial_kwh)
    checks["bus_balance_residual_kwh"] = float(np.abs(q+e+pv+d-load-c-u).max())
    require_checks(checks)
    planned,emergency = float(p@q),float(emergency_multiplier*p@e)
    return {"policy":"fixed_day_ahead_battery","planned_cost_yuan":planned,
            "emergency_cost_yuan":emergency,"cash_cost_yuan":planned+emergency,
            "terminal_kwh":float(s[-1]),"checks":checks,
            "S":s.tolist(),"E":e.tolist(),"U_total_unused":u.tolist()}


def load_baseline_csv(path, *, step_hours):
    """读取已核查派生CSV，严格保留原始记录次序，不推断区间起点。"""
    with Path(path).open(encoding="utf-8-sig",newline="") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 144 or [int(r["point_index"]) for r in rows] != list(range(1,145)):
        raise ValueError("Baseline must contain ordered point_index 1..144")
    if not math.isfinite(step_hours) or step_hours <= 0:
        raise ValueError("Invalid conversion time step")
    p = vector([r["price_yuan_per_kwh"] for r in rows],"price",144)
    load = vector([r["load_kw"] for r in rows],"load_kw",144)*step_hours
    pv = vector([r["pv_forecast_kw"] for r in rows],"pv_kw",144)*step_hours
    return {"point_index":[int(r["point_index"]) for r in rows],
            "source_time_label":[r["time_label"] for r in rows],
            "price":p,"load_kwh":load,"pv_kwh":pv}


def history_before_date(path, decision_date):
    """按源记录所属日期严格截取日前历史，尚不解释10分钟标签。

返回值可供后续预测模块使用；不进行补值、标准化或全年参数估计。
"""
    cutoff = date.fromisoformat(decision_date)
    rows = []
    keys = set()
    with Path(path).open(encoding="utf-8-sig",newline="") as stream:
        for row in csv.DictReader(stream):
            day = date.fromisoformat(row["date"])
            if day >= cutoff:
                continue
            index = int(row["point_index"])
            key = (day,index)
            if not 1 <= index <= 144 or key in keys:
                raise ValueError("Invalid or duplicate historical date/point key")
            keys.add(key)
            values = vector([row["load_kw"],row["pv_actual_kw"]],"historical power",2)
            rows.append({**row,"point_index":index,"load_kw":float(values[0]),"pv_actual_kw":float(values[1])})
    rows.sort(key=lambda r:(r["date"],r["point_index"]))
    return rows
