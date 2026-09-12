"""第二问固定日前充放电策略：整体MILP与多割Benders，共用显式输入。

场景生成不在本模块内，调用者负责提供无未来泄漏的负荷/光伏场景。
terminal_kwh和terminal_cost_per_kwh必须明确传入，不暗定日末规则。
"""
import math
import time

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix

from microgrid_core import vector, require_checks, simulate_fixed_plan


def indices(n):
    return (np.arange(n),np.arange(n,2*n),np.arange(2*n,3*n),
            np.arange(3*n,4*n+1),np.arange(4*n+1,5*n+1))


def solve_q2_fixed(price, load_scenarios_kwh, pv_scenarios_kwh, probabilities, *,
                   battery, initial_kwh, terminal_kwh, terminal_cost_per_kwh,
                   method="extensive", emergency_multiplier=5., time_limit_s=60.,
                   max_iterations=200, absolute_gap=1e-6, relative_gap=1e-8):
    """返回期望现金费、含终端项的目标值、计划、逐场景回放及界。

terminal_cost_per_kwh是Phi(s)=系数*s的带符号系数（元/内部kWh）；
储能具有正价值v时传入-v。它是规划项，不计入实际电费。
terminal_kwh=None表示不额外固定日末储量，否则施加该等式。
"""
    p=vector(price,"price")
    n=len(p)
    load=np.asarray(load_scenarios_kwh,dtype=float)
    pv=np.asarray(pv_scenarios_kwh,dtype=float)
    if load.ndim != 2 or load.shape[0] == 0 or load.shape[1] != n or pv.shape != load.shape:
        raise ValueError("Scenario inputs must both have shape (positive scenario count, period count)")
    if not np.isfinite(load).all() or not np.isfinite(pv).all() or (load<0).any() or (pv<0).any():
        raise ValueError("Scenario load/PV must be finite nonnegative energy")
    k=load.shape[0]
    prob=vector(probabilities,"probabilities",k)
    if (prob<=0).any() or abs(float(prob.sum())-1)>1e-10:
        raise ValueError("Scenario probabilities must be strictly positive and sum to 1; no silent normalization")
    if (p<=0).any() or not math.isfinite(emergency_multiplier) or emergency_multiplier<=0:
        raise ValueError("Positive prices and emergency multiplier are required")
    if not math.isfinite(terminal_cost_per_kwh):
        raise ValueError("A finite signed terminal coefficient is required")
    if method not in ("extensive","benders"):
        raise ValueError("method must be extensive or benders")
    if not math.isfinite(time_limit_s) or time_limit_s<=0 or not isinstance(max_iterations,int) or max_iterations<1:
        raise ValueError("Invalid solve limits")
    if not math.isfinite(absolute_gap) or absolute_gap<=0 or not math.isfinite(relative_gap) or not 0<=relative_gap<1:
        raise ValueError("Invalid gap tolerance")
    battery.validate_initial(initial_kwh)
    if terminal_kwh is not None:
        battery.validate_initial(terminal_kwh)
    net=load-pv
    q,c,d,s,z=indices(n)
    base=5*n+1
    started=time.perf_counter()
    deadline=started+time_limit_s

    def build_common(size):
        objective=np.zeros(size)
        objective[q]=p
        objective[s[-1]]=terminal_cost_per_kwh
        low,high=np.zeros(size),np.full(size,np.inf)
        low[s],high[s]=battery.min_kwh,battery.max_kwh
        low[s[0]]=high[s[0]]=initial_kwh
        if terminal_kwh is not None:
            low[s[-1]]=high[s[-1]]=terminal_kwh
        high[z]=1
        integ=np.zeros(size)
        integ[z]=1
        a=lil_matrix((3*n,size))
        for t in range(n):
            a[t,s[t+1]],a[t,s[t]],a[t,c[t]],a[t,d[t]]=1,-1,-battery.eta_charge,1/battery.eta_discharge
            a[n+t,c[t]],a[n+t,z[t]]=1,-battery.limit_kwh
            a[2*n+t,d[t]],a[2*n+t,z[t]]=1,battery.limit_kwh
        constraints=[LinearConstraint(a.tocsr(),np.r_[np.zeros(n),np.full(2*n,-np.inf)],
                                      np.r_[np.zeros(2*n),np.full(n,battery.limit_kwh)])]
        return objective,Bounds(low,high),integ,constraints

    def solve(objective,bounds,integ,constraints):
        remaining=deadline-time.perf_counter()
        if remaining<=0:
            raise RuntimeError("Total solve time limit reached; no certified result returned")
        r=milp(objective,integrality=integ,bounds=bounds,constraints=constraints,
               options={"time_limit":remaining,"mip_rel_gap":1e-10})
        if not r.success:
            raise RuntimeError(f"MILP status={r.status}: {r.message}")
        return r

    def assess(x):
        cash=float(p@x[q])+float(prob@(np.maximum(net-x[q]+x[c]-x[d],0)@(emergency_multiplier*p)))
        return cash+terminal_cost_per_kwh*float(x[s[-1]])

    history=[]
    cut_count=0
    if method=="extensive":
        size=base+2*k*n
        objective,bounds,integ,constraints=build_common(size)
        objective[base:base+k*n]=(prob[:,None]*emergency_multiplier*p).ravel()
        a=lil_matrix((k*n,size))
        for omega in range(k):
            for t in range(n):
                row=omega*n+t
                a[row,q[t]],a[row,c[t]],a[row,d[t]]=1,-1,1
                a[row,base+row],a[row,base+k*n+row]=1,-1
        constraints.append(LinearConstraint(a.tocsr(),net.ravel(),net.ravel()))
        r=solve(objective,bounds,integ,constraints)
        best=r.x
        lower=float(r.mip_dual_bound)
        upper=assess(best)
        require_checks({"objective_recompute_difference_yuan":abs(upper-float(r.fun))})
        history=[{"iteration":1,"lower_bound_yuan":lower,"upper_bound_yuan":upper,"cuts":0}]
    else:
        size=base+k
        objective,bounds,integ,constraints=build_common(size)
        objective[base:]=prob
        upper=np.inf
        lower=-np.inf
        best=None
        for iteration in range(1,max_iterations+1):
            r=solve(objective,bounds,integ,constraints)
            lower=max(lower,float(r.mip_dual_bound))
            value=assess(r.x)
            if value<upper:
                upper,best=value,r.x.copy()
            if upper<lower-1e-5:
                raise RuntimeError("Benders upper bound is below its lower bound")
            history.append({"iteration":iteration,"lower_bound_yuan":lower,"upper_bound_yuan":upper,"cuts":cut_count})
            if upper-lower<=absolute_gap+relative_gap*abs(upper):
                break
            deficit=net-r.x[q]+r.x[c]-r.x[d]
            recourse=np.maximum(deficit,0)@(emergency_multiplier*p)
            cut_rows,cut_rhs=[],[]
            for omega in range(k):
                if recourse[omega]-r.x[base+omega]<=absolute_gap/10:
                    continue
                lam=np.where(deficit[omega]>0,emergency_multiplier*p,0.)
                row=np.zeros(size)
                row[q],row[c],row[d]=lam,-lam,lam
                row[base+omega]=1
                cut_rows.append(row)
                cut_rhs.append(float(lam@net[omega]))
            if not cut_rows:
                raise RuntimeError("Benders stalled before the requested bound tolerance")
            constraints.append(LinearConstraint(np.asarray(cut_rows),np.asarray(cut_rhs),np.inf))
            cut_count+=len(cut_rows)
        else:
            raise RuntimeError("Benders iteration limit reached before convergence")
    if upper-lower>absolute_gap+relative_gap*abs(upper):
        raise RuntimeError("Requested total objective gap was not attained")
    # 回放模块重建状态和逐时能量平衡，不复用优化矩阵。
    replays=[simulate_fixed_plan(p,load[i],pv[i],best[q],best[c],best[d],battery=battery,
                                 initial_kwh=initial_kwh,emergency_multiplier=emergency_multiplier) for i in range(k)]
    reconstructed=np.asarray(replays[0]["S"])
    checks={"planned_vs_reconstructed_state_kwh":float(np.abs(best[s]-reconstructed).max())}
    if terminal_kwh is not None:
        checks["terminal_state_difference_kwh"]=abs(float(reconstructed[-1])-terminal_kwh)
    expected_cash=float(prob@np.array([r["cash_cost_yuan"] for r in replays]))
    terminal_cost=float(terminal_cost_per_kwh*reconstructed[-1])
    checks["objective_recompute_difference_yuan"]=abs(expected_cash+terminal_cost-upper)
    require_checks(checks)
    return {"method":method,"policy":"fixed_day_ahead_battery","status":"solved_to_requested_tolerance",
            "objective_yuan":upper,"expected_cash_cost_yuan":expected_cash,"terminal_cost_yuan":terminal_cost,
            "lower_bound_yuan":lower,"absolute_gap_yuan":max(0,upper-lower),"runtime_s":time.perf_counter()-started,
            "iterations":len(history),"cuts":cut_count,"bounds_history":history,"checks":checks,
            "plan":{name:best[idx].tolist() for name,idx in zip(("Q","C","D","S","Z"),(q,c,d,s,z))},
            "scenario_replays":replays}
