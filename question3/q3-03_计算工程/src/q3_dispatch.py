"""第三问：固定本次剩余时域计划的随机MILP，支持主/备选结算和CVaR。"""
import math
import time
import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix
from microgrid_core import dispatch_checks, require_checks


def weighted_cvar(losses, probabilities, alpha):
    loss=np.asarray(losses,dtype=float);p=np.asarray(probabilities,dtype=float)
    if loss.shape!=p.shape or loss.ndim!=1 or len(loss)==0 or not np.isfinite(loss).all():
        raise ValueError('Invalid CVaR sample')
    if not 0<alpha<1 or (p<=0).any() or abs(p.sum()-1)>1e-9:
        raise ValueError('Invalid CVaR probabilities or level')
    order=np.argsort(loss)[::-1];remaining=1-alpha;value=0.
    for index in order:
        weight=min(float(p[index]),remaining)
        value+=weight*float(loss[index]);remaining-=weight
        if remaining<=1e-14:break
    return value/(1-alpha)


def settlement(price,q,r,mode='net_refund'):
    p,q,r=[np.asarray(x,dtype=float) for x in (price,q,r)]
    if p.shape!=q.shape or p.shape!=r.shape or mode not in ('net_refund','no_refund_penalty'):
        raise ValueError('Invalid settlement inputs')
    up=np.maximum(r-q,0);down=np.maximum(q-r,0)
    sign=-1 if mode=='net_refund' else 1
    return p*q+1.5*p*up+sign*.5*p*down


def solve_roll(price,load,pv,prob,*,battery,initial_kwh,original_q=None,
               risk_lambda=0.,alpha=.90,terminal_target=6000.,terminal_gamma=0.,
               terminal_kwh=None,settlement_mode='net_refund',relax=False,time_limit=60.):
    p=np.asarray(price,dtype=float);load=np.asarray(load,dtype=float);pv=np.asarray(pv,dtype=float)
    prob=np.asarray(prob,dtype=float);n=len(p)
    if p.ndim!=1 or n<1 or load.ndim!=2 or pv.shape!=load.shape or load.shape[1]!=n:
        raise ValueError('Invalid horizon or scenario shapes')
    if not all(np.isfinite(x).all() for x in [p,load,pv,prob]) or (p<=0).any() or (load<0).any() or (pv<0).any():
        raise ValueError('Nonfinite or invalid price/scenario input')
    k=len(load)
    if prob.shape!=(k,) or (prob<=0).any() or abs(prob.sum()-1)>1e-9:
        raise ValueError('Scenario probabilities must be positive and sum to one')
    if not 0<alpha<1 or not math.isfinite(risk_lambda) or risk_lambda<0 or terminal_gamma<0 or not math.isfinite(terminal_gamma):
        raise ValueError('Invalid risk or terminal parameters')
    if settlement_mode not in ('net_refund','no_refund_penalty') or time_limit<=0:
        raise ValueError('Invalid settlement mode/solver limit')
    battery.validate_initial(initial_kwh);battery.validate_initial(terminal_target)
    if terminal_kwh is not None:battery.validate_initial(terminal_kwh)
    q=None if original_q is None else np.asarray(original_q,dtype=float)
    if q is not None and (q.shape!=(n,) or not np.isfinite(q).all() or q.min()<-1e-6):
        raise ValueError('Invalid original commitment')
    start=time.perf_counter();cursor=0
    def alloc(length):
        nonlocal cursor
        result=np.arange(cursor,cursor+length);cursor+=length;return result
    r,c,d,s,z=alloc(n),alloc(n),alloc(n),alloc(n+1),alloc(n)
    e=alloc(k*n).reshape(k,n)
    ap,am=(alloc(n),alloc(n)) if q is not None else (None,None)
    b=alloc(1)[0] if terminal_gamma>0 else None
    zeta,v=(alloc(1)[0],alloc(k)) if risk_lambda>0 else (None,None)
    objective=np.zeros(cursor);lower=np.zeros(cursor);upper=np.full(cursor,np.inf);integer=np.zeros(cursor)
    lower[s]=battery.min_kwh;upper[s]=battery.max_kwh
    lower[s[0]]=upper[s[0]]=initial_kwh
    if terminal_kwh is not None:lower[s[-1]]=upper[s[-1]]=terminal_kwh
    upper[c]=upper[d]=battery.limit_kwh;upper[z]=1
    if not relax:integer[z]=1
    constant=0.
    if q is None:objective[r]=p
    else:
        constant=float(p@q);objective[ap]=1.5*p
        objective[am]=(-.5 if settlement_mode=='net_refund' else .5)*p
    objective[e]=(prob[:,None]*5*p)
    if b is not None:objective[b]=terminal_gamma
    if zeta is not None:
        lower[zeta]=-np.inf;objective[zeta]=risk_lambda;objective[v]=risk_lambda*prob/(1-alpha)
    rows=[];cols=[];vals=[];lo=[];hi=[]
    def add(ids,coeffs,l=-np.inf,u=np.inf):
        row=len(lo);rows.extend([row]*len(ids));cols.extend(ids);vals.extend(coeffs);lo.append(l);hi.append(u)
    for t in range(n):
        add([s[t+1],s[t],c[t],d[t]],[1,-1,-battery.eta_charge,1/battery.eta_discharge],0,0)
        add([c[t],z[t]],[1,-battery.limit_kwh],u=0)
        add([d[t],z[t]],[1,battery.limit_kwh],u=battery.limit_kwh)
        if q is not None:add([r[t],ap[t],am[t]],[1,-1,1],q[t],q[t])
    # 消去u>=0：r+e-c+d>=L-G；正的期望紧急费使e取最小缺口。
    net=load-pv
    for omega in range(k):
        for t in range(n):
            add([r[t],e[omega,t],c[t],d[t]],[1,1,-1,1],l=net[omega,t])
        if zeta is not None:
            add(list(e[omega])+[zeta,v[omega]],list(5*p)+[-1,-1],u=0)
    if b is not None:add([b,s[-1]],[1,1],l=terminal_target)
    matrix=coo_matrix((vals,(rows,cols)),shape=(len(lo),cursor)).tocsr()
    constraint=LinearConstraint(matrix,np.array(lo),np.array(hi))
    result=milp(objective,integrality=integer,bounds=Bounds(lower,upper),constraints=constraint,
                options={'time_limit':time_limit,'mip_rel_gap':1e-9})
    route='milp';attempts=[{'route':route,'status':int(result.status),'message':str(result.message)}]
    if not result.success and result.status==1 and not relax:
        # 先检验LP下界能否直接给出满足充放电互斥的解。
        # 只有该解同时是原MILP可行解，才能用LP下界证明原问题最优。
        linear=milp(objective,integrality=np.zeros(cursor),bounds=Bounds(lower,upper),constraints=constraint,
                    options={'time_limit':time_limit})
        attempts.append({'route':'lp_certificate','status':int(linear.status),'message':str(linear.message)})
        overlap=np.inf if not linear.success else float(np.minimum(np.maximum(linear.x[c],0),np.maximum(linear.x[d],0)).sum())
        if linear.success and overlap<=1e-7:
            linear.x[z]=(linear.x[c]>1e-7).astype(float)
            linear['mip_dual_bound']=float(linear.fun);result=linear;route='lp_bound_with_exclusive_dispatch'
        else:
            result=milp(objective,integrality=integer,bounds=Bounds(lower,upper),constraints=constraint,
                        options={'time_limit':max(120.,time_limit),'mip_rel_gap':1e-9,'presolve':False})
            route='milp_without_presolve';attempts.append({'route':route,'status':int(result.status),'message':str(result.message),'lp_overlap_kwh':float(overlap)})
    if not result.success:raise RuntimeError(f'Q3 solve failed: {attempts}')
    x=result.x;R,C,D,S=x[r],x[c],x[d],x[s]
    deficit=net+C-D-R;E=np.maximum(deficit,0);U=np.maximum(-deficit,0)
    loss=E@(5*p)
    normal=float(p@R) if q is None else float(settlement(p,q,R,settlement_mode).sum())
    expected=float(prob@loss);tail=weighted_cvar(loss,prob,alpha)
    terminal=terminal_gamma*max(terminal_target-S[-1],0)
    total=normal+expected+risk_lambda*tail+terminal
    dual=float(getattr(result,'mip_dual_bound',result.fun))+constant
    checks=dispatch_checks(C,D,S,battery,initial_kwh)
    if relax:checks.pop('simultaneous_charge_discharge_kwh')
    checks.update(objective_yuan=abs(total-(float(result.fun)+constant)),
                  recourse_kwh=float(abs(x[e]-E).max()),
                  bus_balance_kwh=float(abs(R+E+pv+D-load-C-U).max()))
    if q is not None:checks['opposing_adjustments_kwh']=float(np.minimum(x[ap],x[am]).max())
    if terminal_kwh is not None:checks['terminal_kwh']=abs(S[-1]-terminal_kwh)
    require_checks(checks,1e-4)
    if total-dual>1e-4+1e-8*abs(total):raise RuntimeError('Requested objective bound gap not attained')
    return {'plan':{'R':R,'C':C,'D':D,'S':S,'Z':x[z]},'objective_yuan':float(total),
            'normal_cost_yuan':normal,'expected_emergency_yuan':expected,'scenario_cvar_yuan':tail,
            'terminal_penalty_yuan':float(terminal),'gap_yuan':float(max(0,total-dual)),
            'runtime_s':time.perf_counter()-start,'checks':checks,'scenario_losses_yuan':loss,
            'status':'optimal_to_tolerance','relaxed':relax,'solver_route':route,'attempts':attempts}
