"""model-q4.tex: paired/shrunk recourse and one-update grouped lookahead.

All flow variables are AC-side kWh; stock is internal kWh. The solver only
receives current forecasts/scenarios, never future observations.
"""
from dataclasses import dataclass
import time
import numpy as np
from scipy.optimize import linprog, milp, Bounds, LinearConstraint
from scipy.sparse import coo_matrix


@dataclass(frozen=True)
class Battery:
    eta_c: float = .9
    eta_d: float = .9
    minimum: float = 1200.
    maximum: float = 10800.
    limit: float = 5000/6
    reference: float = 6000.


class LinearModel:
    def __init__(self):
        self.obj, self.lo, self.hi, self.integer = [], [], [], []
        self.eq, self.rhs_eq, self.ub, self.rhs_ub = [], [], [], []

    def variables(self, size, *, cost=0., lo=0., hi=np.inf, integer=False):
        start = len(self.obj)
        for target, value in ((self.obj,cost),(self.lo,lo),(self.hi,hi),(self.integer,int(integer))):
            target.extend(np.broadcast_to(value,(size,)).tolist())
        return np.arange(start,start+size)

    def row(self, pairs, rhs, equality=False):
        (self.eq if equality else self.ub).append(dict(pairs))
        (self.rhs_eq if equality else self.rhs_ub).append(float(rhs))

    def matrix(self, rows):
        ri, ci, data = [], [], []
        for i,row in enumerate(rows):
            for j,v in row.items():
                if v:
                    ri.append(i); ci.append(j); data.append(v)
        return coo_matrix((data,(ri,ci)),shape=(len(rows),len(self.obj))).tocsr()

    def solve(self, method):
        ae, au = self.matrix(self.eq), self.matrix(self.ub)
        if method == 'milp':
            constraints = [LinearConstraint(ae,self.rhs_eq,self.rhs_eq),
                           LinearConstraint(au,np.full(len(self.ub),-np.inf),self.rhs_ub)]
            result = milp(np.array(self.obj),integrality=np.array(self.integer),
                bounds=Bounds(self.lo,self.hi),constraints=constraints,
                options={'time_limit':120.,'mip_rel_gap':1e-9})
        else:
            result = linprog(self.obj,A_ub=au,b_ub=self.rhs_ub,A_eq=ae,b_eq=self.rhs_eq,
                bounds=list(zip(self.lo,self.hi)),method='highs',
                options={'time_limit':120.,'dual_feasibility_tolerance':1e-8,
                         'primal_feasibility_tolerance':1e-8})
        if not result.success:
            raise RuntimeError(f'{method} solver failed: {result.message}')
        return result


def purify(charge, discharge, battery):
    """The exact stock-preserving LP purification in model-q4 §14."""
    c, d = np.array(charge,copy=True), np.array(discharge,copy=True)
    epsilon = np.minimum(c,d/(battery.eta_c*battery.eta_d))
    return c-epsilon,d-battery.eta_c*battery.eta_d*epsilon


def bill(price, net, q, r, c, d, *, adjustment=False, settlement='refund'):
    e = np.maximum(net+c-d-r,0.)
    u = np.maximum(r+d-c-net,0.)
    up, down = np.maximum(r-q,0.), np.maximum(q-r,0.)
    planned = float(price@q)
    adjustment_cost = float(price@(1.5*up + (-.5 if settlement=='refund' else .5)*down)) if adjustment else 0.
    emergency = float(5*price@e)
    return dict(E=e,U=u,planned_cost=planned,adjustment_cost=adjustment_cost,
                emergency_cost=emergency,cash_cost=planned+adjustment_cost+emergency,
                adjustment_up=up,adjustment_down=down)


def physical_checks(net, r, c, d, s, initial, battery, terminal_min=None):
    e, u = np.maximum(net+c-d-r,0.),np.maximum(r+d-c-net,0.)
    return {
        'stock_transition':float(np.max(np.abs(np.diff(s)-battery.eta_c*c+d/battery.eta_d))),
        'stock_bounds':max(0.,float(battery.minimum-s.min()),float(s.max()-battery.maximum)),
        'power_bounds':max(0.,float(c.max()-battery.limit),float(d.max()-battery.limit)),
        'nonnegative':max(0.,float(-min(r.min(),c.min(),d.min()))),
        'simultaneous':float(np.minimum(c,d).max()),
        'initial_stock':abs(float(s[0])-initial),
        'balance':float(np.max(np.abs(r+e+d-c-u-net))),
        'terminal_min':max(0.,float(terminal_min-s[-1])) if terminal_min is not None else 0.
    }


def solve_dispatch(net, price, prob, initial, *, mode='q42', original_q=None,
                   groups=None, prefix=36, rho=1., gamma=.5, terminal_min=None,
                   battery=Battery(), method='lp', settlement='refund'):
    """Common prefix, one conditional tail per group; groups=None is baseline.

    original_q=None means midnight: all q are jointly optimized. Q4-2 forbids
    adjustments. For Q4-3 at midnight the executed prefix has r=q. At later
    updates original_q is fixed. Terminal conditions apply to EVERY branch.
    """
    start = time.perf_counter()
    net, price, prob = np.asarray(net,float),np.asarray(price,float),np.asarray(prob,float)
    if net.ndim!=2 or net.shape!=price.shape or not np.isfinite(net).all() or not np.isfinite(price).all():
        raise ValueError('Finite matching scenario arrays required')
    scenarios, n = net.shape
    if prob.shape!=(scenarios,) or (prob<=0).any() or abs(prob.sum()-1)>1e-10 or (price<=0).any():
        raise ValueError('Positive prices and normalized positive probabilities required')
    if mode not in ('q42','q43') or method not in ('lp','milp') or not 0<=rho<=1 or gamma<0:
        raise ValueError('Invalid model option')
    if not battery.minimum-1e-6 <= initial <= battery.maximum+1e-6:
        raise ValueError('Initial stock outside bounds')
    if mode=='q43' and rho!=1:
        raise ValueError('Q4-3 main model does not combine dependency shrinkage with lookahead')
    if settlement not in ('refund','penalty'):
        raise ValueError('Unknown adjustment settlement')
    midnight = original_q is None
    if not midnight:
        original_q = np.asarray(original_q,float)
        if original_q.shape!=(n,) or not np.isfinite(original_q).all() or (original_q < -1e-7).any():
            raise ValueError('Invalid original commitment')
    branched = groups is not None and prefix<n and mode=='q43'
    if branched:
        labels, group = np.unique(np.asarray(groups),return_inverse=True)
        if group.shape!=(scenarios,) or not 0<prefix<n:
            raise ValueError('Invalid information groups')
    else:
        labels, group, prefix = np.array([0]),np.zeros(scenarios,dtype=int),n
    weights = prob[:,None]*price
    mean_price = weights.sum(axis=0)
    recourse_weights = prob[:,None]*(rho*price+(1-rho)*mean_price)
    model = LinearModel()
    q = model.variables(n,lo=0. if midnight else original_q,hi=np.inf if midnight else original_q)
    nodes = []
    node_map = np.zeros((scenarios,n),dtype=int)
    definitions = [(0,prefix,np.arange(scenarios))]
    if branched:
        for m in range(len(labels)):
            ids = np.flatnonzero(group==m)
            definitions.append((prefix,n,ids)); node_map[ids,prefix:]=m+1
    for number,(a,b,ids) in enumerate(definitions):
        length = b-a
        wp = weights[ids,a:b].sum(axis=0)
        pm = float(prob[ids].sum())
        r = model.variables(length,cost=wp)
        c, d = model.variables(length,hi=battery.limit),model.variables(length,hi=battery.limit)
        stock = model.variables(length+1,lo=battery.minimum,hi=battery.maximum)
        if number==0:
            model.row([(stock[0],1)],initial,True)
        else:
            model.row([(stock[0],1),(nodes[0]['s'][-1],-1)],0,True)
        if b==n:
            v = model.variables(1,cost=gamma*pm)[0]
            model.row([(stock[-1],-1),(v,-1)],-battery.reference)
            if terminal_min is not None:
                model.row([(stock[-1],-1)],-terminal_min)
        fixed = mode=='q42' or (midnight and number==0)
        if fixed:
            for t in range(length):model.row([(r[t],1),(q[a+t],-1)],0,True)
        else:
            up = model.variables(length,cost=.5*wp)
            down = model.variables(length,cost=(.5 if settlement=='refund' else 1.5)*wp)
            for t in range(length):
                model.row([(r[t],1),(q[a+t],-1),(up[t],-1),(down[t],1)],0,True)
        for t in range(length):
            model.row([(stock[t+1],1),(stock[t],-1),(c[t],-battery.eta_c),(d[t],1/battery.eta_d)],0,True)
        if method=='milp':
            z=model.variables(length,hi=1,integer=True)
            for t in range(length):
                model.row([(c[t],1),(z[t],-battery.limit)],0)
                model.row([(d[t],1),(z[t],battery.limit)],battery.limit)
        nodes.append(dict(a=a,b=b,ids=ids,prob=pm,r=r,c=c,d=d,s=stock))
    emergency=model.variables(scenarios*n,cost=(5*recourse_weights).ravel()).reshape(scenarios,n)
    for w in range(scenarios):
        for t in range(n):
            node=nodes[node_map[w,t]]; j=t-node['a']
            model.row([(node['c'][j],1),(node['d'][j],-1),(node['r'][j],-1),(emergency[w,t],-1)],-net[w,t])
    solved=model.solve(method)
    x=solved.x; planned=np.maximum(x[q],0.)
    action_nodes=[]; maxima={}; purification_reduction=0.
    for node in nodes:
        c0,d0=x[node['c']],x[node['d']]
        c,d=purify(np.maximum(c0,0.),np.maximum(d0,0.),battery)
        r=np.maximum(x[node['r']],0.); stock=x[node['s']]
        check=physical_checks(net[node['ids'][0],node['a']:node['b']],r,c,d,stock,float(stock[0]),battery,
                              terminal_min if node['b']==n else None)
        check['purification_stock']=float(np.max(np.abs(battery.eta_c*c0-d0/battery.eta_d-battery.eta_c*c+d/battery.eta_d)))
        for k,v in check.items():maxima[k]=max(maxima.get(k,0.),v)
        purification_reduction+=float((c0-c).sum())
        action_nodes.append(dict(start=node['a'],stop=node['b'],ids=node['ids'].tolist(),prob=node['prob'],
                                 R=r,C=c,D=d,S=stock))
    r_all=np.empty_like(net);c_all=np.empty_like(net);d_all=np.empty_like(net)
    terminal=0.
    for node in action_nodes:
        ids=np.array(node['ids']);a,b=node['start'],node['stop']
        r_all[ids,a:b]=node['R'];c_all[ids,a:b]=node['C'];d_all[ids,a:b]=node['D']
        if b==n:terminal+=gamma*node['prob']*max(battery.reference-node['S'][-1],0.)
    e_all=np.maximum(net+c_all-d_all-r_all,0.)
    up=np.maximum(r_all-planned,0.);down=np.maximum(planned-r_all,0.)
    normal=r_all if mode=='q42' else planned+1.5*up+(-.5 if settlement=='refund' else .5)*down
    cash=float((weights*normal).sum()+5*(recourse_weights*e_all).sum())
    objective=cash+terminal
    maxima['objective_recomputed']=abs(objective-float(solved.fun))
    maxima['bridge']=max([abs(float(node['S'][0])-float(action_nodes[0]['S'][-1])) for node in action_nodes[1:]] or [0.])
    maxima['initial']=abs(float(action_nodes[0]['S'][0])-initial)
    if max(maxima.values())>2e-5:
        raise RuntimeError(f'Dispatch verification failed: {maxima}')
    return dict(Q=planned,nodes=action_nodes,groups=group,objective=objective,expected_cash=cash,
                terminal_cost=terminal,solver_objective=float(solved.fun),checks=maxima,
                runtime_s=time.perf_counter()-start,method=method,
                mip_gap=float(getattr(solved,'mip_gap',0.) or 0.),
                purified_charge_kwh=purification_reduction,scenarios=scenarios,branches=len(labels) if branched else 1)
