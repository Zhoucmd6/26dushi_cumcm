from pathlib import Path
from datetime import date,timedelta
from dataclasses import replace
import sys
import unittest
from unittest.mock import patch
from scipy.optimize import OptimizeResult
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from microgrid_core import Battery
from forecasting import Dataset
from q3_dispatch import solve_roll,settlement,weighted_cvar
from q3_forecasting import integrate_hourly,build_cache,make_scenarios,probability_scores


class Q3Tests(unittest.TestCase):
    def test_timeout_requires_feasible_lp_bound_certificate(self):
        import q3_dispatch
        original=q3_dispatch.milp;calls=[]
        def timeout_once(*args,**kwargs):
            calls.append(1)
            if len(calls)==1:return OptimizeResult(success=False,status=1,message='Controlled test timeout')
            return original(*args,**kwargs)
        with patch('q3_dispatch.milp',side_effect=timeout_once):
            out=solve_roll([.4,1],[ [100.,300.],[200.,400.] ],[[0.,0.],[0.,0.]],[.5,.5],
                           battery=Battery(.9,.9),initial_kwh=6000,terminal_kwh=6000)
        self.assertEqual(out['solver_route'],'lp_bound_with_exclusive_dispatch')
        self.assertLess(out['gap_yuan'],1e-5)
        self.assertLess(max(out['checks'].values()),1e-5)

    def test_settlement_examples_and_alternative(self):
        np.testing.assert_allclose(settlement([2,2],[10,10],[8,12]),[18,26])
        np.testing.assert_allclose(settlement([2,2],[10,10],[8,12],'no_refund_penalty'),[22,26])

    def test_only_final_version_is_paid(self):
        final=float(settlement([1],[10],[8])[0])
        self.assertEqual(final,9)
        self.assertNotEqual(final,float(settlement([1],[10],[12])[0]+settlement([1],[12],[8])[0]))

    def test_discrete_cvar_fractional_quantile_mass(self):
        self.assertAlmostEqual(weighted_cvar([0,10,100],[.8,.15,.05],.9),55)
        self.assertAlmostEqual(weighted_cvar([0,10,100],[.8,.15,.05],.95),100)

    def test_hourly_linear_integral(self):
        power=integrate_hourly(np.arange(1,25)*60,0,0)
        np.testing.assert_allclose(power[:6],[5,15,25,35,45,55])
        self.assertAlmostEqual(power[:6].sum()/6,30)
        self.assertEqual(len(integrate_hourly(np.ones(24)*600,600,18)),36)

    def test_cvar_milp_matches_direct_tail(self):
        battery=Battery(.9,.9)
        load=np.array([[4,7,2],[9,3,8],[15,8,10]],float)*100
        pv=np.array([[0,6,0],[0,8,0],[0,3,0]],float)*100
        for risk in [0,.2]:
            out=solve_roll([.4,1,.8],load,pv,[.5,.3,.2],battery=battery,initial_kwh=6000,
                           terminal_kwh=6000,terminal_gamma=.4,risk_lambda=risk)
            self.assertAlmostEqual(out['scenario_cvar_yuan'],weighted_cvar(out['scenario_losses_yuan'],[.5,.3,.2],.9))
            self.assertLess(max(out['checks'].values()),1e-4)

    def test_single_period_adjustment_quantiles(self):
        battery=Battery(.9,.9);load=np.arange(10,dtype=float)[:,None];pv=np.zeros_like(load)
        for q,low,high in [(1,6,7),(7.5,7.5,7.5),(12,8,9)]:
            out=solve_roll([1],load,pv,np.full(10,.1),battery=battery,initial_kwh=6000,
                           original_q=[q],terminal_kwh=6000)
            self.assertGreaterEqual(out['plan']['R'][0],low-1e-6)
            self.assertLessEqual(out['plan']['R'][0],high+1e-6)

    def test_weighted_crps_matches_pairwise_definition(self):
        x=np.array([[0,1],[5,3],[10,7]],float);p=np.array([.2,.5,.3]);y=np.array([6,5])
        expected=(p[:,None]*abs(x-y)).sum(axis=0)-.5*np.sum(p[:,None,None]*p[None,:,None]*abs(x[:,None,:]-x[None,:,:]),axis=(0,1))
        np.testing.assert_allclose(probability_scores(x,p,y)['crps'],expected)

    @staticmethod
    def dataset():
        days=16;t=np.arange(144)/6;pv=np.maximum(np.sin((t-6)/12*np.pi),0)*4000
        load=np.array([3000+i*30+200*np.cos(t) for i in range(days)])
        pvs=np.array([pv*(1+i*.01) for i in range(days)])
        ds=Dataset([date(2025,1,1)+timedelta(days=i) for i in range(days)],load,pvs,np.ones(144),load[0],pvs[0],{})
        official=np.empty((days,4,24))
        for i in range(days):
            for k in range(4):official[i,k]=np.maximum(np.sin(((np.arange(1,25)+k*6)%24-6)/12*np.pi),0)*3800
        return ds,official

    def test_future_actuals_and_unreleased_forecasts_do_not_leak(self):
        ds,official=self.dataset();day=12
        changed=replace(ds,load_kw=ds.load_kw.copy(),pv_kw=ds.pv_kw.copy())
        changed.load_kw[day:]=99999;changed.pv_kw[day:]=77777
        later=official.copy();later[day,1:]=88888;later[day+1:]=55555
        before=build_cache(ds,official);after=build_cache(changed,later)
        a=make_scenarios(ds,before,day,0,0)
        b=make_scenarios(changed,after,day,0,0)
        for left,right in zip(a[:3],b[:3]):np.testing.assert_array_equal(left,right)

    def test_old_forecast_retains_original_lead(self):
        ds,official=self.dataset();cache=build_cache(ds,official)
        _,_,_,audit=make_scenarios(ds,cache,12,12,0)
        self.assertAlmostEqual(audit['first_lead_hours'],12+1/6)
        self.assertEqual(audit['official_issue_hour'],0)
        self.assertTrue(all(i<12 for i in audit['sampled_day_indices']))
        with self.assertRaises(ValueError):make_scenarios(ds,cache,12,6,12)

    def test_shared_days_preserve_joint_standard_residuals(self):
        ds,official=self.dataset();cache=build_cache(ds,official)
        l,g,p,audit=make_scenarios(ds,cache,12,6,6,draws=64)
        self.assertLessEqual(len(l),audit['available_independent_days'])
        self.assertEqual(len(audit['sampled_day_indices']),64)
        self.assertAlmostEqual(p.sum(),1)
        unique=np.unique(audit['sampled_day_indices']);t=36
        zl=(ds.load_kw[unique,t:]-cache['mu_l'][unique,t:])/cache['sigma_l'][unique,t:]
        np.testing.assert_allclose(l*6,np.maximum(0,cache['mu_l'][12,t:]+cache['sigma_l'][12,t:]*zl))


if __name__=='__main__':unittest.main()
