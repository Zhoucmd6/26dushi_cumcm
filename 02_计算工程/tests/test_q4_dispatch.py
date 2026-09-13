from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from q4_dispatch import Battery, bill, purify, solve_dispatch


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.net=np.array([[40.,200.,30.,250.,20.,140.],[-80.,100.,170.,300.,160.,30.],
                           [250.,10.,190.,50.,350.,230.],[130.,270.,80.,210.,50.,300.]])
        self.price=np.array([[.3,.5,1.2,.8,1.1,.4],[.5,.4,1.5,.7,1.3,.6],
                            [.2,.6,1.,1.2,.9,.3],[.4,.3,1.3,.9,1.4,.5]])
        self.prob=np.array([.1,.2,.3,.4])
        self.kw=dict(initial=6000.,gamma=.5,terminal_min=6000.)

    def test_weighted_eighty_percent_quantile(self):
        b=Battery(minimum=0,maximum=1,limit=0,reference=0)
        r=solve_dispatch(np.array([[100.],[200.],[300.]]),np.array([[1.],[1.],[4.]]),
                         np.array([.6,.3,.1]),0.,battery=b,gamma=0)
        self.assertAlmostEqual(r['Q'][0],300.)

    def test_shrink_rho_zero_matches_constant_price(self):
        r=solve_dispatch(self.net,self.price,self.prob,rho=0.,**self.kw)
        means=np.broadcast_to(self.prob@self.price,self.price.shape)
        s=solve_dispatch(self.net,means,self.prob,rho=1.,**self.kw)
        self.assertAlmostEqual(r['objective'],s['objective'],places=6)

    def test_shrink_matches_independent_cartesian_mixture(self):
        rho=.25
        ns=[self.net];ps=[self.price];weights=[rho*self.prob]
        for j in range(4):
            ns.append(self.net);ps.append(np.broadcast_to(self.price[j],self.price.shape))
            weights.append((1-rho)*self.prob[j]*self.prob)
        a=solve_dispatch(self.net,self.price,self.prob,rho=rho,**self.kw)
        b=solve_dispatch(np.concatenate(ns),np.concatenate(ps),np.concatenate(weights),**self.kw)
        self.assertAlmostEqual(a['objective'],b['objective'],places=6)

    def test_path_repairing_preserves_common_objective(self):
        net=self.net.copy();p=self.price.copy();prob=np.ones(4)/4
        for t in range(6):
            order=np.roll(np.arange(4),t);net[:,t]=self.net[order,t];p[:,t]=self.price[order,t]
        a=solve_dispatch(self.net,self.price,prob,**self.kw)
        b=solve_dispatch(net,p,prob,**self.kw)
        self.assertAlmostEqual(a['objective'],b['objective'],places=6)

    def test_one_group_equals_old_midnight(self):
        a=solve_dispatch(self.net,self.price,self.prob,mode='q43',**self.kw)
        b=solve_dispatch(self.net,self.price,self.prob,mode='q43',groups=np.zeros(4),prefix=2,**self.kw)
        self.assertAlmostEqual(a['objective'],b['objective'],places=6)

    def test_one_group_equals_old_update(self):
        q=np.full(6,100.)
        a=solve_dispatch(self.net,self.price,self.prob,mode='q43',original_q=q,**self.kw)
        b=solve_dispatch(self.net,self.price,self.prob,mode='q43',original_q=q,groups=np.zeros(4),prefix=2,**self.kw)
        self.assertAlmostEqual(a['objective'],b['objective'],places=6)

    def test_lookahead_includes_common_policy_and_shared_bridge(self):
        a=solve_dispatch(self.net,self.price,self.prob,mode='q43',**self.kw)
        b=solve_dispatch(self.net,self.price,self.prob,mode='q43',groups=[0,0,1,1],prefix=2,**self.kw)
        self.assertLessEqual(b['objective'],a['objective']+1e-6)
        for node in b['nodes'][1:]:self.assertAlmostEqual(node['S'][0],b['nodes'][0]['S'][-1])
        np.testing.assert_allclose(b['nodes'][0]['R'],b['Q'][:2],atol=1e-6)

    def test_lp_purification_equals_milp(self):
        for mode,groups in [('q42',None),('q43',[0,0,1,1])]:
            a=solve_dispatch(self.net,self.price,self.prob,mode=mode,groups=groups,prefix=2,**self.kw)
            b=solve_dispatch(self.net,self.price,self.prob,mode=mode,groups=groups,prefix=2,method='milp',**self.kw)
            self.assertAlmostEqual(a['objective'],b['objective'],places=5)
            self.assertLess(a['checks']['simultaneous'],1e-7)

    def test_purification_preserves_stock_and_reduces_deficit(self):
        b=Battery();c=np.array([100.,50.]);d=np.array([40.,100.]);cp,dp=purify(c,d,b)
        np.testing.assert_allclose(b.eta_c*c-d/b.eta_d,b.eta_c*cp-dp/b.eta_d)
        self.assertTrue(np.all(cp-dp<=c-d+1e-9));self.assertEqual(np.minimum(cp,dp).max(),0)

    def test_adjustment_seventy_ninety_band(self):
        b=Battery(minimum=0,maximum=1,limit=0,reference=0)
        net=np.arange(1.,101.)[:,None];price=np.ones_like(net);prob=np.ones(100)/100
        for q,expected in [(40.,70.),(80.,80.),(98.,90.)]:
            r=solve_dispatch(net,price,prob,0.,mode='q43',original_q=[q],battery=b,gamma=0.)
            self.assertLessEqual(abs(r['nodes'][0]['R'][0]-expected),1.000001)

    def test_real_bill_uses_actual_price_and_original_commitment(self):
        a=bill(np.array([2.,3.]),np.array([150.,50.]),np.array([100.,100.]),np.array([120.,60.]),
               np.zeros(2),np.zeros(2),adjustment=True)
        self.assertAlmostEqual(a['cash_cost'],500+60-60+300)
        self.assertAlmostEqual(a['E'].sum(),30.)

    def test_invalid_price_or_mixed_q43_shrink_rejected(self):
        with self.assertRaises(ValueError):solve_dispatch(self.net,-self.price,self.prob,**self.kw)
        with self.assertRaises(ValueError):solve_dispatch(self.net,self.price,self.prob,mode='q43',rho=.5,**self.kw)


if __name__=='__main__':unittest.main()
