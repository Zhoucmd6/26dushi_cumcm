from pathlib import Path
from types import SimpleNamespace
from datetime import date,timedelta
import sys
import unittest
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from q4_forecasting import price_predictions,scenarios,information_groups,classify_signal
from q3_forecasting import integrate_hourly


class ForecastTests(unittest.TestCase):
    def setup_data(self):
        rng=np.random.default_rng(3)
        ds=SimpleNamespace(dates=[date(2025,1,1)+timedelta(days=i) for i in range(40)],
            load_kw=rng.uniform(2500,5000,(40,144)),pv_kw=rng.uniform(0,6000,(40,144)))
        p=rng.uniform(.3,1.,(40,144));raw=rng.uniform(0,1000,(40,4,144))
        cache={'net_q42':np.full((40,144),100.),'net_q43':np.full((40,4,144),80.),'official':raw,
               'log_price':price_predictions(p,np.ones(144)*.5,'weighted_7',.2,.1)}
        cache['log_price_uncorrected']=price_predictions(p,np.ones(144)*.5,'weighted_7')
        return ds,p,cache

    def test_current_and_future_prices_do_not_enter_prediction(self):
        _,p,_=self.setup_data();changed=p.copy();changed[30,36:]=9.;changed[31:]=10.
        a=price_predictions(p,np.ones(144)*.5,'weighted_7',.2,.1)
        b=price_predictions(changed,np.ones(144)*.5,'weighted_7',.2,.1)
        np.testing.assert_array_equal(a[30,1,36:],b[30,1,36:])

    def test_scenarios_ignore_future_observations_and_unreleased_official(self):
        ds,p,cache=self.setup_data()
        baseline=scenarios(ds,p,cache,30,1,mode='q43',groups=2)
        changed=SimpleNamespace(dates=ds.dates,load_kw=ds.load_kw.copy(),pv_kw=ds.pv_kw.copy())
        changed.load_kw[30:,36:]=1e5;changed.pv_kw[30:,36:]=0
        p2=p.copy();p2[30,36:]=8.;p2[31:]=10.
        c2={k:v.copy() for k,v in cache.items()};c2['official'][30,2:]=1e6
        c2['net_q43'][30,2:]=1e6
        result=scenarios(changed,p2,c2,30,1,mode='q43',groups=2)
        for a,b in zip(baseline[:4],result[:4]):np.testing.assert_array_equal(a,b)
        self.assertLess(max(result[4]['history_days']),30)

    def test_price_net_residuals_keep_same_day_pairing(self):
        ds,p,cache=self.setup_data()
        net,price,prob,_,meta=scenarios(ds,p,cache,30,0,window=1)
        np.testing.assert_allclose(net[0],100+(ds.load_kw[29]-ds.pv_kw[29])/6-cache['net_q42'][29])
        np.testing.assert_allclose(np.log(price[0]),cache['log_price'][30,0]+np.log(p[29])-cache['log_price'][29,0])
        self.assertEqual(meta['history_days'],[29]);self.assertEqual(prob.tolist(),[1.])

    def test_groups_require_five_distinct_days_and_mapping_is_executable(self):
        labels,meta=information_groups(np.array([[0,0],[1,1],[2,2.]]),2)
        self.assertEqual(meta['actual_k'],1)
        signal=np.r_[np.tile([-1.,0.],(10,1)),np.tile([1.,2.],(10,1))]
        labels,meta=information_groups(signal,2)
        self.assertEqual(meta['actual_k'],2)
        for ix in range(20):self.assertEqual(classify_signal(signal[ix],meta),labels[ix])

    def test_negative_net_load_is_not_clipped(self):
        ds,p,cache=self.setup_data();cache['net_q42'][30]=-10000.
        net,*_=scenarios(ds,p,cache,30,0)
        self.assertTrue((net<0).all())

    def test_official_linear_interpolation_energy(self):
        forecast=integrate_hourly(np.full(24,1200.),0.,0)
        self.assertAlmostEqual(float(forecast[:6].sum()/6),600.)
        self.assertAlmostEqual(float(forecast[6:12].sum()/6),1200.)


if __name__=='__main__':unittest.main()
