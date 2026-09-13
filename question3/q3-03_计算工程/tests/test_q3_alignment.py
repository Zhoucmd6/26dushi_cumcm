"""针对PDF差异的反例测试；避免仅检查程序能够运行。"""
import sys,unittest
from pathlib import Path
from dataclasses import replace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import test_q3
from q3_forecasting import build_cache,masks,make_scenarios,scale_from_reliability,probability_scores
from q3_workflow import build_experiments
from q3_dispatch import solve_roll
from microgrid_core import Battery

class AlignmentTests(unittest.TestCase):
    def test_pdf_bias_formula_retains_dawn_without_historical_support(self):
        ds,official=test_q3.Q3Tests.dataset()
        # 目标第一时段过去永远为零，但同组其它点有正偏差；式11仍允许正中心。
        ds=replace(ds,pv_kw=np.full_like(ds.pv_kw,1000.));ds.pv_kw[:,0]=0
        official[:]=0;cache=build_cache(ds,official)
        self.assertGreater(cache['mu_g'][12,0,0],0)
        self.assertAlmostEqual(cache['mu_g'][12,0,0],max(0,cache['raw_g'][12,0,0]+cache['bias'][12,0,0]))
        _,scenes,_,_=make_scenarios(ds,cache,12,0,0)
        self.assertTrue(np.isfinite(scenes).all())

    def test_three_hour_bins_cover_each_valid_point_once(self):
        for width,expected in [(3,20),(6,10)]:
            _,_,_,valid,groups=masks(width)
            self.assertEqual(len(groups),expected)
            np.testing.assert_array_equal(sum(groups.values()),valid.astype(int))
        ds,official=test_q3.Q3Tests.dataset();c=build_cache(ds,official,bin_hours=3)
        self.assertEqual(c['bias'].shape,(16,4,8))

    def test_exponential_and_reliability_inverse_including_low_scale(self):
        ds,o=test_q3.Q3Tests.dataset();c=build_cache(ds,o,bin_hours=3);g,h,_,valid,_=masks(3)
        for day in (0,7,12):
            sigma=c['sigma_g_exponential'][day]
            np.testing.assert_allclose(sigma[valid],(c['sigma0'][day][g]*np.exp(c['beta'][day]*h))[valid])
            for mode in ('pooled','binned','exponential'):
                r=c['reliability_g_'+mode][day][valid]
                self.assertTrue(((r>0)&(r<=1)).all())
                np.testing.assert_allclose(scale_from_reliability(r,c['sref'][day]),c['sigma_g_'+mode][day][valid],rtol=1e-10)
            # 同一目标组跨发布时刻，h升序对应R非增。
            for group in range(4):
                sel=valid&(g==group);order=np.argsort(h[sel]);r=c['reliability_g_exponential'][day][sel][order]
                self.assertLessEqual(float(np.diff(r).max(initial=0)),1e-12)

    def test_scenario_window_follows_selected_cache(self):
        ds,o=test_q3.Q3Tests.dataset();c=build_cache(ds,o,window=7,bin_hours=3,shrink=3)
        *_,a=make_scenarios(ds,c,15,12,6)
        self.assertEqual(a['available_independent_days'],7)
        self.assertTrue(all(8<=j<15 for j in a['sampled_day_indices']))

    def test_b4_changes_only_risk_relative_to_b3(self):
        selected={'mode':'binned','gamma':.5,'calibration':{'window':14,'shrink':3.,'bin_hours':3,'floor':25.,'interpretation':'linear_nodes'}}
        e=build_experiments({'selected':selected,'probability_selected':selected,'global_cash_selected':selected},.5)
        for name,p in e.items():
            if name.startswith('cvar'):self.assertEqual({k:v for k,v in p.items() if k not in ('risk_lambda','alpha')},e['main'])
        self.assertEqual(e['scale_pooled'],{**e['main'],'mode':'pooled'})
        self.assertTrue(e['terminal_hard']['daily_terminal'])

    def test_hard_terminal_really_changes_dispatch(self):
        kwargs=dict(battery=Battery(.9,.9),initial_kwh=6000)
        free=solve_roll([1],[[300]],[[0]],[1],**kwargs)
        hard=solve_roll([1],[[300]],[[0]],[1],terminal_kwh=6000,**kwargs)
        self.assertAlmostEqual(hard['plan']['S'][-1],6000)
        self.assertLess(free['plan']['S'][-1],6000)
        self.assertGreater(hard['objective_yuan'],free['objective_yuan'])

    def test_hourly_mean_alternative_has_correct_energy(self):
        ds,o=test_q3.Q3Tests.dataset();c=build_cache(ds,o,interpretation='hourly_mean')
        np.testing.assert_allclose(c['raw_g'][12,1,36:42],o[12,1,0])
        self.assertAlmostEqual(c['raw_g'][12,1,36:42].sum()/6,o[12,1,0])

    def test_rmse_uses_squared_errors_not_mean_absolute_error(self):
        scores=probability_scores([[0,0]],[1],[0,4])
        self.assertAlmostEqual(np.sqrt(scores['squared_error'].mean()),np.sqrt(8))
        self.assertEqual(scores['mae'].mean(),2)

    def test_midday_actuals_after_decision_and_later_release_do_not_leak(self):
        ds,o=test_q3.Q3Tests.dataset();changed=replace(ds,load_kw=ds.load_kw.copy(),pv_kw=ds.pv_kw.copy())
        day=12;changed.load_kw[day,72:]=90000;changed.pv_kw[day,72:]=80000
        later=o.copy();later[day,3]=70000
        before=build_cache(ds,o,window=14,shrink=3,bin_hours=3)
        after=build_cache(changed,later,window=14,shrink=3,bin_hours=3)
        for a,b in zip(make_scenarios(ds,before,day,12,12)[:3],make_scenarios(changed,after,day,12,12)[:3]):
            np.testing.assert_array_equal(a,b)

if __name__=='__main__':unittest.main()
