from datetime import date, timedelta
from dataclasses import replace
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from forecasting import Dataset
from forecasting_v2 import MODELS, build_cache, scenarios, predict, arima_forecast, tree_features
from microgrid_core import Battery
from stochastic_dispatch import solve_q2_fixed


def example():
    rng = np.random.default_rng(8)
    load = rng.uniform(200, 600, (35,144))
    pv = rng.uniform(0, 180, (35,144))
    return Dataset([date(2025,1,1)+timedelta(days=i) for i in range(35)], load, pv,
                   np.ones(144), np.full(144, 400.), np.full(144, 50.), {})


class ForecastTests(unittest.TestCase):
    def test_candidate_forecasts_use_history_prefix(self):
        d = example()
        changed = replace(d, load_kw=d.load_kw.copy(), pv_kw=d.pv_kw.copy())
        changed.load_kw[10:] = 1e7
        changed.pv_kw[10:] = 1e6
        for m in MODELS:
            a = build_cache(d, m, stop=11)
            b = build_cache(changed, m, stop=11)
            np.testing.assert_array_equal(a['load_kw'][:11], b['load_kw'][:11])
            np.testing.assert_array_equal(a['pv_kw'][:11], b['pv_kw'][:11])

    def test_tree_features_no_same_day_target(self):
        d = example()
        before = tree_features(d.load_kw, 12, d.dates[12])
        d.load_kw[12:] = 1e9
        np.testing.assert_array_equal(before, tree_features(d.load_kw, 12, d.dates[12]))

    def test_reject_future_history(self):
        d = example()
        with self.assertRaises(ValueError):
            predict(d.load_kw[:3], d.pv_kw[:3], d.dates[:3], d.dates[2], MODELS[0],
                    d.baseline_load_kw, d.baseline_pv_kw, 42)

    def test_arima_reproduces_exact_periodic_signal(self):
        shape = 400 + 80*np.sin(np.arange(144)*2*np.pi/144)
        forecast, info = arima_forecast(np.tile(shape, (10,1)))
        np.testing.assert_allclose(forecast, shape, atol=1e-9)

    def test_bootstrap_pairing_clipping_and_compression(self):
        d = example()
        c = build_cache(d, MODELS[0], stop=32)
        l, g, prob, audit = scenarios(d, c, 31, seed=42, draws=256)
        pool = np.arange(3,31)
        sampled = np.random.default_rng(np.random.SeedSequence([42,31])).choice(pool,256,replace=True)
        ids, counts = np.unique(sampled, return_counts=True)
        np.testing.assert_array_equal(audit['residual_counts'], counts)
        np.testing.assert_allclose(prob, counts/256)
        np.testing.assert_allclose(l, np.maximum(c['load_kw'][31]+d.load_kw[ids]-c['load_kw'][ids],0)/6)
        np.testing.assert_allclose(g, np.maximum(c['pv_kw'][31]+d.pv_kw[ids]-c['pv_kw'][ids],0)/6)
        raw_net = (np.maximum(c['load_kw'][31]+d.load_kw[sampled]-c['load_kw'][sampled],0)
                  -np.maximum(c['pv_kw'][31]+d.pv_kw[sampled]-c['pv_kw'][sampled],0))/6
        q = np.full(144,50.)
        self.assertAlmostEqual(float(prob@(np.maximum(l-g-q,0).sum(axis=1))),
                               float(np.maximum(raw_net-q,0).sum(axis=1).mean()), places=10)

    def test_bootstrap_future_actual_changes_have_no_effect(self):
        d = example()
        c = build_cache(d, MODELS[0], stop=32)
        original = scenarios(d,c,31)
        d.load_kw[31:] = 1e9
        d.pv_kw[31:] = 2e9
        changed = scenarios(d,c,31)
        for a,b in zip(original[:3],changed[:3]):
            np.testing.assert_array_equal(a,b)

    def test_cold_start_uses_only_provided_prior(self):
        d = example()
        c = build_cache(d, MODELS[0], stop=1)
        l,g,p,a = scenarios(d,c,0)
        np.testing.assert_allclose(l[0], d.baseline_load_kw/6)
        self.assertEqual(a['training_end'], None)
        self.assertEqual(a['residual_dates'], [])

    def test_compressed_and_expanded_optimization_equivalent(self):
        b = Battery(.9,.9,min_kwh=0,max_kwh=10,max_power_kw=12)
        l = np.array([[1.,4.,2.],[3.,1.,4.]])
        g = np.zeros_like(l)
        params = dict(battery=b,initial_kwh=5.,terminal_kwh=5.,terminal_cost_per_kwh=-.1)
        a = solve_q2_fixed([1.,2.,1.], l,g,[.75,.25],**params)
        ids = [0,0,0,1]
        c = solve_q2_fixed([1.,2.,1.], l[ids],g[ids],[.25]*4,**params)
        self.assertAlmostEqual(a['objective_yuan'],c['objective_yuan'],places=7)


if __name__ == '__main__':
    unittest.main()
