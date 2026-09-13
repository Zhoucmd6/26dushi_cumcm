from pathlib import Path
import sys
import unittest
import json
from datetime import date, timedelta
from dataclasses import replace

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from forecasting import Dataset, causal_forecast_cache, make_scenarios, point_forecast, interval_label
from reporting import emergency_segments, four_hour_rows
from microgrid_core import Battery
from run_pipeline import validate_config


class PipelineTests(unittest.TestCase):
    def dataset(self):
        n=40;dates=[date(2025,1,1)+timedelta(days=i) for i in range(n)]
        load=np.array([np.full(144,3000+20*i) for i in range(n)])
        pv=np.zeros((n,144));pv[:,40:105]=1000+np.arange(n)[:,None]*10
        return Dataset(dates,load,pv,np.ones(144),load[0],pv[0],{})

    def test_future_actual_values_cannot_change_current_plan_inputs(self):
        original=self.dataset();day=31
        future=replace(original,load_kw=original.load_kw.copy(),pv_kw=original.pv_kw.copy())
        future.load_kw[day:]=999999
        future.pv_kw[day:]=777777
        for model in ("mean7","weighted28"):
            before=causal_forecast_cache(original,model)
            after=causal_forecast_cache(future,model)
            inputs_before=make_scenarios(original,before,day)
            inputs_after=make_scenarios(future,after,day)
            for a,b in zip(inputs_before[:3],inputs_after[:3]):
                np.testing.assert_array_equal(a,b)
            self.assertLess(inputs_before[3]["residual_end"],original.dates[day].isoformat())

    def test_residuals_are_out_of_sample_and_paired(self):
        ds=self.dataset();cache=causal_forecast_cache(ds,"mean7")
        self.assertTrue(np.issubdtype(cache["load_kw"].dtype,np.floating))
        L,G,prob,audit=make_scenarios(ds,cache,31,residual_days=1)
        np.testing.assert_allclose(L[0],(cache["load_kw"][31]+ds.load_kw[30]-cache["load_kw"][30])/6)
        np.testing.assert_allclose(G[0],np.maximum(cache["pv_kw"][31]+ds.pv_kw[30]-cache["pv_kw"][30],0)/6)
        self.assertEqual(prob.tolist(),[1.])

    def test_forecast_rejects_current_day_in_history(self):
        ds=self.dataset()
        with self.assertRaises(ValueError):
            point_forecast(ds.load_kw[:32],ds.pv_kw[:32],ds.dates[:32],ds.dates[31],"mean7")

    def test_first_last_and_requested_time_mapping(self):
        self.assertEqual(interval_label(0),"00:00-00:10")
        self.assertEqual(interval_label(143),"23:50-24:00")
        self.assertEqual(interval_label(60),"10:00-10:10")

    def test_emergency_segments_preserve_sum_and_last_day_boundary(self):
        e=np.zeros(144);e[0:2]=[1,2];e[3]=4;e[142:]=[5,6]
        segments,excluded=emergency_segments(e)
        self.assertEqual([s["interval"] for s in segments],["00:00-00:20","00:30-00:40","23:40-24:00"])
        self.assertEqual(sum(s["energy_kwh"] for s in segments),18)
        self.assertEqual(excluded,0)

    def test_four_hour_aggregation_uses_24_periods(self):
        c=np.arange(144);d=np.zeros(144);s=np.full(145,6000)
        rows=four_hour_rows(c,d,s)
        self.assertEqual(sum(row[1] for row in rows),float(c.sum()))
        self.assertEqual(rows[0][1],sum(range(24)))
        self.assertEqual(rows[-1][0],"20:00-24:00")

    def test_initial_boundary_tolerance_does_not_allow_real_violation(self):
        battery=Battery(.9,.9)
        battery.validate_initial(1200-1e-10)
        with self.assertRaises(ValueError):
            battery.validate_initial(1200-.001)

    def test_unsupported_model_configuration_is_not_ignored(self):
        cfg=json.loads((Path(__file__).resolve().parents[1]/"configs/q1_q2_baseline.json").read_text(encoding="utf-8"))
        validate_config(cfg)
        cfg["main_forecast"]="unimplemented_future_weather_model"
        with self.assertRaises(ValueError):
            validate_config(cfg)

    def test_negative_emergency_reporting_threshold_is_rejected(self):
        with self.assertRaises(ValueError):
            emergency_segments(np.zeros(144),threshold=-1)


if __name__=="__main__":
    unittest.main(verbosity=2)
