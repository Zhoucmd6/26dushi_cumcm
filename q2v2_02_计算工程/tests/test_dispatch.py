"""手算、物理边界、跨日衔接与分解一致性检查。"""
from pathlib import Path
import math
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from microgrid_core import Battery, solve_q1, simulate_fixed_plan, history_before_date
from stochastic_dispatch import solve_q2_fixed


class DispatchTests(unittest.TestCase):
    def setUp(self):
        self.battery=Battery(.9,.9)

    def test_two_period_energy_loss_matches_hand_calculation(self):
        for battery,expected in ((self.battery,100.),(Battery(math.sqrt(.9),math.sqrt(.9)),90.)):
            r=solve_q1([1,2],[0,81],[0,0],battery=battery,initial_kwh=1200)
            self.assertAlmostEqual(r["cost_yuan"],expected,places=6)
            self.assertAlmostEqual(sum(r["plan"]["C"]),expected,places=6)
            self.assertAlmostEqual(sum(r["plan"]["D"]),81,places=6)

    def test_full_battery_curtails_free_pv(self):
        r=solve_q1([1],[40],[100],battery=self.battery,initial_kwh=10800)
        self.assertAlmostEqual(r["cost_yuan"],0)
        self.assertAlmostEqual(r["plan"]["W"][0],60)
        self.assertEqual(r["simultaneous_charge_discharge_kwh"],0)

    def test_invalid_inputs_are_rejected(self):
        for kwargs in ({"eta_charge":1.1,"eta_discharge":.9},
                       {"eta_charge":.9,"eta_discharge":.9,"max_power_kw":0}):
            with self.assertRaises(ValueError):
                Battery(**kwargs)
        for load,initial in (([-1],6000),([1,2],6000),([1],0)):
            with self.assertRaises(ValueError):
                solve_q1([1],load,[0],battery=self.battery,initial_kwh=initial)

    def test_fixed_charging_can_trigger_emergency_purchase(self):
        r=simulate_fixed_plan([1],[100],[0],[100],[100],[0],battery=self.battery,
                              initial_kwh=6000,emergency_multiplier=5)
        self.assertEqual(r["E"],[100])
        self.assertEqual(r["cash_cost_yuan"],600)
        self.assertEqual(r["terminal_kwh"],6090)

    def test_total_unused_can_be_free_pv(self):
        r=simulate_fixed_plan([1],[40],[100],[0],[0],[0],battery=self.battery,
                              initial_kwh=6000,emergency_multiplier=5)
        self.assertEqual(r["U_total_unused"],[60])
        self.assertEqual(r["cash_cost_yuan"],0)

    def test_cross_day_state_is_carried_forward(self):
        day1=simulate_fixed_plan([1],[90],[0],[0],[0],[90],battery=self.battery,
                                 initial_kwh=6000,emergency_multiplier=5)
        day2=simulate_fixed_plan([1],[0],[0],[100],[100],[0],battery=self.battery,
                                 initial_kwh=day1["terminal_kwh"],emergency_multiplier=5)
        self.assertEqual(day1["terminal_kwh"],5900)
        self.assertEqual(day2["S"][0],5900)
        self.assertEqual(day2["terminal_kwh"],5990)

    def test_history_excludes_current_and_future_dates(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/"actual.csv"
            path.write_text("date,point_index,time_label,load_kw,pv_actual_kw\n"
                            "2025-01-31,144,24:00,100,0\n"
                            "2025-02-01,1,00:10,9999,0\n"
                            "2025-12-31,1,00:10,8888,0\n",encoding="utf-8")
            history=history_before_date(path,"2025-02-01")
        self.assertEqual(len(history),1)
        self.assertEqual(history[0]["load_kw"],100)

    def test_single_period_quantile_matches_hand_calculation(self):
        for method in ("extensive","benders"):
            r=solve_q2_fixed([1],[[100],[200],[300]],[[0],[0],[0]],[.6,.3,.1],
                             battery=self.battery,initial_kwh=1200,terminal_kwh=1200,
                             terminal_cost_per_kwh=0,method=method)
            self.assertAlmostEqual(r["plan"]["Q"][0],200,places=6)
            self.assertAlmostEqual(r["objective_yuan"],250,places=6)

    def test_benders_matches_extensive_with_battery_and_pv(self):
        p=[.4,.4,.7,1.2,1.,.5]
        net=np.array([[220,250,-100,300,650,200],
                      [250,200,-200,500,700,250],
                      [300,300,50,650,850,300]],dtype=float)
        results=[solve_q2_fixed(p,np.maximum(net,0),np.maximum(-net,0),[.5,.3,.2],
                               battery=self.battery,initial_kwh=1200,terminal_kwh=1200,
                               terminal_cost_per_kwh=0,method=m) for m in ("extensive","benders")]
        self.assertAlmostEqual(results[0]["objective_yuan"],1223.5596707818934,places=5)
        self.assertAlmostEqual(results[0]["objective_yuan"],results[1]["objective_yuan"],places=5)
        self.assertLess(results[1]["absolute_gap_yuan"],1e-5)

    def test_terminal_value_is_separate_from_cash_cost(self):
        for method in ("extensive","benders"):
            r=solve_q2_fixed([1],[[0]],[[0]],[1],battery=self.battery,initial_kwh=6000,
                             terminal_kwh=None,terminal_cost_per_kwh=-.2,method=method)
            self.assertAlmostEqual(r["expected_cash_cost_yuan"],0,places=6)
            self.assertAlmostEqual(r["terminal_cost_yuan"],-1200,places=6)
            self.assertAlmostEqual(r["objective_yuan"],-1200,places=6)

    def test_probabilities_are_not_silently_normalized(self):
        with self.assertRaises(ValueError):
            solve_q2_fixed([1],[[100],[200]],[[0],[0]],[.5,.4],battery=self.battery,
                           initial_kwh=6000,terminal_kwh=6000,terminal_cost_per_kwh=0)


if __name__=="__main__":
    unittest.main(verbosity=2)
