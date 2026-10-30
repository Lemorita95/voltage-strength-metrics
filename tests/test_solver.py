"""Targeted solver checks (stdlib unittest, no extra dependencies).

Run from the repository root:
    python -m unittest discover -s tests -v
"""
import contextlib
import csv
import io
import unittest
import warnings
from pathlib import Path

import numpy as np

from models.Buses import Bus
from models.CustomNetwork import CustomNetwork
from models.PowerFlow import PowerFlow
from models.VoltageLimits import VoltageLimits

ROOT = Path(__file__).resolve().parents[1]
CASE = ROOT / 'cases' / 'glover37.json'
PW_BUSES = ROOT / 'data' / 'powerworld' / 'glover37' / 'base' / 'buses.csv'


def load_case():
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        net = CustomNetwork(str(CASE))
        net.create_network()
        net.build_ybus()
    return net


def solve(net, **kwargs):
    pf = PowerFlow()
    with contextlib.redirect_stdout(io.StringIO()):
        result = pf.solve_power_flow(net, **kwargs)
    if result is None:
        raise AssertionError('power flow did not converge')
    return pf, result


def powerworld_v_pu():
    with PW_BUSES.open(encoding='utf-8-sig') as f:
        f.readline()  # object-type line
        rows = list(csv.DictReader(f))
    return {int(r['Number']): float(r['Vpu'].strip().replace(',', '.')) for r in rows}


class BaseCaseTest(unittest.TestCase):
    def test_base_matches_powerworld_without_tap_moves(self):
        net = load_case()
        pf, (_, v) = solve(net)
        pw = powerworld_v_pu()
        err = max(abs(v[net.idx[b]] - pw[b]) for b in net.buses)
        self.assertLess(err, 1e-4)
        self.assertEqual(sum(abs(s) for s in pf.control_steps.values()), 0)


class GeneratorQLimitTest(unittest.TestCase):
    GEN = 'generator.g44-1'
    BUS = 44

    def test_lock_at_qmax_then_release(self):
        net = load_case()
        gen = net.generators[self.GEN]
        q_max = gen.Q_max

        # a Qmax below what the generator needs locks it at Qmax (PQ)
        gen.Q_max = 0.0
        pf, locked = solve(net, controls=False)
        self.assertEqual(pf.pv_state[self.BUS], 1)

        # restored limit: starting from the locked state, it returns to PV
        gen.Q_max = q_max
        pf, released = solve(net, controls=False, warm_start=locked, pv_state=pf.pv_state)
        self.assertEqual(pf.pv_state[self.BUS], 0)
        _, fresh = solve(net, controls=False)
        np.testing.assert_allclose(released[1], fresh[1], atol=1e-8)


class TapControllerTest(unittest.TestCase):
    CONTROLLER = 'tap_controller.12-40'

    def raised_band_case(self):
        net = load_case()
        c = net.tap_controllers[self.CONTROLLER]
        c.v_low, c.v_high, c.auto = 1.035, 1.045, True
        return net, c

    def test_steps_regulated_bus_into_band(self):
        net, c = self.raised_band_case()
        pf, (_, v) = solve(net)
        v_reg = v[net.idx[c.reg_bus]]
        self.assertGreater(pf.control_steps.get(c.name, 0), 0)
        at_limit = not all(t.can_step(c.raise_direction) for t in c.transformers)
        self.assertTrue(c.v_low <= v_reg <= c.v_high or at_limit, f'V={v_reg:.4f}')

    def test_controls_false_holds_taps(self):
        net, c = self.raised_band_case()
        taps = c.taps
        pf, _ = solve(net, controls=False)
        self.assertEqual(c.taps, taps)
        self.assertEqual(pf.control_steps, {})


class VoltageLimitsTest(unittest.TestCase):
    def setUp(self):
        self.limits = VoltageLimits([{'kv': [0, 300], 'pu': [0.90, 1.05]},
                                     {'kv': [300, None], 'pu': [0.95, 1.10]}],
                                    fallback_pu=[0.80, 1.20])

    def test_range_boundaries(self):
        self.assertEqual(self.limits.band_pu(Bus(1, nom_kv=299.9)), (0.90, 1.05))
        self.assertEqual(self.limits.band_pu(Bus(2, nom_kv=300.0)), (0.95, 1.10))

    def test_missing_kv_uses_fallback_and_is_listed(self):
        buses = {1: Bus(1, nom_kv=None), 2: Bus(2, nom_kv=138.0),
                 3: Bus(3, nom_kv=None, v_min_pu=0.9, v_max_pu=1.1)}
        self.assertEqual(self.limits.band_pu(buses[1]), (0.80, 1.20))
        self.assertEqual(self.limits.band_pu(buses[3]), (0.9, 1.1))
        self.assertEqual(self.limits.unclassified(buses), [1])


if __name__ == '__main__':
    unittest.main()
