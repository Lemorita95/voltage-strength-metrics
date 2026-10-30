"""Checks of the metric framework (models.SystemStrength, models.FullConverter).

Each test guards a step that could silently produce a wrong claim:
- SCMVA from the short-circuit solver agrees with the builder's impedance (R1 = R3);
- ESCR's impedance MIIF equals |Z_ji / Z_ii| from a direct inversion of Y_SCC;
- both MIIF constructions put the disturbed bus i in the denominator (CIGRE eq. 1-1);
- a constant-current converter injects its pre-fault current whatever its voltage.

Run from the repository root:
    python -m unittest discover -s tests -v
"""
import contextlib
import io
import unittest
import warnings
from types import SimpleNamespace

import numpy as np

from experiments.glover37.config import GEN_LIMITS_FILE, IBR_MW, SITES
from experiments.glover37.network import (
    augment_network, baseline_load_mw, prepare_network, read_gen_limits, sequential_dispatch,
)
from experiments.glover37.simulate import solve_pf
from models.FullConverter import FullConverter
from models.SystemStrength import (
    ESCR, FAULT_Z_PU, S_BASE_MVA, fault_scmva, short_circuit_admittance,
)


def quiet(fn, *args, **kwargs):
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        return fn(*args, **kwargs)


def solved_state(ibr):
    """Intact state at load scale 1, 37-bus (ibr False) or 46-bus with converters."""
    limits = read_gen_limits(GEN_LIMITS_FILE)
    target = baseline_load_mw() - (IBR_MW if ibr else 0.0)
    net = prepare_network(1.0, sequential_dispatch(target, limits), limits)
    if ibr:
        augment_network(net)
    v, pf = quiet(solve_pf, net)
    return net, v, pf


class TestFrameworkConsistency(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.base = solved_state(ibr=False)
        cls.ibr = solved_state(ibr=True)

    def test_scmva_matches_builder_impedance(self):
        # With no element responding to the fault, the solver's bolted fault current is
        # V_pre / (Z_ii + zf), Z from the builder: SCMVA (R1) and Z (R3) agree.
        for net, v, _ in (self.base, self.ibr):
            buses = [s.poi_bus for s in SITES] + ([s.escr_bus for s in SITES] if net.full_converters else [])
            z = np.linalg.inv(short_circuit_admittance(net, v).toarray())
            scmva = quiet(fault_scmva, net, buses, v_pf=v)
            for b in buses:
                i = net.idx[b]
                expected = S_BASE_MVA * abs(v[i]) / abs(z[i, i] + FAULT_Z_PU)
                self.assertAlmostEqual(scmva[b] / expected, 1.0, places=11)

    def test_impedance_miif_is_z_ratio(self):
        net, v, pf = self.ibr
        rated = {s.escr_bus: s.pmax_mw for s in SITES}
        record = quiet(ESCR(rated, probe_steps=()).evaluate, 'test', net, v, pf.pv_state)['impedance']
        z = np.linalg.inv(short_circuit_admittance(net, v).toarray())
        for (j, i), value in record['if_magnitude'].items():
            self.assertAlmostEqual(value, abs(z[net.idx[j], net.idx[i]] / z[net.idx[i], net.idx[i]]),
                                   places=12)
        # and the metric is CIGRE eq. 1-4 with Qf = 0
        for i in rated:
            den = rated[i] + sum(record['if_magnitude'][(j, i)] * rated[j] for j in rated if j != i)
            self.assertAlmostEqual(record['value'][i], record['scmva'][i] / den, places=12)


class TestProbeOrientation(unittest.TestCase):

    def test_disturbed_bus_in_denominator(self):
        # A probe at bus i gives ΔV = dv[i] at every bus; MIIF_ji = |ΔV_j| / |ΔV_i|.
        buses = (1, 2, 3)
        net = SimpleNamespace(idx={b: k for k, b in enumerate(buses)})
        v0 = np.ones(3, dtype=complex)
        dv = {1: np.array([0.010, 0.004, 0.001]),
              2: np.array([0.002, 0.020, 0.005]),
              3: np.array([0.0005, 0.003, 0.030])}
        escr = ESCR({b: 1.0 for b in buses})
        escr._solve_voltage_step = lambda net, v, i, step, state: (0.0, 1.0, 1.0, v + dv[i], {})
        interaction = escr._probe('test', net, v0, 0.01, {})['if_magnitude']
        for i in buses:
            for j in buses:
                if j != i:
                    k_i, k_j = buses.index(i), buses.index(j)
                    self.assertAlmostEqual(interaction[(j, i)], dv[i][k_j] / dv[i][k_i], places=12)


class TestConverterFaultModel(unittest.TestCase):

    def converter(self, **kwargs):
        return FullConverter('c', 1, kVA_nominal=66000.0, cos_phi=1.0, kV_nominal=0.69,
                             P=0.6, Q=0.0, **kwargs)

    def test_constant_current_holds_pre_fault_current(self):
        conv = self.converter()
        self.assertEqual(conv.fault_model, 'constant_current')
        v_pre, s_pre = 1.02 * np.exp(0.1j), complex(0.6, 0.0)
        expected = np.conjugate(s_pre) / np.conjugate(v_pre)
        for v_terminal in (v_pre, 0.5 * v_pre, 0.0 + 0.0j):
            self.assertEqual(conv.get_I(v_terminal, v_pre, s_pre), expected)

    def test_unknown_fault_model_rejected(self):
        with self.assertRaises(ValueError):
            self.converter(fault_model='linear')


if __name__ == '__main__':
    unittest.main()
