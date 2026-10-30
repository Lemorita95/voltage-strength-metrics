"""Solver-integrity checks of models/: element models, Newton-Raphson power flow,
short-circuit solver.

Fixtures: one folder per network under fixtures/networks/, each a frozen case the
tests own (they read neither cases/ nor data/):
    case.json       the network, CustomNetwork format (required)
    reference.csv   optional known solution of the case as given, one row per bus,
                    any of: V_pu, V_angle_deg (pre-fault power flow),
                    I_scc_pu, I_scc_angle_deg (bolted three-phase fault)
Every solver check runs on every network; a new network is a new folder.
glover37/ holds the Glover 37-bus case and PowerWorld's base solution of it.

Run from the repository root:
    python -m unittest discover -s tests -v
"""
import contextlib
import functools
import io
import unittest
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from models.CustomNetwork import CustomNetwork
from models.Lines import Line
from models.PowerFlow import PowerFlow, build_jacobian
from models.ShortCircuit import ShortCircuit as SC
from models.Transformers import Transformer

NETWORKS_DIR = Path(__file__).resolve().parent / 'fixtures' / 'networks'
NETWORKS = sorted(p for p in NETWORKS_DIR.iterdir() if (p / 'case.json').is_file())
GLOVER37 = NETWORKS_DIR / 'glover37'
# agreement with reference.csv (PowerWorld exports |V| to 1e-5 pu, angles to 0.01 deg)
TOLERANCES = {'V_pu': 5e-5, 'V_angle_deg': 0.01, 'I_scc_rel': 1e-4, 'I_scc_angle_deg': 0.01}
ZF = 1e-6  # SCC_NR's default fault impedance


def load_case(folder):
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        net = CustomNetwork(str(folder / 'case.json'))
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


@functools.cache
def solved(folder):
    """The case solved once: power flow, then a bolted fault at every bus."""
    net = load_case(folder)
    pf, (theta, v) = solve(net)
    x = np.r_[theta, v]
    v_complex = v * np.exp(1j * theta)
    active = net.get_active_elements(v_complex, pf.get_active_elements_flow(net, x),
                                     pf.get_active_element_status(net))
    with contextlib.redirect_stdout(io.StringIO()):
        current = {b: SC.SCC_NR(net.idx[b], v_complex, net.YBus, active) for b in net.buses}
    return SimpleNamespace(net=net, pf=pf, x=x, n=len(theta), v=v_complex, active=active,
                           current=current)


def injections(s, x):
    return np.r_[s.pf.PQ(s.net.YBus, x)].ravel()


def stamp(element):
    """Dense 2x2 primitive admittance of a two-terminal element, terminals (bus_1, bus_2)."""
    rows, cols, data = element.get_Y()
    pos = {element.bus_1: 0, element.bus_2: 1}
    y = np.zeros((2, 2), dtype=complex)
    for r, c, d in zip(rows, cols, data):
        y[pos[r], pos[c]] += d
    return y


class ElementStampTest(unittest.TestCase):

    def test_line_carries_only_charging_at_equal_voltages(self):
        line = Line('l', 1, 2, 0.01 + 0.1j, Y_shunt=0.02j)
        v = 1.02 * np.exp(-0.3j)
        np.testing.assert_allclose(stamp(line) @ [v, v], [0.01j * v, 0.01j * v], atol=1e-15)

    def test_transformer_unloaded_at_its_ratio(self):
        # V_bus1 = m V_bus2 drives no current through the series impedance
        t = Transformer('t', 1, 2, 0.005 + 0.08j, m_min=0.9, m_max=1.1, m_step=0.00625, n0=4)
        v = 0.98 * np.exp(0.2j)
        np.testing.assert_allclose(stamp(t) @ [t.m * v, v], [0, 0], atol=1e-13)


class NetworkTestCase(unittest.TestCase):

    def networks(self):
        """[(folder, solved case)] for every fixture network; tests open a subTest per network."""
        self.assertTrue(NETWORKS, f'no fixture network in {NETWORKS_DIR}')
        return [(folder, solved(folder)) for folder in NETWORKS]


class PowerFlowTest(NetworkTestCase):

    def test_injections_match_dense_complex_power(self):
        for folder, s in self.networks():
            with self.subTest(network=folder.name):
                sv = s.v * np.conj(s.net.YBus @ s.v)
                np.testing.assert_allclose(injections(s, s.x), np.r_[sv.real, sv.imag], atol=1e-12)

    def test_jacobian_matches_finite_differences(self):
        for folder, s in self.networks():
            with self.subTest(network=folder.name):
                jac = build_jacobian(s.net.YBus, s.x).toarray()
                h, numeric = 1e-7, np.zeros_like(jac)
                for k in range(2 * s.n):
                    step = np.zeros(2 * s.n)
                    step[k] = h
                    numeric[:, k] = (injections(s, s.x + step) - injections(s, s.x - step)) / (2 * h)
                self.assertLess(abs(jac - numeric).max() / abs(jac).max(), 1e-7)

    def test_quadratic_convergence(self):
        # fixed bus types from a flat start; the error roughly squares at every iteration
        for folder, s in self.networks():
            with self.subTest(network=folder.name):
                net, n = s.net, s.n
                pq, pv = net.buses_type()
                slack = net.idx[net.slack_id]
                pv_idx = np.array(sorted(net.idx[b] for b in pv), dtype=int)
                flat = np.r_[np.zeros(n), np.ones(n)]
                for g in net.generators.values():
                    flat[n + net.idx[g.bus_1]] = g.V
                flat[n + slack] = net.buses[net.slack_id].V
                spec = np.r_[[net.buses[net.id[i]].P for i in range(n)],
                             [net.buses[net.id[i]].Q for i in range(n)]]
                unknown = np.r_[[net.idx[b] for b in pq | pv], [n + net.idx[b] for b in pq]].astype(int)
                errors = []
                for iterations in range(1, 7):
                    x, _, _ = s.pf._newton_raphson_inner(flat.copy(), spec, net.YBus, slack, pv_idx,
                                                          iterations, 0.0)
                    errors.append(abs(spec - injections(s, x))[unknown].max())
                self.assertLess(errors[-1], 1e-10)
                for before, after in zip(errors, errors[1:]):
                    self.assertLess(after, max(10 * before ** 2, 1e-12))  # 1e-12: round-off floor

    def test_solution_meets_specification(self):
        for folder, s in self.networks():
            with self.subTest(network=folder.name):
                net, n = s.net, s.n
                net.compute_bus_net_power()
                pq, pv = net.buses_type()
                p, q = injections(s, s.x)[:n], injections(s, s.x)[n:]
                v, theta = s.x[n:], s.x[:n]
                q_load = {b: 0.0 for b in net.buses}
                for load in net.loads.values():
                    if load.status:
                        q_load[load.bus_1] += load.Q
                for b in pq:
                    i = net.idx[b]
                    self.assertAlmostEqual(p[i], net.buses[b].P, places=9)
                    self.assertAlmostEqual(q[i], net.buses[b].Q, places=9)
                setpoint = {g.bus_1: g.V for g in net.generators.values() if g.status}
                for b in pv:
                    i, state = net.idx[b], s.pf.pv_state[b]
                    self.assertAlmostEqual(p[i], net.buses[b].P, places=9)
                    if state == 0:
                        self.assertAlmostEqual(v[i], setpoint[b], places=12)
                    else:  # held at the reactive limit it crossed
                        limit = net.buses[b].Qg_max if state == 1 else net.buses[b].Qg_min
                        self.assertAlmostEqual(q[i] + q_load[b], limit, places=9)
                slack = net.idx[net.slack_id]
                self.assertAlmostEqual(v[slack], net.buses[net.slack_id].V, places=12)
                self.assertAlmostEqual(np.degrees(theta[slack]), net.buses[net.slack_id].angle,
                                       places=12)

    def test_power_balance(self):
        # total bus injection = series and shunt branch losses + shunt capacitors
        for folder, s in self.networks():
            with self.subTest(network=folder.name):
                total = (s.v * np.conj(s.net.YBus @ s.v)).sum()
                flows = s.pf.element_flow(s.net, s.x)
                absorbed = (sum(f['Sij'] + f['Sji'] for k in ('Lines', 'Transformers') for f in flows[k])
                            + sum(f['Sij'] for f in flows['Capacitors']))
                self.assertLess(abs(total - absorbed), 1e-10)


class ShortCircuitTest(NetworkTestCase):

    def test_sources_reproduce_pre_fault_state(self):
        # without a fault, the Norton sources and constant-Z loads hold the power-flow voltages
        for folder, s in self.networks():
            with self.subTest(network=folder.name):
                residual = SC.compute_residual(s.v.copy(), SC.SCC_tasks(s.active), s.net.YBus.tocsc())
                self.assertLess(abs(residual).max(), 1e-10)

    def test_fault_current_is_pre_fault_voltage_over_thevenin(self):
        # with linear elements only: I_f = V_pre / (Z_ii + zf)
        for folder, s in self.networks():
            with self.subTest(network=folder.name):
                if any(e[0].element_type == 'full_converter' for e in s.active.values()):
                    continue  # converter injections depend on the fault voltage
                z = np.linalg.inv(SC.build_Y_SCC(s.v, SC.SCC_tasks(s.active), s.net.YBus).toarray())
                for b, current in s.current.items():
                    i = s.net.idx[b]
                    self.assertLess(abs(current / (s.v[i] / (z[i, i] + ZF)) - 1), 1e-9, f'bus {b}')


class ReferenceTest(NetworkTestCase):
    """Agreement with each network's known solution, for the columns reference.csv has."""

    def test_matches_reference(self):
        for folder, s in self.networks():
            with self.subTest(network=folder.name):
                path = folder / 'reference.csv'
                if not path.is_file():
                    continue
                ref = pd.read_csv(path).set_index('bus')
                self.assertEqual(set(ref.index), set(s.net.buses))
                for b in s.net.buses:
                    i, r = s.net.idx[b], ref.loc[b]
                    current = s.current[b]
                    if 'V_pu' in r:
                        self.assertAlmostEqual(s.x[s.n + i], r['V_pu'], delta=TOLERANCES['V_pu'],
                                               msg=f'bus {b}')
                    if 'V_angle_deg' in r:
                        self.assertAlmostEqual(np.degrees(s.x[i]), r['V_angle_deg'],
                                               delta=TOLERANCES['V_angle_deg'], msg=f'bus {b}')
                    if 'I_scc_pu' in r:
                        self.assertLess(abs(abs(current) / r['I_scc_pu'] - 1), TOLERANCES['I_scc_rel'],
                                        f'bus {b}')
                    if 'I_scc_angle_deg' in r:
                        angle = (np.degrees(np.angle(current)) - r['I_scc_angle_deg'] + 180) % 360 - 180
                        self.assertLess(abs(angle), TOLERANCES['I_scc_angle_deg'], f'bus {b}')


class GeneratorQLimitTest(unittest.TestCase):
    """Glover 37: generator g44-1 at bus 44."""
    GEN = 'generator.g44-1'
    BUS = 44

    def test_lock_at_qmax_then_release(self):
        net = load_case(GLOVER37)
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
    """Glover 37: tap controller 12-40."""
    CONTROLLER = 'tap_controller.12-40'

    def raised_band_case(self):
        net = load_case(GLOVER37)
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


if __name__ == '__main__':
    unittest.main()
