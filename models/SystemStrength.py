"""System-strength screening metrics, with independent per-state histories.

Study conventions:
  SCR/WSCR: pre-installation 37-bus network (no proposed converter equipment).
  ESCR: augmented network, 34.5 kV padmount-HV buses; CIGRE multi-infeed ESCR
        (eq. 1-4, Qf = 0) with impedance and voltage-step interaction factors.
  SDSCR: augmented network, 0.69 kV converter terminals, Wu (2018).

One framework for every metric:
  SCMVA       the short-circuit solver (ShortCircuit.SCC_NR), a bolted fault started
              from the solved pre-fault state: SCMVA = S_base * V_n * |I_f|, V_n = 1 pu.
  impedances  the solver's admittance builder (ShortCircuit.build_Y_SCC): passive Y-bus,
              source admittances and load admittances at the pre-fault voltages.
  voltages    the pre-fault power flow supplied by the experiment runner.
Each converter injects what its own fault model returns (FullConverter.fault_model).

All voltages and impedances are on the 100 MVA common base. The calculation
state (including synchronous commitment) is supplied by the experiment runner.
"""
from __future__ import annotations

from copy import deepcopy
from contextlib import redirect_stdout
from dataclasses import dataclass, field
import io
import warnings
from typing import Mapping

import numpy as np
from scipy.sparse import csc_matrix
from scipy.sparse.linalg import splu

from models.Generators import Generator
from models.PowerFlow import PowerFlow
from models.ShortCircuit import ShortCircuit as SC

S_BASE_MVA = 100.0
EPS = 1e-12
FAULT_Z_PU = 1e-6  # SCC_NR's bolted-fault impedance


def fault_elements(net, v_pf):
    """Active sources and loads of the solved pre-fault state, as the short-circuit
    solver takes them: {(name, bus index): (element, V_pre, S_pre)}."""
    v = np.asarray(v_pf, dtype=complex)
    if v.shape != (len(net.idx),):
        raise ValueError("Expected complete complex pre-fault voltage vector")
    pf = PowerFlow()
    state = np.r_[np.angle(v), np.abs(v)]
    return net.get_active_elements(v, pf.get_active_elements_flow(net, state),
                                   pf.get_active_element_status(net))


def short_circuit_admittance(net, v_pf, elements=None):
    """Y_SCC from the short-circuit solver's builder (no fault admittance): the passive
    YBus plus each active element's admittance at the pre-fault state. Converters
    contribute none (FullConverter.get_Y)."""
    if not hasattr(net, "YBus") or not hasattr(net, "idx"):
        raise ValueError("Build net.YBus first")
    elements = fault_elements(net, v_pf) if elements is None else elements
    return SC.build_Y_SCC(np.asarray(v_pf, dtype=complex), SC.SCC_tasks(elements), net.YBus)


def z_block(net, bus_ids, *, v_pf, y=None):
    """(Y_SCC^-1)[R,R], calculated using sparse solves."""
    bus_ids = tuple(int(b) for b in bus_ids)
    if not bus_ids or len(bus_ids) != len(set(bus_ids)):
        raise ValueError("Evaluation buses must be distinct")
    indices = np.asarray([net.idx[b] for b in bus_ids], dtype=int)
    if y is None:
        y = short_circuit_admittance(net, v_pf)
    rhs = np.zeros((y.shape[0], len(indices)), dtype=complex)
    rhs[indices, np.arange(len(indices))] = 1
    return splu(csc_matrix(y)).solve(rhs)[indices, :]


def fault_scmva(net, bus_ids, *, v_pf, elements=None, s_base_mva=S_BASE_MVA, zf=FAULT_Z_PU):
    """{bus: SCMVA} = S_base * V_n * |I_f| (V_n = 1 pu), one SCC_NR bolted fault per bus
    from the solved pre-fault state. A fault that does not converge raises."""
    v = np.asarray(v_pf, dtype=complex)
    elements = fault_elements(net, v) if elements is None else elements
    out = {}
    for bus in bus_ids:
        i_fault = SC.SCC_NR(net.idx[bus], v, net.YBus, elements, zf=zf)
        if i_fault is None:
            raise RuntimeError(f'Short circuit did not converge for a fault at bus {bus}')
        out[bus] = s_base_mva * 1.0 * abs(i_fault)
    return out


def _voltage(net, v, bus):
    v = np.asarray(v, dtype=complex)
    if v.shape != (len(net.idx),):
        raise ValueError('Expected complete complex pre-fault voltage vector')
    return v[net.idx[bus]]


def _ratio(num, den):
    if abs(den) < EPS or not (np.isfinite(num) and np.isfinite(den)):
        return float('nan')
    return float(num / den)


@dataclass
class _MetricHistory:
    """Each recorded state contains inputs, intermediate values, and result."""
    history: list[dict] = field(default_factory=list, init=False)

    def _record(self, step, **values):
        if any(r['step'] == step for r in self.history):
            raise ValueError(f'Duplicate metric state: {step!r}')
        record = {'step': step, **values}
        self.history.append(record)
        return record


@dataclass
class SCR(_MetricHistory):
    """IEEE 2800 POI SCR, pre-installation 37-bus reference network."""
    rated_mw: Mapping[int, float] = field(default_factory=dict)
    s_base_mva: float = S_BASE_MVA

    def evaluate(self, step, net, v_pf, scmva=None):
        """scmva: {bus: SCMVA} already computed for this state (fault_scmva), or None."""
        if net.full_converters:
            raise ValueError('SCR requires no-IBR reference network')
        if any(b not in net.idx for b in self.rated_mw):
            raise ValueError('POI missing from reference network')
        buses = tuple(self.rated_mw)
        if scmva is None:
            scmva = fault_scmva(net, buses, v_pf=v_pf, s_base_mva=self.s_base_mva)
        z = z_block(net, buses, v_pf=v_pf)
        values = {bus: _ratio(scmva[bus], self.rated_mw[bus]) for bus in buses}
        return self._record(step, value=values, scmva={b: scmva[b] for b in buses},
                            v_pf={b: _voltage(net, v_pf, b) for b in buses},
                            z_ii=dict(zip(buses, np.diag(z))),
                            rated_mw=dict(self.rated_mw))


@dataclass
class WSCR(_MetricHistory):
    """IEEE 2800 grouped WSCR, same no-IBR SCMVA convention as SCR."""
    rated_mw: Mapping[int, float] = field(default_factory=dict)
    s_base_mva: float = S_BASE_MVA

    def evaluate(self, step, net, v_pf, scmva=None):
        """scmva: {bus: SCMVA} already computed for this state (fault_scmva), or None."""
        if net.full_converters:
            raise ValueError('WSCR requires no-IBR reference network')
        buses = tuple(self.rated_mw)
        if scmva is None:
            scmva = fault_scmva(net, buses, v_pf=v_pf, s_base_mva=self.s_base_mva)
        z = z_block(net, buses, v_pf=v_pf)
        p_total = sum(self.rated_mw.values())
        numerator = sum(scmva[b] * self.rated_mw[b] for b in buses)
        return self._record(step, value=_ratio(numerator, p_total*p_total),
                            scmva={b: scmva[b] for b in buses}, rated_mw=dict(self.rated_mw),
                            z_ii=dict(zip(buses, np.diag(z))))


@dataclass
class ESCR(_MetricHistory):
    """CIGRE multi-infeed ESCR (eq. 1-4) at the MV evaluation buses, with Qf = 0:

        ESCR_i = SCMVA_i / (P_i + sum_j MIIF_ji * P_j),   MIIF_ji = dV_j / dV_i,

    a disturbance at bus i. The MIIF is evaluated in several ways (variants), all
    sharing the state's SCMVA:
      'impedance'   |Z_ji / Z_ii| from Y_SCC: the instant network response (CIGRE)
      probe step s  a temporary ideal numerical PV probe holds bus i at (1 + s) times
                    its pre-fault voltage magnitude in a power flow; IF[j,i] =
                    | |V_j(new)| - |V_j(base)| | / | |V_i(new)| - |V_i(base)| |.

    The probe has infinite series impedance (zero Y-bus contribution), zero active
    power, and wide reactive limits. It is added to a deep copy of the solved operating
    state, so neither the production network nor other perturbations are mutated.
    This avoids a Q-injection root search and its possible PV/PQ solution-branch
    discontinuities. It is an explicitly prescribed voltage-step method, not a
    claim of exact reproduction of any particular grid-code implementation.
    """
    rated_mw: Mapping[int, float] = field(default_factory=dict)
    s_base_mva: float = S_BASE_MVA
    probe_steps: tuple = (0.01,)
    impedance: bool = True
    voltage_tolerance_pu: float = 1e-5
    probe_q_limit_pu: float = 1e6

    def _solve_voltage_step(self, net, v0, bus, relative_step, base_state=None):
        """Solve one prescribed voltage step on a disposable network copy.

        The solve starts from the base voltages and generator states (base_state).
        Returns also the generators whose state changed during the step,
        {generator bus: (base state, probe state)}.
        """
        working = deepcopy(net)
        name = f'generator.escr_voltage_probe_{bus}'
        if name in working.generators:
            raise ValueError(f'Temporary generator name collision: {name}')

        base_mag = abs(_voltage(net, v0, bus))
        target_mag = base_mag * (1.0 + relative_step)
        # Infinite Z contributes zero source admittance; the active generator
        # object is used only to classify this bus as PV and prescribe |V|.
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            with redirect_stdout(io.StringIO()):
                probe = Generator(
                    name=name,
                    bus_1=bus,
                    Z_series=float('inf'),
                    V=float(base_mag),
                    P=0.0,
                    Q=0.0,
                    Q_max=self.probe_q_limit_pu,
                    Q_min=-self.probe_q_limit_pu,
                    status=True,
                )
        working.generators[name] = probe
        probe.V = float(target_mag)

        # Taps (and any other discrete controllers) stay frozen during the near-instant
        # voltage step; the solve starts from the solved base state.
        pf = PowerFlow()
        v0 = np.asarray(v0, dtype=complex)
        solved = pf.solve_power_flow(working, warm_start=(np.angle(v0), np.abs(v0)),
                                     pv_state=base_state, controls=False)
        if solved is None:
            raise RuntimeError(f'Power flow failed for prescribed voltage step at bus {bus}')
        angles, magnitudes = solved
        v_new = np.asarray(magnitudes, dtype=float) * np.exp(1j * np.asarray(angles, dtype=float))
        actual_mag = abs(_voltage(working, v_new, bus))
        if not np.isfinite(actual_mag) or abs(actual_mag - target_mag) > self.voltage_tolerance_pu:
            raise RuntimeError(
                f'Voltage-step target not met at bus {bus}: '
                f'target={target_mag:.8f} pu, actual={actual_mag:.8f} pu'
            )

        # Record the probe's reactive output as an audit variable. The evaluated
        # bus has no other generator in this study, so this is its probe Q.
        # bus_flow() returns (PQ_net, PQ_gen, PQ_converter, PQ_load, PQ_shunt);
        # each PQ array has shape (2, n_bus), with P in row 0 and Q in row 1.
        pq_gen = pf.bus_flow(working, np.r_[angles, magnitudes])[1]
        probe_q_pu = pq_gen[1, working.idx[bus]]
        probe_q_mvar = float(probe_q_pu) * self.s_base_mva

        # generators (other than the probe) that changed PV/PQ state during the step
        base_state = base_state or {}
        pv_changes = {g: (base_state[g], s) for g, s in pf.pv_state.items()
                      if g in base_state and s != base_state[g]}
        return probe_q_mvar, target_mag, actual_mag, v_new, pv_changes

    def _values(self, scmva, interaction):
        buses = tuple(self.rated_mw)
        return {i: _ratio(scmva[i], self.rated_mw[i] + sum(
                    interaction[(j, i)] * self.rated_mw[j] for j in buses if j != i))
                for i in buses}

    def _probe(self, step, net, v_pf, relative_step, base_state):
        buses = tuple(self.rated_mw)
        interaction, probe_q, target_v, actual_v, delta_v, pv_changes = {}, {}, {}, {}, {}, {}
        for i in buses:
            q_mvar, target_mag, actual_mag, v_new, changes = self._solve_voltage_step(
                net, v_pf, i, relative_step, base_state)
            pv_changes[i] = len(changes)  # generator mode changes during the step (recorded, not warned)
            probe_q[i] = q_mvar
            target_v[i] = target_mag
            actual_v[i] = actual_mag
            delta_v[i] = {
                j: abs(_voltage(net, v_new, j)) - abs(_voltage(net, v_pf, j))
                for j in buses
            }
            dv_i = delta_v[i][i]
            if not np.isfinite(dv_i) or abs(dv_i) < EPS:
                raise RuntimeError(f'Invalid local voltage response at ESCR bus {i}: {dv_i}')
            for j in buses:
                if j != i:
                    interaction[(j, i)] = _ratio(abs(delta_v[i][j]), abs(dv_i))
        return {'if_magnitude': interaction, 'probe_reactive_power_mvar': probe_q,
                'probe_pv_changes': pv_changes, 'target_voltage_pu': target_v,
                'actual_voltage_pu': actual_v, 'delta_v_pu': delta_v}

    def evaluate(self, step, net, v_pf, pv_state):
        """{variant: record}, variant 'impedance' or a probe step (float). Every record
        holds value, if_magnitude and the shared terms (scmva, v_pf, z_ii, rated_mw);
        probe records add their audit terms.

        pv_state: generator states (0 PV, +1 at Qmax, -1 at Qmin) of the pre-fault solve
        that produced v_pf (PowerFlow.pv_state); every probe starts from them."""
        buses = tuple(self.rated_mw)
        elements = fault_elements(net, v_pf)
        scmva = fault_scmva(net, buses, v_pf=v_pf, elements=elements, s_base_mva=self.s_base_mva)
        z = z_block(net, buses, v_pf=v_pf, y=short_circuit_admittance(net, v_pf, elements))
        shared = {'scmva': scmva, 'rated_mw': dict(self.rated_mw),
                  'v_pf': {b: _voltage(net, v_pf, b) for b in buses},
                  'z_ii': dict(zip(buses, np.diag(z)))}

        variants = {}
        if self.impedance:
            interaction = {(j, i): _ratio(abs(z[kj, ki]), abs(z[ki, ki]))
                           for ki, i in enumerate(buses) for kj, j in enumerate(buses) if j != i}
            variants['impedance'] = {'if_magnitude': interaction}
        base_state = dict(pv_state)
        for relative_step in self.probe_steps:
            variants[relative_step] = self._probe(step, net, v_pf, relative_step, base_state)
        for record in variants.values():
            record.update(value=self._values(scmva, record['if_magnitude']), **shared)
        self._record(step, variants=variants)
        return variants


@dataclass
class SDSCR(_MetricHistory):
    """Wu (2018) terminal-based SDSCR using complex coupling weights."""
    rated_mw: Mapping[int, float] = field(default_factory=dict)
    s_base_mva: float = S_BASE_MVA

    def evaluate(self, step, net, v_pf):
        buses = tuple(self.rated_mw)
        z = z_block(net, buses, v_pf=v_pf)
        voltages = {b: _voltage(net, v_pf, b) for b in buses}
        weights = {}
        values = {}
        for k, i in enumerate(buses):
            zii, vi = z[k,k], voltages[i]
            if abs(zii) < EPS:
                values[i] = float('nan')
                continue
            effective_p = complex(self.rated_mw[i] / self.s_base_mva)
            for h, j in enumerate(buses):
                if i == j:
                    continue
                vj = voltages[j]
                w = (z[k,h] / zii) * np.conjugate(vi/vj) if abs(vj) > EPS else complex('nan')
                weights[(i,j)] = w
                effective_p += w * self.rated_mw[j] / self.s_base_mva
            values[i] = _ratio(abs(vi)**2, abs(zii)*abs(effective_p))
        return self._record(step, value=values, w_complex=weights,
                            v_pf=voltages, z_rr=z.copy(),
                            rated_mw=dict(self.rated_mw))
