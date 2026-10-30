"""Operating states of the Glover 37-bus case and its 46-bus IBR augmentation.

Public SE3 load profile and week selection, sequential dispatch, network
preparation (load scaling, commitment, line outages), the in-memory IBR
augmentation and the per-state audit columns. The SE3 data provide only the
temporal scaling signal; the benchmark is not an electrical model of SE3.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from models.Buses import Bus
from models.CustomNetwork import CustomNetwork
from models.FullConverter import FullConverter
from models.Lines import Line
from models.Transformers import Transformer
from experiments.glover37.config import (
    ALL_ONLINE, CASE_FILE, CONVERTER_FAULT_MODEL, FEEDER_B_SYSTEM_BASE, FEEDER_Z_SYSTEM_BASE,
    LV_KV, MV_KV, S_BASE_MVA, SE3_LOAD_COLUMN, SE3_TIMESTAMP_COLUMN, SEQUENTIAL, SITES,
    SLACK_BUS, WINDOW_HOURS, YEAR,
)


@dataclass(frozen=True)
class GenLimit:
    key: str
    bus: int
    unit_id: str
    mw_min: float
    mw_max: float
    priority: int


# =============================================================================
# 1. Public SE3 load profile
# =============================================================================

def num(value) -> float:
    """Parse ordinary or PowerWorld-style decimal-comma numbers."""
    return float(str(value).strip().replace(" ", "").replace(",", "."))


def norm_name(value: str) -> str:
    return (
        str(value).strip().lower()
        .replace("å", "a").replace("ä", "a").replace("ö", "o")
        .replace("_", " ").replace("-", " ")
    )


def choose_column(columns, explicit, kind):
    columns = list(columns)
    if explicit:
        if explicit not in columns:
            raise KeyError(f"{kind} column '{explicit}' not found. Columns: {columns}")
        return explicit

    normalized = {c: norm_name(c) for c in columns}
    if kind == "load":
        se3 = [c for c, n in normalized.items() if "se3" in n]
        if len(se3) == 1:
            return se3[0]
        tokens = ("consumption", "load", "demand", "forbrukning")
    else:
        tokens = ("timestamp", "datetime", "date time", "datum", "period", "time")

    matches = [c for c, n in normalized.items() if any(t in n for t in tokens)]
    if len(matches) != 1:
        raise ValueError(
            f"Could not identify one unique {kind} column. "
            f"Candidates: {matches}. Pass --{kind}-column explicitly."
        )
    return matches[0]


def read_se3(path: Path, timestamp_col=SE3_TIMESTAMP_COLUMN,
             load_col=SE3_LOAD_COLUMN) -> pd.DataFrame:
    """Read the exact public SE3 source file retained with the experiment."""
    if not path.exists():
        raise FileNotFoundError(f"SE3 file not found: {path}")

    if path.suffix.lower() in {".csv", ".txt"}:
        raw = pd.read_csv(path, sep=None, engine="python")
    elif path.suffix.lower() in {".xls", ".xlsx"}:
        try:
            raw = pd.read_excel(path)
        except ImportError as exc:
            raise ImportError(
                "Excel input needs an appropriate pandas engine. "
                "Easiest reproducible option: export the official file to CSV."
            ) from exc
    else:
        raise ValueError("SE3 input must be CSV, XLS, or XLSX.")

    tcol = choose_column(raw.columns, timestamp_col, "timestamp")
    lcol = choose_column(raw.columns, load_col, "load")

    df = raw[[tcol, lcol]].copy()
    df.columns = ["timestamp", "se3_mw"]
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["se3_mw"] = df["se3_mw"].map(lambda x: num(x) if not pd.isna(x) else np.nan)
    df = df.dropna().sort_values("timestamp").reset_index(drop=True)
    df = df[df["timestamp"].dt.year == YEAR].copy()

    if df["timestamp"].duplicated().any():
        raise ValueError("Duplicate timestamps found. Resolve timezone/DST representation first.")
    if len(df) < WINDOW_HOURS:
        raise ValueError(f"Need >= {WINDOW_HOURS} usable hourly observations.")

    return df


def select_week(profile: pd.DataFrame) -> pd.DataFrame:
    """Pick max-range continuous 168 h window using load data only."""
    best_start, best_range = None, -np.inf

    for start in range(len(profile) - WINDOW_HOURS + 1):
        w = profile.iloc[start:start + WINDOW_HOURS]
        steps = w["timestamp"].diff().dropna().dt.total_seconds() / 3600
        if not np.allclose(steps.to_numpy(), 1.0):
            continue

        spread = float(w["se3_mw"].max() - w["se3_mw"].min())
        if spread > best_range:
            best_start, best_range = start, spread

    if best_start is None:
        raise ValueError("No complete continuous 168-hour window found.")

    w = profile.iloc[best_start:best_start + WINDOW_HOURS].copy().reset_index(drop=True)
    w["hour_index"] = np.arange(WINDOW_HOURS)
    w["load_scale"] = w["se3_mw"] / w["se3_mw"].mean()
    return w[["hour_index", "timestamp", "se3_mw", "load_scale"]]


# =============================================================================
# 2. Generator priority and sequential dispatch
# =============================================================================

def read_gen_limits(path: Path) -> list[GenLimit]:
    """Read MWMin/MWMax from the existing PowerWorld generator export."""
    raw = pd.read_csv(path, skiprows=1, dtype=str)
    need = {"BusNum", "ID", "Status", "MWMin", "MWMax"}
    if missing := need.difference(raw.columns):
        raise KeyError(f"Missing columns in generators.csv: {sorted(missing)}")

    rows = []
    for _, r in raw.iterrows():
        bus = int(r["BusNum"].strip())
        uid = r["ID"].strip()
        pmin = num(r["MWMin"])
        pmax = num(r["MWMax"])

        if bus == SLACK_BUS or r["Status"].strip().lower() != "closed" or pmax <= 0:
            continue

        rows.append((bus, uid, pmin, pmax))

    # Locked rule: descending MWMax.
    # Deterministic tie-break independent of SCL.
    rows.sort(key=lambda x: (-x[3], x[0], x[1]))

    return [
        GenLimit(f"generator.g{bus}-{uid}".lower(), bus, uid, pmin, pmax, i + 1)
        for i, (bus, uid, pmin, pmax) in enumerate(rows)
    ]


def sequential_dispatch(target_load_mw: float, limits: list[GenLimit]) -> dict[str, float]:
    """
    Fill non-slack generators in priority order.

    If remaining demand is below the next generator MWMin, stop; slack supplies
    that residual. If non-slack MWMax is exhausted, slack supplies the remainder.
    """
    remaining = max(float(target_load_mw), 0.0)
    dispatch = {g.key: 0.0 for g in limits}

    for g in limits:
        if remaining <= 0 or remaining < g.mw_min:
            break

        p = min(g.mw_max, remaining)
        dispatch[g.key] = max(g.mw_min, p)
        remaining -= dispatch[g.key]

    return dispatch


def all_online_dispatch(target_load_mw: float, limits: list[GenLimit]) -> dict[str, float]:
    """
    Rule (a): every non-slack generator at max(MWMin, s * MWMax), s in [0, 1] chosen so
    the units meet the target. The sum is continuous and non-decreasing in s, so s is
    found by bisection. With s = 0 (MWMin alone exceeds the target) the slack exports;
    with s = 1 (MWMax exhausted) the slack supplies the remainder.
    """
    def total(s):
        return sum(max(g.mw_min, s * g.mw_max) for g in limits)

    target = float(target_load_mw)
    if total(0.0) >= target:
        s = 0.0
    elif total(1.0) <= target:
        s = 1.0
    else:
        lo, hi = 0.0, 1.0
        for _ in range(100):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if total(mid) < target else (lo, mid)
        s = 0.5 * (lo + hi)
    return {g.key: max(g.mw_min, s * g.mw_max) for g in limits}


def schedule(rule, target_load_mw, limits):
    """(dispatch MW, committed) per generator key under a commitment rule (config)."""
    if rule == SEQUENTIAL:
        dispatch = sequential_dispatch(target_load_mw, limits)
        return dispatch, {k: p > 0 for k, p in dispatch.items()}
    if rule == ALL_ONLINE:
        return all_online_dispatch(target_load_mw, limits), {g.key: True for g in limits}
    raise ValueError(f"Unknown commitment rule: {rule!r}")


def baseline_load_mw() -> float:
    net = CustomNetwork(str(CASE_FILE))
    net.create_network()
    return S_BASE_MVA * sum(load.P for load in net.loads.values())


# =============================================================================
# 3. One quasi-static operating state
# =============================================================================

def prepare_network(load_scale, dispatch, limits, contingency=None, committed=None):
    """Load-scaled, dispatched 37-bus network with the `contingency` line out of service
    (None: intact). `committed` (generator key -> bool) defaults to dispatch > 0."""
    net = CustomNetwork(str(CASE_FILE))
    net.create_network()

    for load in net.loads.values():
        load.P *= load_scale
        load.Q *= load_scale

    for g in limits:
        if g.key not in net.generators:
            raise KeyError(f"{g.key} not found in glover37.json")

        p_mw = dispatch[g.key]
        online = p_mw > 0 if committed is None else bool(committed[g.key])
        net.change_element_status(g.key, online)

        if online:
            net.generators[g.key].P = p_mw / S_BASE_MVA

    # Slack generator is deliberately untouched:
    # always connected and balancing.
    net.build_ybus()
    if contingency is not None:
        net.change_element_status(contingency, False)
        net.build_ybus()
    return net


def augment_network(net):
    """Augment an initialized CustomNetwork, rebuild YBus, return IBR sites.

    padmount_hv_bus is the 34.5 kV ESCR evaluation bus.
    IEEE SCR/WSCR use poi_bus; SDSCR uses sdscr_bus.
    This function expects net.create_network() to have run already.
    """
    all_new_ids = [bus for site in SITES for bus in
                   (site.main_mv_bus, site.padmount_hv_bus, site.converter_lv_bus)]
    if len(set(all_new_ids)) != len(all_new_ids):
        raise ValueError("Duplicate IBR bus IDs in configuration")
    if any(site.host_bus not in net.buses for site in SITES):
        raise ValueError("One or more original Glover host buses are missing")
    if any(bus in net.buses for bus in all_new_ids):
        raise ValueError("IBR bus IDs already exist; network may be augmented")

    for site in SITES:
        for bus_id, label, nom_kv in (
            (site.main_mv_bus, "COLLECTOR_MV", MV_KV),
            (site.padmount_hv_bus, "PADMOUNT_HV", MV_KV),
            (site.converter_lv_bus, "CONVERTER_LV", LV_KV),
        ):
            net.buses[bus_id] = Bus(bus_id, f"IBR{site.host_bus}_{label}", "PQ", None, None,
                                    nom_kv=nom_kv)

        main = f"transformer.ibr{site.host_bus}_main"
        feeder = f"line.ibr{site.host_bus}_collector"
        pad = f"transformer.ibr{site.host_bus}_padmount"
        converter = f"full_converter.ibr{site.host_bus}"
        if (main in net.transformers or pad in net.transformers or
                feeder in net.lines or converter in net.full_converters):
            raise ValueError(f"Duplicate equipment name at host {site.host_bus}")

        z = site.transformer_z_pu
        # On corresponding voltage bases, nominal transformer tap ratio is 1 pu.
        # No reg_bus: IBR transformer taps stay at n0 = 0 (m = m0 = 1.0) in this study.
        net.transformers[main] = Transformer(main, site.host_bus, site.main_mv_bus, z)
        net.lines[feeder] = Line(feeder, site.main_mv_bus, site.padmount_hv_bus,
                                 FEEDER_Z_SYSTEM_BASE, Y_shunt=1j * FEEDER_B_SYSTEM_BASE)
        net.transformers[pad] = Transformer(pad, site.padmount_hv_bus, site.converter_lv_bus, z)
        # FullConverter P, Q are per unit; rating is kVA, nominal voltage is kV.
        net.full_converters[converter] = FullConverter(
            name=converter, bus_1=site.converter_lv_bus,
            kVA_nominal=site.converter_mva * 1000.0, cos_phi=1.0, kV_nominal=LV_KV,
            P=site.pmax_mw / S_BASE_MVA, Q=0.0, fault_model=CONVERTER_FAULT_MODEL,
        )

    net.build_ybus()
    return SITES


# =============================================================================
# 4. Per-state audit columns
# =============================================================================
# Voltage violations are reported, never corrected (no curtailment in this study).
# Columns are flat so they append directly to a run_summary.csv row; `prefix`
# separates several networks solved for the same state (e.g. "base_", "ibr_").

def _join(items):
    return ';'.join(str(x) for x in items)


def operating_state_audit(net, pf, v_pu, limits, prefix=''):
    '''
        net     solved CustomNetwork
        pf      the PowerFlow instance that solved it (pv_state, control records)
        v_pu    |V| per bus, indexed like net.idx
        limits  VoltageLimits
    '''
    v_pu = np.asarray(v_pu, dtype=float)
    bus_of = {i: b for b, i in net.idx.items()}
    violations = limits.violations(net, v_pu)
    under = [x for x in violations if x.deviation_pu < 0]
    over = [x for x in violations if x.deviation_pu > 0]
    worst = violations[0] if violations else None
    pv_state = getattr(pf, 'pv_state', {}) or {}
    steps = getattr(pf, 'control_steps', {}) or {}

    audit = {
        'voltage_flag': bool(violations),
        'n_under_v': len(under),
        'n_over_v': len(over),
        'under_v_buses': _join(x.bus for x in under),
        'over_v_buses': _join(x.bus for x in over),
        'worst_violation_bus': worst.bus if worst else '',
        'worst_violation_pu': worst.deviation_pu if worst else 0.0,
        'v_min_pu': float(v_pu.min()),
        'v_min_bus': bus_of[int(v_pu.argmin())],
        'v_max_pu': float(v_pu.max()),
        'v_max_bus': bus_of[int(v_pu.argmax())],
        'unclassified_buses': _join(limits.unclassified(net.buses)),
        'tap_steps_total': int(sum(abs(s) for s in steps.values())),
        'tap_steps': _join(f'{name.split(".", 1)[-1]}:{s:+d}' for name, s in steps.items() if s),
        'control_passes': int(getattr(pf, 'control_passes', 0) or 0),
        'frozen_controllers': _join(getattr(pf, 'frozen_controllers', []) or []),
        'gens_at_qmax': _join(b for b, s in pv_state.items() if s == 1),
        'gens_at_qmin': _join(b for b, s in pv_state.items() if s == -1),
    }
    return {f'{prefix}{k}': v for k, v in audit.items()}
