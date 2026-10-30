"""
Analysis stage of the system-strength metrics study.

Every claim is an observation, a mechanism (the formula term responsible) and a
confirmation: that term is isolated, and the effect must disappear or be
accounted for. No network, power-flow or short-circuit calculation is performed.

Inputs: results/glover37/raw/ (one simulate run and its manifest). The core claims use
the sequential-commitment states (E1 intact, E2 outages); E5 (all generators on line,
intact) is analysed alongside. ESCR is read with its reference MIIF (config
REFERENCE_VARIANT); the MIIF check (E3) reads the probe variants of the same run. All
denominators are rated MW.

Common basis: SCMVA_i from the short-circuit solver, Z = Y_SCC^-1 from its builder:

    SCR_i   = SCMVA_i / P_i                          37-bus, 69 kV POI
    WSCR    = Σ SCMVA_i P_i / (Σ P_i)²               37-bus, 69 kV POI
    ESCR_i  = SCMVA_i / (P_i + Σ MIIF_ji P_j)        46-bus, 34.5 kV MV
    SDSCR_i = |V_i|² / (|Z_ii| |P_i + Σ w_ij P_j|)   46-bus, 0.69 kV LV

Isolating terms. Each metric m is recomputed from its recorded terms as

    m_noV   |V| = 1 pu: SCMVA replaced by 100 / |Z_ii|, coupling kept
    m_noC   coupling removed (MIIF = 0, w = 0), |V| kept
    m_Z     both removed: Z_ii only

so a change in ln m between two states (hours or scenarios) splits exactly into

    Δln m = (Δln m - Δln m_noV) + (Δln m_noV - Δln m_Z) + Δln m_Z
             |V| part              coupling part           Z_ii part

For SCR, WSCR and ESCR the |V| part is ln(SCMVA |Z_ii| / 100) = ln|V| plus the
solver's fault-impedance term (about 1e-5). SCR and WSCR have no coupling term
(m_noC = m, m_noV = m_Z).

Coupling load, the interaction a site carries relative to its own power:
    ESCR  k_i = Σ IF_ji P_j / P_i          SDSCR  k_i = |P_i + Σ w_ij P_j| / P_i - 1
An outage that lowers k raises the metric while the network weakens; whether that
explains a metric's "stronger" hours is tested by removing the coupling term (C7).

Tables (each carries `experiment` and `serves`, see TABLE_TAGS)
---------------------------------------------------------------
d1_weakest_hour     weakest hour, its attribution to Z_ii, |V|, coupling   (E1, E2; D1, RQ2)
d4_outage           per corridor and site: shares stronger, Δln parts, coupling load (E2; D4)
d4_hourly           r and the network-only r per hour, corridor and site   (E2; D4, F3)
d5_snapshots        minimum-load and worse-of-two snapshot gaps            (E1, E2; D5)
d5_dispatch         commitment against load scale                          (E1; D5)
e5_weakest_hour, e5_snapshots   the same with every generator on line      (E5; RQ2, D5)
t3_commitment       E1 against E5 per metric and site                      (E1, E5; RQ2, D5)
miif_variants, miif_hourly      ESCR under each MIIF variant               (E3; RQ3)
a3_denominators, parallel_circuits, reconstruction_check                  checks
summary.csv         the inputs, then one row per number the text and captions quote
                    (claims C1-C9, captions), with its source table

Scenarios: intact plus one outage per corridor; circuit 1 represents a corridor
of parallel circuits, whose other circuits are checked against it (parallel_circuits).
D4 counts corridors.

Usage
-----
    python -m experiments.glover37.analyze
    python -m experiments.glover37.analyze \
        --input-dir results/glover37/raw --output-dir results/glover37/analysis
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.glover37 import manifest
from experiments.glover37.config import (
    ANALYSIS_DIR, AS_RUN, EVAL_BUS, INTACT, METRICS, NETWORK, PROBE_VARIANTS, RAW_DIR,
    ALL_ONLINE, F3_FLOOR_PCT, MATERIAL_PCT, NEAR_MIN_PCT, RECONSTRUCTION_TOL, REFERENCE_VARIANT, REL_TIE, S_BASE_MVA, SEQUENTIAL, SITES,
)

DEFAULT_INPUT = RAW_DIR
DEFAULT_OUTPUT = ANALYSIS_DIR
INPUT_FILES = ("metrics_timeseries.csv", "metric_variables.csv", "run_summary.csv",
               "dispatch_schedule.csv", "contingency_set.csv")
# Traceability: the experiment(s) and RQ/dimension each output table serves. "by case" takes
# the experiment from the row's case (intact: E1, an outage: E2) under sequential commitment.
TABLE_TAGS = {
    "reconstruction_check": ("E1;E2", "check"),
    "d1_weakest_hour": ("by case", "RQ1-D1;RQ2"),
    "d4_outage": ("E2", "RQ1-D4"),
    "d4_hourly": ("E2", "RQ1-D4"),
    "d5_snapshots": ("by case", "RQ1-D5"),
    "d5_dispatch": ("E1", "RQ1-D5"),
    "a3_denominators": ("E1;E2", "check"),
    "parallel_circuits": ("E2", "check"),
    "miif_variants": ("E3", "RQ3;RQ1-D4"),
    "miif_hourly": ("E3", "RQ3;RQ1-D4"),
    "e5_weakest_hour": ("E5", "RQ2;RQ1-D1"),
    "e5_snapshots": ("E5", "RQ2;RQ1-D5"),
    "t3_commitment": ("E1;E5", "RQ2;RQ1-D5"),
}


def tag(name, frame):
    """The table with `experiment` and `serves` as its first columns."""
    experiment, serves = TABLE_TAGS[name]
    frame = frame.copy()
    if experiment == "by case":
        experiment = np.where(frame["case"] == INTACT, "E1", "E2")
    frame.insert(0, "serves", serves)
    frame.insert(0, "experiment", experiment)
    return frame


# ESCR terms written once per state, with the reference variant
SHARED_ESCR_TERMS = ("scmva", "rated_mw", "z_ii", "v_pf")

SITE_IDS = tuple(str(s.host_bus) for s in SITES)
GROUP = "group"
UNITS = tuple((m, s) for m in METRICS for s in ((GROUP,) if m == "WSCR" else SITE_IDS))
VARIANTS = ("m", "noV", "noC", "Z")

N_COMMITTED_COLUMN = {"base": "base_n_committed", "ibr": "ibr_case_n_committed"}
P_MW = {
    "rated": {str(s.host_bus): s.pmax_mw for s in SITES},
    "mva": {str(s.host_bus): s.converter_mva for s in SITES},
}


# =============================================================================
# 1. Inputs
# =============================================================================

def _bus_id(x):
    if pd.isna(x):
        return None
    try:
        return int(float(x))
    except (TypeError, ValueError):
        return str(x)


def _join(items):
    return ";".join(str(x) for x in items)


def read_run(input_dir):
    """The raw files of one simulate run, every commitment rule and ESCR variant."""
    d = Path(input_dir)
    if missing := [f for f in INPUT_FILES if not (d / f).is_file()]:
        raise FileNotFoundError(f"Missing inputs in {d}: {missing}")
    metrics = pd.read_csv(d / "metrics_timeseries.csv", dtype={"bus": str, "variant": str},
                          keep_default_na=False, na_values={"value": [""]})
    metrics["bus"] = metrics["bus"].map(_bus_id)
    variables = pd.read_csv(d / "metric_variables.csv",
                            dtype={"bus": str, "related_bus": str, "variant": str},
                            keep_default_na=False, na_values={c: [""] for c in ("real", "imag", "magnitude")})
    variables["bus"] = variables["bus"].map(_bus_id)
    variables["related_bus"] = variables["related_bus"].replace("", np.nan).map(_bus_id)
    return {
        "metrics": metrics, "variables": variables,
        "summary": pd.read_csv(d / "run_summary.csv"),
        "dispatch": pd.read_csv(d / "dispatch_schedule.csv"),
        "contingencies": pd.read_csv(d / "contingency_set.csv"),
        "sha256": {f: hashlib.sha256((d / f).read_bytes()).hexdigest() for f in INPUT_FILES},
        "manifest": manifest.read(d),
    }


def view(run, variant=REFERENCE_VARIANT, commitment=SEQUENTIAL):
    """The states of one commitment rule (sequential: E1, E2; all_online: E5) with ESCR
    read through one MIIF variant; the ESCR terms shared by every variant come from the
    reference rows."""
    def sequential(frame):
        return frame[frame["commitment"] == commitment].drop(columns="commitment")

    metrics = sequential(run["metrics"])
    metrics = metrics[metrics["variant"].isin(("", variant))].drop(columns="variant")
    variables = sequential(run["variables"])
    escr = variables["metric"] == "ESCR"
    own = variables["variant"] == variant
    shared = (variables["variant"] == REFERENCE_VARIANT) & variables["variable"].isin(SHARED_ESCR_TERMS)
    variables = variables[~escr | own & ~variables["variable"].isin(SHARED_ESCR_TERMS) | shared]
    variables = variables.drop(columns="variant")
    return {
        "metrics": metrics.reset_index(drop=True),
        "variables": dict(tuple(variables.groupby(["metric", "variable"]))),
        "summary": sequential(run["summary"]).reset_index(drop=True),
        "dispatch": sequential(run["dispatch"]).reset_index(drop=True),
        "contingencies": run["contingencies"],
        "sha256": run["sha256"],
        "variant": variant,
        "commitment": commitment,
    }


def scenarios(contingencies):
    """Representative case per corridor (circuit 1), and {case: representative}."""
    inc = contingencies[contingencies["included"].astype(bool)]
    representative = {}
    for _, grp in inc.groupby(inc.apply(lambda r: tuple(sorted((r.from_bus, r.to_bus))), axis=1),
                              sort=False):
        rep = next((r.case for r in grp.itertuples() if r.line.endswith("ckt 1")), grp["case"].iloc[0])
        representative.update({c: rep for c in grp["case"]})
    corridors = [c for c in inc["case"] if representative[c] == c]
    return corridors, representative


def check_inputs(data, cases=None):
    """Fail on anything incomplete; never treat a missing state as valid. `cases`: the
    expected cases (default: intact plus every included contingency)."""
    summary, metrics = data["summary"], data["metrics"]
    if cases is None:
        cases = [INTACT] + list(data["contingencies"].query("included")["case"])
    if set(summary["case"]) != set(cases):
        raise ValueError(f"run_summary cases differ from contingency_set: "
                         f"{sorted(set(cases) ^ set(summary['case']))}")
    hours = sorted(summary["hour_index"].unique())
    for case, s in summary.groupby("case"):
        if sorted(s["hour_index"]) != hours:
            raise ValueError(f"{case}: hours differ from the other cases or repeat")
    if not (summary["base_pf_ok"].astype(bool).all() and summary["ibr_pf_ok"].astype(bool).all()):
        raise ValueError("Unconverged power flow in run_summary.csv")
    if summary["error"].fillna("").astype(str).str.strip().ne("").any():
        raise ValueError("Errors recorded in run_summary.csv")
    if len(metrics) != len(hours) * len(cases) * len(UNITS):
        raise ValueError(f"metrics_timeseries.csv has {len(metrics)} rows, expected "
                         f"{len(hours) * len(cases) * len(UNITS)}")
    v = metrics["value"].to_numpy(float)
    if not np.isfinite(v).all() or (v <= 0).any():
        raise ValueError("Non-finite or non-positive metric values")
    for column in N_COMMITTED_COLUMN.values():
        if summary.groupby("hour_index")[column].nunique().max() > 1:
            raise ValueError(f"{column}: commitment differs between scenarios")
    return np.asarray(hours, dtype=int), cases


# =============================================================================
# 2. Metrics and their isolated-term variants
# =============================================================================

def _wide(groups, metric, variable, part="real", pair=False):
    rows = groups.get((metric, variable))
    if rows is None:
        raise ValueError(f"No {metric} {variable} rows in metric_variables.csv")
    if part == "complex":
        vals = rows["real"].to_numpy(float) + 1j * rows["imag"].to_numpy(float)
    else:
        vals = rows[part].to_numpy(float)
    keys = ["case", "hour_index", "bus"] + (["related_bus"] if pair else [])
    s = pd.Series(vals, index=pd.MultiIndex.from_frame(rows[keys]))
    if s.index.duplicated().any():
        raise ValueError(f"Duplicate {metric} {variable} rows")
    return s.unstack(["bus", "related_bus"] if pair else "bus").sort_index()


def _by_site(frame, metric):
    site_of = {b: s for s, b in EVAL_BUS[metric].items()}
    return frame.rename(columns=site_of)[list(SITE_IDS)]


def variant_values(groups, p_mw):
    """{variant: {metric: DataFrame(index (case, hour_index), columns site)}}, P per metric."""
    out = {v: {} for v in VARIANTS}

    p = pd.Series(p_mw["SCR"])[list(SITE_IDS)]
    scmva = _by_site(_wide(groups, "SCR", "scmva"), "SCR")
    z_abs = _by_site(_wide(groups, "SCR", "z_ii", "complex"), "SCR").abs()
    out["m"]["SCR"] = out["noC"]["SCR"] = scmva / p
    out["noV"]["SCR"] = out["Z"]["SCR"] = S_BASE_MVA / z_abs / p

    p = pd.Series(p_mw["WSCR"])[list(SITE_IDS)]
    scmva = _by_site(_wide(groups, "WSCR", "scmva"), "WSCR")
    z_abs = _by_site(_wide(groups, "WSCR", "z_ii", "complex"), "WSCR").abs()

    def wscr(sc):
        return (sc * p).sum(axis=1).to_frame(GROUP) / p.sum() ** 2
    out["m"]["WSCR"] = out["noC"]["WSCR"] = wscr(scmva)
    out["noV"]["WSCR"] = out["Z"]["WSCR"] = wscr(S_BASE_MVA / z_abs)

    p = p_mw["ESCR"]
    scmva = _by_site(_wide(groups, "ESCR", "scmva"), "ESCR")
    z_abs = _by_site(_wide(groups, "ESCR", "z_ii", "complex"), "ESCR").abs()
    interaction = _wide(groups, "ESCR", "if_magnitude", pair=True)  # columns (j, i): IF_ji
    bus = EVAL_BUS["ESCR"]
    den = pd.DataFrame({i: p[i] + sum(interaction[(bus[j], bus[i])] * p[j]
                                      for j in SITE_IDS if j != i) for i in SITE_IDS})
    p_own = pd.Series(p)[list(SITE_IDS)]
    out["m"]["ESCR"] = scmva / den
    out["noV"]["ESCR"] = S_BASE_MVA / z_abs / den
    out["noC"]["ESCR"] = scmva / p_own
    out["Z"]["ESCR"] = S_BASE_MVA / z_abs / p_own

    p = {s: mw / S_BASE_MVA for s, mw in p_mw["SDSCR"].items()}
    bus = EVAL_BUS["SDSCR"]
    v_abs = _by_site(_wide(groups, "SDSCR", "v_pf", "complex"), "SDSCR").abs()
    z_rr = _wide(groups, "SDSCR", "z_rr", "complex", pair=True)
    w = _wide(groups, "SDSCR", "w_complex", "complex", pair=True)  # columns (i, j): w_ij
    z_abs = pd.DataFrame({i: z_rr[(bus[i], bus[i])].abs() for i in SITE_IDS})
    eff = pd.DataFrame({i: (p[i] + sum(w[(bus[i], bus[j])] * p[j]
                                       for j in SITE_IDS if j != i)).abs() for i in SITE_IDS})
    p_own = pd.Series(p)[list(SITE_IDS)]
    out["m"]["SDSCR"] = v_abs ** 2 / (z_abs * eff)
    out["noV"]["SDSCR"] = 1 / (z_abs * eff)
    out["noC"]["SDSCR"] = v_abs ** 2 / (z_abs * p_own)
    out["Z"]["SDSCR"] = 1 / (z_abs * p_own)
    return out


def check_reconstruction(metrics, values):
    """The recomputed m must equal the recorded metric (guards the schema reading)."""
    rows = []
    for metric in METRICS:
        rec = metrics[metrics["metric"] == metric]
        site = (rec["bus"].map({b: s for s, b in EVAL_BUS[metric].items()})
                if metric != "WSCR" else pd.Series(GROUP, index=rec.index))
        recorded = pd.Series(rec["value"].to_numpy(float),
                             index=pd.MultiIndex.from_arrays([rec["case"], rec["hour_index"], site]))
        computed = values["m"][metric].stack()
        computed.index = computed.index.set_names(recorded.index.names)
        rel = (computed.reindex(recorded.index) / recorded - 1).abs()
        if rel.isna().any():
            raise ValueError(f"{metric}: recorded and recomputed rows do not match")
        rows.append({"metric": metric, "max_rel_diff": float(rel.max())})
    out = pd.DataFrame(rows)
    if (out["max_rel_diff"] > RECONSTRUCTION_TOL).any():
        raise ValueError(f"Recomputed metrics differ from the recorded values:\n{out}")
    return out


def series(values, variant, metric, site, case):
    col = GROUP if metric == "WSCR" else site
    return values[variant][metric][col].xs(case, level="case")


def coupling_load(values, metric, site, case):
    """Interaction load relative to the site's own power, over hours:
    ESCR  k = Σ IF_ji P_j / P_i;   SDSCR  k = |P_i + Σ w_ij P_j| / P_i - 1;   SCR, WSCR  0.
    (m_Z / m_noV = denominator / P_i, so k = m_Z / m_noV - 1.)"""
    return series(values, "Z", metric, site, case) / series(values, "noV", metric, site, case) - 1


def split(values, metric, site, frm, to):
    """Δln m from state frm to state to, split into |V|, coupling and Z_ii parts.
    A state is (case, hour_index)."""
    col = GROUP if metric == "WSCR" else site
    d = {v: float(np.log(values[v][metric].loc[to, col] / values[v][metric].loc[frm, col]))
         for v in ("m", "noV", "Z")}
    return {"dln_total": d["m"], "dln_V": d["m"] - d["noV"],
            "dln_coupling": d["noV"] - d["Z"], "dln_Z_ii": d["Z"]}


# =============================================================================
# 3. Context: load scale, commitment
# =============================================================================

def lowest(s, k):
    threshold = s.sort_values(kind="mergesort").iloc[k - 1]
    return set(s.index[s <= threshold + REL_TIE * abs(threshold)])


def t_star(s):
    return sorted(lowest(s, 1))


def sign0(x):
    x = np.asarray(x, dtype=float)
    return np.where(np.abs(x) <= REL_TIE, 0.0, np.sign(x))


def build_context(data, hours):
    s = data["summary"].set_index(["case", "hour_index"]).sort_index()
    intact = s.loc[INTACT]
    b_count, n_committed = {}, {}
    for network, column in N_COMMITTED_COLUMN.items():
        n = intact[column].astype(int)
        n_committed[network] = n
        b_count[network] = set(n.index[n == n.min()])
    return {"hours": hours, "load_scale": intact["load_scale"], "n_committed": n_committed,
            "b_count": b_count}


# =============================================================================
# 4. Claims
# =============================================================================

def _explain(in_b):
    """Why the weakest hour is (or is not) a minimum-commitment hour."""
    if not in_b["Z"]:
        return "commitment not confirmed (Z_ii-only weakest hour outside B_count)"
    if in_b["m"]:
        return "commitment"
    if in_b["noV"] and in_b["noC"]:
        return "|V| or coupling"
    if in_b["noV"]:
        return "|V|"
    if in_b["noC"]:
        return "coupling"
    return "|V| and coupling"


def d1_weakest_hour(values, ctx, cases):
    rows = []
    lam = ctx["load_scale"]
    for case in cases:
        for metric, site in UNITS:
            b = ctx["b_count"][NETWORK[metric]]
            t = {v: t_star(series(values, v, metric, site, case)) for v in VARIANTS}
            in_b = {v: bool(set(t[v]) & b) for v in VARIANTS}
            rows.append({
                "case": case, "metric": metric, "site": site, "network": NETWORK[metric],
                "B_count": _join(sorted(b)),
                "t_star": _join(t["m"]), "t_star_in_B_count": in_b["m"],
                "t_star_V_1pu": _join(t["noV"]), "t_star_coupling_removed": _join(t["noC"]),
                "t_star_Z_ii_only": _join(t["Z"]), "Z_ii_only_t_star_in_B_count": in_b["Z"],
                "t_star_equals_Z_ii_only": t["m"] == t["Z"],
                "explanation": _explain(in_b),
                "load_scale_at_t_star": float(lam.loc[t["m"][0]]),
                # from the Z_ii-only weakest hour to the actual weakest hour
                **split(values, metric, site, (case, t["Z"][0]), (case, t["m"][0])),
            })
    return pd.DataFrame(rows)


def d4_outage(values, ctx, corridors):
    rows = []
    for case in corridors:
        for metric, site in UNITS:
            ln = {v: np.log(series(values, v, metric, site, case)
                            / series(values, v, metric, site, INTACT)) for v in VARIANTS}
            part_v, part_c, part_z = ln["m"] - ln["noV"], ln["noV"] - ln["Z"], ln["Z"]
            # stronger: metric_contingency > metric_intact in that hour
            stronger = sign0(ln["m"]) > 0
            k_int, k_out = coupling_load(values, metric, site, INTACT), coupling_load(values, metric, site, case)
            r = np.expm1(ln["m"])
            rows.append({
                "case": case, "metric": metric, "site": site,
                "median_r": float(r.median()),
                "share_hours_stronger": float(stronger.mean()),
                "share_hours_stronger_V_1pu": float((sign0(ln["noV"]) > 0).mean()),
                "share_hours_stronger_coupling_removed": float((sign0(ln["noC"]) > 0).mean()),
                "share_hours_stronger_Z_ii_only": float((sign0(ln["Z"]) > 0).mean()),
                "max_r_Z_ii_only": float(np.expm1(ln["Z"]).max()),
                "median_dln_total": float(ln["m"].median()),
                "median_dln_V": float(part_v.median()),
                "median_dln_coupling": float(part_c.median()),
                "median_dln_Z_ii": float(part_z.median()),
                "median_coupling_load_intact": float(k_int.median()),
                "median_coupling_load_outage": float(k_out.median()),
                "n_hours_stronger": int(stronger.sum()),
                "median_r_stronger": float(r[stronger].median()) if stronger.any() else float("nan"),
                "max_r_stronger": float(r[stronger].max()) if stronger.any() else float("nan"),
                "median_dln_Z_ii_stronger": float(part_z[stronger].median()) if stronger.any() else float("nan"),
                "n_stronger_coupling_load_falls": int((stronger & (sign0(k_out - k_int) < 0)).sum()),
            })
    return pd.DataFrame(rows)


def d5_snapshots(values, ctx, cases):
    lam = ctx["load_scale"]
    peak, minimum = int(lam.idxmax()), int(lam.idxmin())
    rows = []
    for case in cases:
        for metric, site in UNITS:
            s = series(values, "m", metric, site, case)
            t = t_star(s)
            snap = peak if s.loc[peak] <= s.loc[minimum] else minimum
            rows.append({
                "case": case, "metric": metric, "site": site, "t_peak": peak, "t_minimum": minimum,
                "t_minimum_in_B_count": minimum in ctx["b_count"][NETWORK[metric]],
                "t_snapshot": snap, "t_star": _join(t), "snapshot_is_weekly_min": snap in t,
                "snapshot_value": float(s.loc[snap]), "weekly_min_value": float(s.min()),
                "g": float(s.loc[snap] / s.min() - 1),  # worse of peak and minimum load
                "g_minimum": float(s.loc[minimum] / s.min() - 1),
                # from the weekly minimum to the snapshot
                **split(values, metric, site, (case, t[0]), (case, snap)),
            })
    return pd.DataFrame(rows)


def d5_dispatch(ctx):
    """Sequential dispatch is monotone in λ: committed count never falls as λ rises."""
    lam = ctx["load_scale"].sort_values(kind="mergesort")
    rows = []
    for network, n in ctx["n_committed"].items():
        ordered = n.loc[lam.index].to_numpy()
        rows.append({"network": network, "n_hours": len(ordered),
                     "n_decreases_as_load_rises": int((np.diff(ordered) < 0).sum()),
                     "min_n_committed": int(n.min()),
                     "B_count": _join(sorted(ctx["b_count"][network]))})
    return pd.DataFrame(rows)


def t3_commitment(d1_e1, d5_e1, ctx_e1, d1_e5, d5_e5, ctx_e5):
    """T3 (draft): intact network, E1 (sequential commitment) against E5 (all units
    online), per metric and site: weakest hour, load scale and committed units there,
    and the snapshot gaps (minimum load; worse of peak and minimum load), in %."""
    rows = {}
    for tag, d1, d5, ctx in (("E1", d1_e1, d5_e1, ctx_e1), ("E5", d1_e5, d5_e5, ctx_e5)):
        d1 = d1[d1["case"] == INTACT].set_index(["metric", "site"])
        d5 = d5[d5["case"] == INTACT].set_index(["metric", "site"])
        for (metric, site), r in d1.iterrows():
            t = int(str(r["t_star"]).split(";")[0])
            rows.setdefault((metric, site), {"metric": metric, "site": site}).update({
                f"{tag}_t_star": t,
                f"{tag}_load_scale": float(ctx["load_scale"].loc[t]),
                f"{tag}_n_committed": int(ctx["n_committed"][NETWORK[metric]].loc[t]),
                f"{tag}_gap_minimum_pct": 100 * float(d5.loc[(metric, site), "g_minimum"]),
                f"{tag}_gap_worse_pct": 100 * float(d5.loc[(metric, site), "g"]),
            })
    return pd.DataFrame(list(rows.values()))


def d4_hourly(values, corridors):
    """Per hour, corridor and site: r = (metric_contingency - metric_intact) / metric_intact,
    and r_network, the same ratio with Z_ii only (|V| = 1 pu, coupling removed): the
    network's own change, -Δ|Z_ii| / |Z_ii'| in relative terms (F3)."""
    frames = []
    for case in corridors:
        for metric, site in UNITS:
            r = (series(values, "m", metric, site, case) / series(values, "m", metric, site, INTACT) - 1)
            r_net = series(values, "Z", metric, site, case) / series(values, "Z", metric, site, INTACT) - 1
            frames.append(pd.DataFrame({"case": case, "metric": metric, "site": site,
                                        "hour_index": r.index, "r": r.to_numpy(),
                                        "r_network": r_net.reindex(r.index).to_numpy()}))
    return pd.concat(frames, ignore_index=True)


def miif_hourly(runs, corridors):
    """ESCR's r per hour, corridor and site under each MIIF variant, with the network's own
    change r_network (the same for every variant: Z_ii does not depend on the MIIF) (F4)."""
    frames = []
    for variant, values in runs:
        for case in corridors:
            for site in SITE_IDS:
                r = series(values, "m", "ESCR", site, case) / series(values, "m", "ESCR", site, INTACT) - 1
                r_net = series(values, "Z", "ESCR", site, case) / series(values, "Z", "ESCR", site, INTACT) - 1
                frames.append(pd.DataFrame({"variant": variant, "case": case, "site": site,
                                            "hour_index": r.index, "r": r.to_numpy(),
                                            "r_network": r_net.reindex(r.index).to_numpy()}))
    return pd.concat(frames, ignore_index=True)


def core_tables(values, ctx, cases, corridors):
    return {
        "d1_weakest_hour": d1_weakest_hour(values, ctx, cases),
        "d4_outage": d4_outage(values, ctx, corridors),
        "d4_hourly": d4_hourly(values, corridors),
        "d5_snapshots": d5_snapshots(values, ctx, cases),
    }


def compare_tables(reference, other, rtol=1e-9):
    """A3: every column without 'value' in its name must be unchanged."""
    rows = []
    for name, ref in reference.items():
        alt = other[name]
        cols = [c for c in ref.columns if "value" not in c]
        differing = []
        for c in cols:
            x, y = ref[c].reset_index(drop=True), alt[c].reset_index(drop=True)
            if pd.api.types.is_float_dtype(x):
                same = np.allclose(x.to_numpy(float), y.to_numpy(float), rtol=rtol, atol=1e-12,
                                   equal_nan=True)
            else:
                same = x.astype(str).equals(y.astype(str))
            if not same:
                differing.append(c)
        rows.append({"table": name, "n_rows": len(ref), "n_columns_compared": len(cols),
                     "unchanged": not differing, "differing_columns": _join(differing)})
    return pd.DataFrame(rows)


def miif_variants(runs, cases, corridors):
    """E3: ESCR conclusions under each MIIF variant.
    runs: [(variant, values)], the first is the reference variant."""
    rows = []
    for variant, values in runs:
        for case in cases:
            for site in SITE_IDS:
                row = {"variant": variant, "case": case, "site": site,
                       "t_star": _join(t_star(series(values, "m", "ESCR", site, case)))}
                if case != INTACT:
                    ln = np.log(series(values, "m", "ESCR", site, case)
                                / series(values, "m", "ESCR", site, INTACT))
                    row.update({"share_hours_stronger": float((sign0(ln) > 0).mean()),
                                "median_r": float(np.expm1(ln).median())})
                rows.append(row)
    return pd.DataFrame(rows)


def parallel_circuits(values, representative):
    rows = []
    for case, rep in representative.items():
        if case == rep:
            continue
        for metric, site in UNITS:
            a = series(values, "m", metric, site, case)
            b = series(values, "m", metric, site, rep)
            rows.append({"case": case, "representative": rep, "metric": metric, "site": site,
                         "same_t_star": t_star(a) == t_star(b),
                         "max_abs_dln": float(np.log(a / b).abs().max())})
    return pd.DataFrame(rows)


# =============================================================================
# 5. Summary and driver
# =============================================================================

def _count(mask):
    mask = pd.Series(mask)
    return f"{int(mask.sum())}/{len(mask)}"


def claim_rows(out, ctx, ctx_e5, values, e5_values):
    """The numbers the text and captions quote, one row each: claim (C1-C9, captions), item,
    result and source table. report.md formats exactly these rows."""
    rows = []

    def add(claim, item, result, table):
        rows.append({"claim": claim, "item": item, "result": result, "table": table})

    def pct(x, digits=1):
        return f"{100 * float(x):.{digits}f} %"

    def week(vals, metric, site):
        return series(vals, "m", metric, site, INTACT)

    d1, d4, d4h = out["d1_weakest_hour"], out["d4_outage"], out["d4_hourly"]
    e5w, t3, d5d = out["e5_weakest_hour"], out["t3_commitment"], out["d5_dispatch"]
    mh, mv, par = out["miif_hourly"], out["miif_variants"], out["parallel_circuits"]
    lam = ctx["load_scale"]

    c = "C1 / D1 (E1): weakest hour under the unit-commitment rule"
    intact = d1[d1["case"] == INTACT]
    exc = d1[~d1["t_star_in_B_count"]]
    add(c, "weakest hour in the fewest-committed-units hours, intact",
        f"{int(intact['t_star_in_B_count'].sum())}/{len(intact)}", "d1_weakest_hour")
    add(c, "same, all cases (intact and outages)", f"{int(d1['t_star_in_B_count'].sum())}/{len(d1)}",
        "d1_weakest_hour")
    add(c, "confirmation: Z_ii-only weakest hour in those hours",
        f"{int(d1['Z_ii_only_t_star_in_B_count'].sum())}/{len(d1)}", "d1_weakest_hour")
    add(c, "exceptions and their explanation", f"{len(exc)}: " + (", ".join(
        f"{k} {v}" for k, v in exc["explanation"].value_counts().items()) or "none"), "d1_weakest_hour")
    add(c, "exceptions in the intact case", int((exc["case"] == INTACT).sum()), "d1_weakest_hour")
    add(c, "metrics with exceptions", ", ".join(sorted(exc["metric"].unique())) or "none", "d1_weakest_hour")
    r1 = [week(values, m, s).max() / week(values, m, s).min() - 1 for m, s in UNITS]
    add(c, "week range per metric-site (max / min - 1)", f"{pct(min(r1))} to {pct(max(r1))}", "metric values")

    c = "C2 / RQ2 (E5): every generator on line"
    add(c, "load scale at the weakest hour",
        f"{e5w['load_scale_at_t_star'].min():.2f} to {e5w['load_scale_at_t_star'].max():.2f}", "e5_weakest_hour")
    add(c, "Z_ii-only weakest hour", ", ".join(sorted(set(e5w["t_star_Z_ii_only"].astype(str)))),
        "e5_weakest_hour")
    add(c, "minimum-load hour", int(lam.idxmin()), "load scale")
    add(c, "actual weakest hour equals the Z_ii-only one", f"{int(e5w['t_star_equals_Z_ii_only'].sum())}/{len(e5w)}",
        "e5_weakest_hour")
    add(c, "Δln |V| part, Z_ii-only to actual weakest hour", f"{e5w['dln_V'].min():.3f} to {e5w['dln_V'].max():.3f}",
        "e5_weakest_hour")
    add(c, "Δln Z_ii part, same", f"{e5w['dln_Z_ii'].min():.3f} to {e5w['dln_Z_ii'].max():.3f}", "e5_weakest_hour")
    r5 = [week(e5_values, m, s).max() / week(e5_values, m, s).min() - 1 for m, s in UNITS]
    add(c, "week range per metric-site (max / min - 1)", f"{pct(min(r5))} to {pct(max(r5))}", "metric values")
    for metric, site in UNITS:
        if metric in ("SCR", "WSCR"):
            w = week(e5_values, metric, site)
            add(c, f"{metric} {site}: share of hours within {NEAR_MIN_PCT:g} % of the minimum",
                pct((w <= w.min() * (1 + NEAR_MIN_PCT / 100)).mean(), 0), "metric values")

    c = "C3 / D5: the minimum-load snapshot"
    add(c, "largest gap to the weekly minimum, E1", f"{t3['E1_gap_minimum_pct'].max():.2f} %", "t3_commitment")
    add(c, "largest gap, E5 minimum load", f"{t3['E5_gap_minimum_pct'].max():.2f} %", "t3_commitment")
    add(c, "largest gap, E5 worse of minimum and peak load", f"{t3['E5_gap_worse_pct'].max():.2f} %", "t3_commitment")
    for r in d5d.itertuples():
        add(c, f"committed units fall as load rises [{r.network}]", f"{r.n_decreases_as_load_rises} times",
            "d5_dispatch")

    c = "C4 / RQ1: agreement between metrics"
    add(c, "E1 weakest hours", ", ".join(sorted(set(t3["E1_t_star"].astype(str)))), "t3_commitment")
    add(c, "E5 weakest hours", ", ".join(sorted(set(t3["E5_t_star"].astype(str)))), "t3_commitment")
    add(c, "E5 weakest hours span", f"{int(t3['E5_t_star'].max() - t3['E5_t_star'].min())} h", "t3_commitment")

    c = "C5 / D4(a): what an outage does to the network"
    add(c, "network-only change, most strengthening", pct(d4h["r_network"].max(), 2), "d4_hourly")
    low = d4h.loc[d4h["r_network"].idxmin()]
    add(c, "network-only change, most weakening", pct(low["r_network"], 0), "d4_hourly")
    add(c, "most weakening at", f"{low['case']} / {low['metric']} / site {low['site']} / hour {int(low['hour_index'])}",
        "d4_hourly")

    c = "C6 / D4(b): SCR and WSCR follow the network"
    c7 = "C7 / D4(c): the coupling metrics occasionally oppose it"
    for metric in METRICS:
        g = d4h[d4h["metric"] == metric]
        claim = c if metric in ("SCR", "WSCR") else c7
        add(claim, f"{metric}: correlation of metric and network change",
            f"{np.corrcoef(g['r_network'], g['r'])[0, 1]:.3f}", "d4_hourly")
        add(claim, f"{metric}: share of points > {MATERIAL_PCT:g} % stronger",
            pct((g["r"] > MATERIAL_PCT / 100).mean()), "d4_hourly")
        add(claim, f"{metric}: points > {MATERIAL_PCT:g} % stronger while the network weakens",
            int(((g["r_network"] < 0) & (g["r"] > MATERIAL_PCT / 100)).sum()), "d4_hourly")
        add(claim, f"{metric}: largest change toward stronger", pct(g["r"].max(), 2), "d4_hourly")
    for metric in ("SDSCR", "ESCR"):
        part = d4[d4["metric"] == metric]
        add(c7, f"{metric}: stronger hours where the coupling load falls",
            f"{int(part['n_stronger_coupling_load_falls'].sum())}/{int(part['n_hours_stronger'].sum())}", "d4_outage")
        for label, col in (("as computed", "share_hours_stronger"), ("|V| = 1 pu", "share_hours_stronger_V_1pu"),
                           ("coupling removed", "share_hours_stronger_coupling_removed"),
                           ("Z_ii only", "share_hours_stronger_Z_ii_only")):
            add(c7, f"{metric}: share of hours stronger, {label}", pct(part[col].mean()), "d4_outage")

    c = "C8 / RQ3 (E3): ESCR's interaction factor"
    ref = mv[mv["variant"] == REFERENCE_VARIANT].set_index(["case", "site"])["t_star"]
    for variant, g in mh.groupby("variant", sort=False):
        add(c, f"{variant}: share of points > {MATERIAL_PCT:g} % stronger", pct((g["r"] > MATERIAL_PCT / 100).mean()),
            "miif_hourly")
        add(c, f"{variant}: largest change toward stronger", pct(g["r"].max(), 1), "miif_hourly")
        if variant != REFERENCE_VARIANT:
            agree = mv[mv["variant"] == variant].set_index(["case", "site"])["t_star"] == ref
            add(c, f"{variant}: weakest hour as under {REFERENCE_VARIANT}", f"{int(agree.sum())}/{len(agree)}",
                "miif_variants")

    c = "C9 / checks"
    add(c, "parallel circuits: same weakest hour as circuit 1", f"{int(par['same_t_star'].sum())}/{len(par)}",
        "parallel_circuits")
    add(c, "parallel circuits: max |Δln| against circuit 1", f"{par['max_abs_dln'].max():.4f}", "parallel_circuits")
    add(c, "denominators (rated, MVA) change no conclusion", bool(out["a3_denominators"]["unchanged"].all()),
        "a3_denominators")

    c = "captions"
    for r in d5d.itertuples():
        add(c, f"F2: fewest committed units under the rule [{r.network}]",
            f"{r.min_n_committed} units, hours {r.B_count}", "d5_dispatch")
    add(c, "F2: committed units with every generator on line (E5)",
        ", ".join(str(n) for n in sorted(set(t3["E5_n_committed"]))), "t3_commitment")
    add(c, "F2a: minimum and peak load (load scale, hour)",
        f"{lam.min():.2f} at {int(lam.idxmin())}; {lam.max():.2f} at {int(lam.idxmax())}", "load scale")
    add(c, "F3: display window", f"down to {F3_FLOOR_PCT:g} %", "config")
    for metric in METRICS:
        g = d4h[d4h["metric"] == metric]
        beyond = (100 * g["r_network"] < F3_FLOOR_PCT) | (100 * g["r"] < F3_FLOOR_PCT)
        add(c, f"F3: {metric} points beyond the window (of all, lowest value)",
            f"{int(beyond.sum())}/{len(g)}, {pct(min(g['r'].min(), g['r_network'].min()), 0)}", "d4_hourly")
    return pd.DataFrame(rows)


def build_summary(data, ctx, cases):
    """The run's inputs: hours, scenarios, input SHA-256, minimum-commitment hours. The claims
    and captions follow as claim_rows."""
    rows = []

    def add(claim, item, result, table=""):
        rows.append({"claim": claim, "item": item, "result": result, "table": table})

    add("inputs", "hours per scenario", len(ctx["hours"]))
    add("inputs", "scenarios (intact + one per corridor)", len(cases))
    for f, h in data["sha256"].items():
        add("inputs", f"SHA-256 {f}", h)
    for network, b in ctx["b_count"].items():
        add("inputs", f"minimum-commitment hours B_count [{network}]", _join(sorted(b)))
    return pd.DataFrame(rows)


def run(input_dir=DEFAULT_INPUT, output_dir=DEFAULT_OUTPUT):
    started = manifest.now()
    input_dir, output_dir = Path(input_dir).resolve(), Path(output_dir).resolve()
    if input_dir == output_dir:
        raise ValueError("Output directory must differ from the input directory")
    raw = read_run(input_dir)
    raw_manifest = raw["manifest"]
    if not raw_manifest or raw_manifest.get("stage") != "simulate":
        raise ValueError(f"{input_dir} has no simulate manifest; run the simulate stage")
    if not raw_manifest.get("git_commit"):
        print(f"warning: raw run {raw_manifest['run_id']} names no commit (not a git checkout)")
    elif not raw_manifest.get("of_record"):
        print(f"warning: raw run {raw_manifest['run_id']} is not of record (uncommitted changes)")
    data = view(raw)
    hours, _ = check_inputs(data)
    corridors, representative = scenarios(data["contingencies"])
    cases = [INTACT] + corridors
    ctx = build_context(data, hours)

    values = variant_values(data["variables"], {m: P_MW[AS_RUN[m]] for m in METRICS})
    out = {"reconstruction_check": check_reconstruction(data["metrics"], values)}
    core = core_tables(values, ctx, cases, corridors)
    out.update(core)
    out["d5_dispatch"] = d5_dispatch(ctx)
    out["a3_denominators"] = pd.concat([
        compare_tables(core, core_tables(variant_values(data["variables"], {m: P_MW[kind] for m in METRICS}),
                                         ctx, cases, corridors)).assign(denominator=kind)
        for kind in ("rated", "mva")
    ], ignore_index=True)
    out["parallel_circuits"] = parallel_circuits(values, representative)
    runs = [(REFERENCE_VARIANT, values)]
    for variant in PROBE_VARIANTS:
        alt = view(raw, variant)
        check_inputs(alt)
        alt_values = variant_values(alt["variables"], {m: P_MW[AS_RUN[m]] for m in METRICS})
        check_reconstruction(alt["metrics"], alt_values)
        runs.append((variant, alt_values))
    out["miif_variants"] = miif_variants(runs, cases, corridors)
    out["miif_hourly"] = miif_hourly(runs, corridors)

    # E5: the intact network with every unit online (RQ2, D5); no outages
    e5 = view(raw, REFERENCE_VARIANT, ALL_ONLINE)
    e5_hours, _ = check_inputs(e5, cases=[INTACT])
    ctx_e5 = build_context(e5, e5_hours)
    e5_values = variant_values(e5["variables"], {m: P_MW[AS_RUN[m]] for m in METRICS})
    check_reconstruction(e5["metrics"], e5_values)
    weakest = d1_weakest_hour(e5_values, ctx_e5, [INTACT])
    out["e5_weakest_hour"] = weakest.drop(columns=[  # every hour has every unit online
        "B_count", "t_star_in_B_count", "Z_ii_only_t_star_in_B_count", "explanation"])
    out["e5_snapshots"] = d5_snapshots(e5_values, ctx_e5, [INTACT]).drop(columns="t_minimum_in_B_count")
    out["t3_commitment"] = t3_commitment(out["d1_weakest_hour"], out["d5_snapshots"], ctx,
                                         weakest, out["e5_snapshots"], ctx_e5)
    out["summary"] = pd.concat([build_summary(data, ctx, cases),
                                claim_rows(out, ctx, ctx_e5, values, e5_values)], ignore_index=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in out.items():
        (frame if name == "summary" else tag(name, frame)).to_csv(output_dir / f"{name}.csv", index=False)
    manifest.write(output_dir, "analyze", started=started,
                   inputs=[input_dir / f for f in INPUT_FILES],
                   outputs=[output_dir / f"{name}.csv" for name in out],
                   extra={"raw_run_id": raw_manifest.get("run_id"),
                          "raw_of_record": raw_manifest.get("of_record")})
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    out = run(args.input_dir, args.output_dir)
    with pd.option_context("display.max_rows", None, "display.max_colwidth", 100,
                           "display.width", 200):
        print(out["summary"].drop(columns="table").to_string(index=False))
    print(f"\nSaved {len(out)} tables to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
