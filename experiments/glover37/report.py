"""
Report stage: the paper's floats and the generated report of the system-strength study.

    F1a  network, with each host bus's plant converter          (case, AXD geometry)
    F1b  one plant's connection and each metric's evaluation point
    F2a  load scale over the intact week, minimum and peak load  (D1, RQ2, D5)
    F2b  where each metric is lowest, E1 against E5              (D1, RQ2, D5)
    F3   metric change against the network's change, per outage (D4)
    F4   ESCR's outage change, impedance against voltage-step interaction factor (C8, E3)
    T1   metric conventions, generated from config.py
    report.md  the claims C1-C9 with their numbers, computed from the analysis tables

Usage
-----
    python -m experiments.glover37.report
    python -m experiments.glover37.report --png

Inputs:  results/glover37/raw/ and results/glover37/analysis/
Outputs: figures/glover37/{f1a,f1b,f2a,f2b,f3,f4}*.pdf, t1_conventions.{tex,csv};
         results/glover37/report/report.md and manifest.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.glover37 import layout, manifest
from experiments.glover37.config import (
    ALL_ONLINE, ANALYSIS_DIR, AS_RUN, AUDITED_PROBE, F3_FLOOR_PCT, EVAL_BUS, EVAL_POINT, FIGURES_DIR, IF_STEPS,
    INTACT, MATERIAL_PCT, METRIC_DEFINITIONS, METRICS, NETWORK, RAW_DIR, REFERENCE_VARIANT, RESULTS_ROOT,
    SEQUENTIAL, SITES, VALIDATION_DIR, probe_variant,
)

REPORT_DIR = RESULTS_ROOT / "report"
IF_STEP_OF = {probe_variant(s): s for s in IF_STEPS}  # probe variant name -> relative voltage step


SITE_IDS = tuple(str(s.host_bus) for s in SITES)
GROUP = "group"


# F2 style, after the explorer's minimum view (tools/explorer.html)
INK, FAINT, RULE, GRID_F2 = "#2b2f36", "#767b83", "#e4e1db", "#eeebe6"
METRIC_COLOR = {"SCR": "#4a6382", "WSCR": "#9a9ea5", "ESCR": "#b08a52", "SDSCR": "#4f8a8b"}
SITE_COLOR = {"21": "#9a7459", "48": "#5f8270", "16": "#75719a"}
WINE = "#7b2d3b"  # F2a: the two planning snapshots (minimum load, peak load)
F2_METRICS = ("SCR", "WSCR", "ESCR", "SDSCR")
BAND = (74 / 255, 99 / 255, 130 / 255, 0.10)  # the explorer's --band: minimum-commitment hours


def week_data(raw_dir=RAW_DIR):
    """Intact-network series for F2: {(commitment, metric, site): values over hours},
    the load scale, the hour timestamps and, per network, the E1 hours with the fewest
    committed units."""
    m = pd.read_csv(raw_dir / "metrics_timeseries.csv", dtype={"bus": str, "variant": str},
                    keep_default_na=False)
    m = m[(m["case"] == INTACT) & m["variant"].isin(("", REFERENCE_VARIANT))]
    m["bus"] = m["bus"].str.replace(r"\.0$", "", regex=True)
    site_of = {str(b): s for metric in EVAL_BUS for s, b in EVAL_BUS[metric].items()}
    series = {}
    for (commitment, metric, bus), g in m.groupby(["commitment", "metric", "bus"]):
        site = GROUP if bus == GROUP else site_of[bus]
        series[commitment, metric, site] = g.set_index("hour_index")["value"].sort_index()
    s = pd.read_csv(raw_dir / "run_summary.csv")
    s = s[(s["case"] == INTACT) & (s["commitment"] == SEQUENTIAL)].set_index("hour_index").sort_index()
    min_hours = {network: sorted(s.index[s[column] == s[column].min()])
                 for network, column in (("base", "base_n_committed"), ("ibr", "ibr_case_n_committed"))}
    return series, s["load_scale"], pd.to_datetime(s["timestamp"]), min_hours


def _runs(hours):
    """Contiguous runs [(first, last)] of a sorted list of hours."""
    runs = []
    for h in hours:
        if runs and h == runs[-1][1] + 1:
            runs[-1][1] = h
        else:
            runs.append([h, h])
    return runs


def _week_axis(ax, hours, stamps, labels=True, hour_ticks=False):
    """Day gridlines and labels, recessive frame, as in the explorer. hour_ticks: ticks every
    24 h labelled with the hour index (the text's hour numbers) instead of the dates."""
    ax.set_xlim(hours[0] - 3, hours[-1] + 3)
    if hour_ticks:
        ticks = list(range(int(hours[0]), int(hours[-1]) + 2, 24))
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks] if labels else [])
    else:
        days = [h for h in hours if stamps.loc[h].hour == 0]
        ax.set_xticks(days)
        ax.set_xticklabels([stamps.loc[h].strftime("%a %d") for h in days] if labels else [])
    ax.grid(axis="x", color=GRID_F2, linewidth=0.7)
    ax.tick_params(axis="x", length=0, labelsize=7, colors=FAINT, pad=3)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(RULE)
    ax.set_axisbelow(True)


def plot_load(raw_dir=RAW_DIR):
    """F2a: the load scale over the intact week (public SE3 demand / its weekly mean)."""
    _, lam, stamps, _ = week_data(raw_dir)
    hours = lam.index.to_numpy()
    fig, ax = plt.subplots(figsize=(7.14, 1.25))
    ax.plot(hours, lam.to_numpy(), color=INK, linewidth=1.2)
    for t, label, offset, ha in ((int(lam.idxmin()), "minimum load", (7, 0), "left"),
                                 (int(lam.idxmax()), "peak load", (0, 9), "right")):
        ax.scatter([t], [lam.loc[t]], s=30, facecolors="none", edgecolors=WINE, linewidths=0.9, zorder=3)
        ax.annotate(f"{label}, λ = {lam.loc[t]:.2f}", (t, lam.loc[t]), xytext=offset,
                    textcoords="offset points", ha=ha, va="center", fontsize=7, color=WINE)
    _week_axis(ax, hours, stamps)
    ax.set_ylim(lam.min() - 0.05, lam.max() + 0.12)  # headroom for the peak label
    ax.grid(axis="y", color=GRID_F2, linewidth=0.7)
    ax.tick_params(axis="y", length=0, labelsize=7, colors=FAINT)
    ax.set_ylabel("load scale λ", fontsize=7.5, color=FAINT)
    return fig


def plot_minima(raw_dir=RAW_DIR):
    """F2b: where each metric is lowest over the intact week. Dot: the hour of the weekly
    minimum, filled for E1 (sequential commitment), open for E5 (every unit on line).
    Shaded: the E1 hours with the fewest committed units, per row's network."""
    series, lam, stamps, min_hours = week_data(raw_dir)
    hours = lam.index.to_numpy()
    fig, axes = plt.subplots(len(SITE_IDS), 1, figsize=(7.14, 4.1), sharex=True,
                             gridspec_kw={"hspace": 0.55})
    for ax, site in zip(axes, SITE_IDS):
        for y, metric in enumerate(F2_METRICS):
            key = (metric, GROUP if metric == "WSCR" else site)
            color = METRIC_COLOR[metric]
            for first, last in _runs(min_hours[NETWORK[metric]]):
                ax.fill_between([first - 0.5, last + 0.5], y - 0.36, y + 0.36, color=BAND,
                                linewidth=0, zorder=1)
            e1 = series[(SEQUENTIAL, *key)]
            ax.scatter([e1.idxmin()], [y], s=34, color=color, edgecolors=color, linewidths=1.2, zorder=3)
            e5 = series[(ALL_ONLINE, *key)]
            ax.scatter([e5.idxmin()], [y], s=34, facecolors="white", edgecolors=color,
                       linewidths=1.3, zorder=3)
        ax.set_ylim(len(F2_METRICS) - 0.4, -0.6)
        ax.set_yticks(range(len(F2_METRICS)))
        ax.set_yticklabels(F2_METRICS, fontsize=7, color=FAINT, family="monospace")
        ax.tick_params(axis="y", length=0, pad=6)
        ax.grid(axis="y", color=GRID_F2, linewidth=0.7)
        _week_axis(ax, hours, stamps, labels=True)
        ax.annotate("●", (0, 1), xycoords="axes fraction", xytext=(-2, 4), textcoords="offset points",
                    fontsize=9, color=SITE_COLOR[site], va="bottom", ha="left")
        ax.annotate(f"site {site}", (0, 1), xycoords="axes fraction", xytext=(10, 4),
                    textcoords="offset points", fontsize=8.5, color=INK, fontweight="bold",
                    va="bottom", ha="left")
        ax.tick_params(axis="x", labelbottom=True)
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=5, color=FAINT,
                          label="unit commitment rule (E1)"),
               plt.Line2D([], [], marker="o", linestyle="", markersize=5, markerfacecolor="white",
                          markeredgecolor=FAINT, markeredgewidth=1.2, label="all generators on line (E5)"),
               plt.Rectangle((0, 0), 1, 1, color=BAND, label="fewest committed units (E1)")]
    axes[0].legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 1.0), ncol=3,
                   frameon=False, fontsize=7, labelcolor=INK, handletextpad=0.4, columnspacing=1.2,
                   borderaxespad=0.2)
    return fig


COLUMN_IN = 3.49  # IEEE one-column width
SITE_OFFSET = {"21": -0.22, "48": 0.0, "16": 0.22}  # F2: sites side by side within a metric row


def plot_week(raw_dir=RAW_DIR, width_in=COLUMN_IN):
    """F2, one column: a) the load scale over the intact week, minimum and peak load marked;
    b) the hour of each metric's weekly minimum per site, filled for E1 (largest-first
    commitment), open for E5 (every unit on line); shaded: the E1 hours with the fewest
    committed units in the row's network."""
    series, lam, stamps, min_hours = week_data(raw_dir)
    hours = lam.index.to_numpy()
    fig, (ax_l, ax_m) = plt.subplots(2, 1, figsize=(width_in, 2.75), sharex=True,
                                     gridspec_kw={"height_ratios": (1, 1.7), "hspace": 0.18})
    ax_l.plot(hours, lam.to_numpy(), color=INK, linewidth=1.0)
    for t, ha, va, offset in ((int(lam.idxmin()), "left", "center", (5, 0)),
                              (int(lam.idxmax()), "right", "bottom", (0, 4))):
        ax_l.scatter([t], [lam.loc[t]], s=18, facecolors="none", edgecolors=WINE, linewidths=0.8, zorder=3)
        ax_l.annotate(f"{lam.loc[t]:.2f}", (t, lam.loc[t]), xytext=offset, textcoords="offset points",
                      ha=ha, va=va, fontsize=6.5, color=WINE)
    ax_l.set_ylim(lam.min() - 0.08, lam.max() + 0.16)  # headroom for the peak label
    ax_l.grid(axis="y", color=GRID_F2, linewidth=0.6)
    ax_l.tick_params(axis="y", length=0, labelsize=6.5, colors=FAINT)
    ax_l.set_ylabel("load scale", fontsize=6.5, color=FAINT)
    _week_axis(ax_l, hours, stamps, labels=False, hour_ticks=True)

    for y, metric in enumerate(F2_METRICS):
        for first, last in _runs(min_hours[NETWORK[metric]]):
            ax_m.fill_between([first - 0.5, last + 0.5], y - 0.42, y + 0.42, color=BAND, linewidth=0, zorder=1)
        for site in ((GROUP,) if metric == "WSCR" else SITE_IDS):
            color = INK if site == GROUP else SITE_COLOR[site]
            dy = 0.0 if site == GROUP else SITE_OFFSET[site]
            e1 = series[(SEQUENTIAL, metric, site)]
            e5 = series[(ALL_ONLINE, metric, site)]
            ax_m.scatter([e1.idxmin()], [y + dy], s=14, color=color, linewidths=0, zorder=3)
            ax_m.scatter([e5.idxmin()], [y + dy], s=14, facecolors="white", edgecolors=color,
                         linewidths=0.9, zorder=3)
    ax_m.set_ylim(len(F2_METRICS) - 0.5, -0.5)
    ax_m.set_yticks(range(len(F2_METRICS)))
    ax_m.set_yticklabels(F2_METRICS, fontsize=6.5, color=FAINT)
    ax_m.tick_params(axis="y", length=0, pad=3)
    for y in range(len(F2_METRICS) - 1):  # hairlines between metric rows, not through the markers
        ax_m.axhline(y + 0.5, color=RULE, linewidth=0.6, zorder=0)
    _week_axis(ax_m, hours, stamps, hour_ticks=True)
    ax_m.tick_params(axis="x", labelsize=6.5)
    ax_m.set_xlabel("hour", fontsize=6.5, color=FAINT, labelpad=2)
    for ax, tag in ((ax_l, "a)"), (ax_m, "b)")):
        ax.annotate(tag, (0, 1), xycoords="axes fraction", xytext=(-24, 2), textcoords="offset points",
                    fontsize=7.5, fontweight="bold", color=INK, va="bottom")
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=3.5, color=SITE_COLOR[s],
                          label=f"site {s}") for s in SITE_IDS]
    handles += [plt.Line2D([], [], marker="o", linestyle="", markersize=3.5, color=INK, label="group"),
                plt.Line2D([], [], marker="o", linestyle="", markersize=3.5, color=FAINT, label="largest-first"),
                plt.Line2D([], [], marker="o", linestyle="", markersize=3.5, markerfacecolor="white",
                           markeredgecolor=FAINT, label="all online"),
                plt.Rectangle((0, 0), 1, 1, color=BAND, label="fewest units")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.13), ncol=4, frameon=False,
               fontsize=6, labelcolor=INK, handletextpad=0.2, columnspacing=0.8)
    return fig


# Where each plant's converter symbol hangs off its host bus: (point on the bus, direction)
CONVERTER_PLACEMENT = {21: ((79.5, 36.0), "up", 2.5), 48: ((66.0, 21.8), "down", 6.5),
                       16: ((11.0, 19.6), "left", 2.5)}


def draw_converter(ax, anchor, direction, color, size=3.6, stem=2.2, lw=0.9):
    """DC/AC converter symbol (square, diagonal, DC lines top-left, sine bottom-right) on a
    stem from `anchor` in `direction`."""
    dx, dy = {"up": (0, 1), "down": (0, -1), "left": (-1, 0), "right": (1, 0)}[direction]
    x0, y0 = anchor
    cx, cy = x0 + dx * (stem + size / 2), y0 + dy * (stem + size / 2)
    ax.plot([x0, x0 + dx * stem], [y0, y0 + dy * stem], color=color, linewidth=lw, zorder=12)
    h = size / 2
    ax.add_patch(plt.Rectangle((cx - h, cy - h), size, size, facecolor="white", edgecolor=color,
                               linewidth=lw, zorder=12))
    ax.plot([cx - h, cx + h], [cy - h, cy + h], color=color, linewidth=lw * 0.8, zorder=13)
    ax.plot([cx - 0.8 * h, cx - 0.2 * h], [cy + 0.65 * h] * 2, color=color, linewidth=lw * 0.8, zorder=13)
    ax.plot([cx - 0.8 * h, cx - 0.2 * h], [cy + 0.45 * h] * 2, color=color, linewidth=lw * 0.8,
            linestyle=(0, (1.5, 1)), zorder=13)
    t = np.linspace(0, 2 * np.pi, 40)
    ax.plot(cx + 0.2 * h + 0.6 * h * t / (2 * np.pi), cy - 0.55 * h + 0.13 * h * np.sin(t),
            color=color, linewidth=lw * 0.8, zorder=13)


def plot_network(width_in=COLUMN_IN):
    """F1a: the Glover 37-bus network; each host bus (21, 48, 16) carries its plant's
    converter, in the site's colour."""
    fig, ax = plt.subplots()
    layout.draw_network(ax, bus_colors={s.host_bus: SITE_COLOR[str(s.host_bus)] for s in SITES})
    for site in SITES:
        anchor, direction, stem = CONVERTER_PLACEMENT[site.host_bus]
        draw_converter(ax, anchor, direction, SITE_COLOR[str(site.host_bus)], size=4.6, stem=stem, lw=1.0)
    layout.fit_axes_to_drawn_network(fig, ax, target_width_in=width_in)
    return fig


def plot_plant_chain(width_in=COLUMN_IN):
    """F1b: one plant's connection, and where each metric is evaluated."""
    fig, ax = plt.subplots(figsize=(width_in, 1.2))
    nodes = [(0.0, "POI", "69 kV", ("SCR", "WSCR")), (1.6, "collector", "34.5 kV", ()),
             (3.2, "MV bus", "34.5 kV", ("ESCR",)), (4.8, "terminal", "0.69 kV", ("SDSCR",))]
    links = [(0.0, 1.6, "transformer"), (1.6, 3.2, "feeder"), (3.2, 4.8, "transformer")]
    for x0, x1, kind in links:
        ax.plot([x0, x1], [0, 0], color=INK, linewidth=0.9, zorder=1)
        mid = (x0 + x1) / 2
        if kind == "transformer":
            for dx in (-0.09, 0.09):
                ax.add_patch(plt.Circle((mid + dx, 0), 0.14, fill=False, edgecolor=INK, linewidth=0.9, zorder=2))
            ax.add_patch(plt.Circle((mid, 0), 0.0001, color="white", zorder=1))
        ax.text(mid, -0.32, kind, fontsize=6.5, color=FAINT, ha="center", va="top")
    ax.plot([-0.9, 0.0], [0, 0], color=INK, linewidth=0.9, linestyle=(0, (2, 1.5)), zorder=1)
    ax.text(-0.95, 0, "network", fontsize=6.5, color=FAINT, ha="right", va="center")
    ax.plot([4.8, 5.5], [0, 0], color=INK, linewidth=0.9, zorder=1)
    ax.add_patch(plt.Rectangle((5.5, -0.2), 0.5, 0.4, fill=False, edgecolor=INK, linewidth=0.9, zorder=2))
    ax.text(5.75, 0, "~", fontsize=9, color=INK, ha="center", va="center")
    ax.text(5.75, -0.32, "IBR", fontsize=6.5, color=FAINT, ha="center", va="top")
    for x, name, kv, metrics in nodes:
        ax.plot([x, x], [-0.22, 0.22], color=INK, linewidth=2.2, solid_capstyle="butt", zorder=3)
        ax.text(x, 0.34, f"{name}\n{kv}", fontsize=6.5, color=INK, ha="center", va="bottom", linespacing=1.1)
        for k, metric in enumerate(metrics):
            ax.text(x, -0.62 - 0.3 * k, metric, fontsize=7, fontweight="bold",
                    color=METRIC_COLOR[metric], ha="center", va="top")
    ax.set_xlim(-1.7, 6.1)
    ax.set_ylim(-1.25, 0.85)
    ax.set_aspect("equal")
    ax.set_axis_off()
    return fig


NETWORK_NAME = {"base": "no IBR", "ibr": "with IBR"}


def table_conventions():
    """T1: what each metric measures, where and with what, from config (cannot drift).
    Returns (DataFrame, LaTeX)."""
    rows = []
    for metric in METRICS:
        point, kv = EVAL_POINT[metric]
        d = METRIC_DEFINITIONS[metric]
        rows.append({"metric": metric, "network": NETWORK_NAME[NETWORK[metric]],
                     "evaluated_at": point, "kV": kv, "definition": d["formula"],
                     "interaction": d["interaction"], "denominator": f"{AS_RUN[metric]} MW",
                     "source": d["source"], "cite": d["cite"]})
    t = pd.DataFrame(rows)
    body = "\n".join(
        # strut with height and depth: room around each fraction, so only these rows grow;
        # * marks the points evaluated on the network without the IBR plants (see note)
        f"\\rule[-2.6ex]{{0pt}}{{6.8ex}}{r.metric} & {r.definition} & "
        f"{r.evaluated_at}{'$^*$' if r.network == NETWORK_NAME['base'] else ''} & \\cite{{{r.cite}}} \\\\"
        for r in t.itertuples())
    # interaction factors and conventions not visible in the formulas, as a note under the rows
    note = ("$^*$On the network without the IBR plants; ESCR and SDSCR on the network with them. "
            + "; ".join(d["interaction"] for d in METRIC_DEFINITIONS.values() if d["interaction"].startswith("$"))
            + "; $Q_f = 0$ in ESCR; $P$: rated active power.")
    # needs \\usepackage{booktabs} in the paper
    latex = ("\\setlength{\\tabcolsep}{3pt}\n"
             "\\begin{tabular}{@{}lllc@{}}\n\\toprule\n"
             "Metric & Definition & Point & Ref. \\\\\n\\midrule\n"
             f"{body}\n\\midrule\n"
             f"\\multicolumn{{4}}{{@{{}}p{{\\linewidth}}@{{}}}}{{\\scriptsize {note}}} \\\\\n"
             "\\bottomrule\n\\end{tabular}\n")
    return t, latex


OPPOSITE = (176 / 255, 122 / 255, 104 / 255, 0.08)  # network weaker, metric stronger


def _outage_panel(ax, frame, title, xlim, ylim):
    """Metric change against the network's own change (both in %), one point per corridor,
    hour and site; the diagonal is "follows the network"."""
    ax.fill_between([xlim[0], 0], 0, ylim[1], color=OPPOSITE, linewidth=0, zorder=0)
    ax.plot([min(xlim[0], ylim[0]), max(xlim[1], ylim[1])], [min(xlim[0], ylim[0]), max(xlim[1], ylim[1])],
            color=FAINT, linewidth=0.7, zorder=1)
    ax.axhline(0, color=RULE, linewidth=0.7, zorder=1)
    ax.axvline(0, color=RULE, linewidth=0.7, zorder=1)
    for site in SITE_IDS:
        part = frame[frame["site"].isin([site, GROUP])] if (frame["site"] == GROUP).any() and site == SITE_IDS[0] \
            else frame[frame["site"] == site]
        color = INK if (part["site"] == GROUP).all() and len(part) else SITE_COLOR[site]
        ax.scatter(100 * part["r_network"], 100 * part["r"], s=3, color=color, alpha=0.35,
                   linewidths=0, rasterized=True, zorder=2)
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)
    ax.set_title(title, loc="left", fontsize=8.5, color=INK, fontweight="bold", pad=4)
    ax.grid(color=GRID_F2, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=7, colors=FAINT)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)


def _site_key(fig, with_group=False):
    handles = [plt.Line2D([], [], marker="o", linestyle="", markersize=4.5, color=SITE_COLOR[s],
                          label=f"site {s}") for s in SITE_IDS]
    if with_group:
        handles.append(plt.Line2D([], [], marker="o", linestyle="", markersize=4.5, color=INK, label="group"))
    handles.append(plt.Rectangle((0, 0), 1, 1, color=OPPOSITE, label="network weaker, metric stronger"))
    fig.legend(handles=handles, loc="upper right", ncol=len(handles), frameon=False, fontsize=7,
               labelcolor=INK, handletextpad=0.3, columnspacing=1.0, bbox_to_anchor=(1.0, 1.13))




def plot_outage(path=ANALYSIS_DIR / "d4_hourly.csv", floor=F3_FLOOR_PCT):
    """F3 (D4): each metric's outage change against the network's own change, in a window
    down to `floor` % (the points beyond it are stated in the text)."""
    d = pd.read_csv(path, dtype={"site": str})
    top = float(np.ceil(100 * d["r"].max() / 2) * 2 + 2)
    xlim, ylim = (floor, 2.0), (floor, top)
    fig, axes = plt.subplots(1, 4, figsize=(7.14, 2.25), sharex=True, sharey=True)
    for ax, (tag, metric) in zip(axes, zip("abcd", ("SCR", "WSCR", "SDSCR", "ESCR"))):
        part = d[d["metric"] == metric]
        beyond = (100 * part["r_network"] < floor) | (100 * part["r"] < floor)
        _outage_panel(ax, part[~beyond], f"{tag}) {metric}", xlim, ylim)
        ax.set_xlabel("network change [%]", fontsize=7, color=FAINT)
    axes[0].set_ylabel("metric change [%]", fontsize=7, color=FAINT)
    _site_key(fig, with_group=True)
    return fig


PROBE_INK = "#c0573e"


def plot_interaction_factor(path=ANALYSIS_DIR / "miif_hourly.csv", probe=AUDITED_PROBE):
    """F4 (C8): cumulative share of outage-hours by ESCR's change under the outage, all sites
    and corridors pooled, impedance interaction factor against the voltage-step probe. The
    network-only change is never materially positive (C5), so the shaded region, ESCR more
    than MATERIAL_PCT stronger, is ESCR opposing the network."""
    d = pd.read_csv(path, dtype={"site": str})
    lines = {REFERENCE_VARIANT: (INK, "impedance"),
             probe: (PROBE_INK, f"voltage step {100 * IF_STEP_OF[probe]:g} %")}
    r = {v: np.sort(100 * d.loc[d["variant"] == v, "r"].to_numpy()) for v in lines}
    lo = float(np.floor(min(x.min() for x in r.values()) / 5) * 5)
    hi = float(np.ceil(max(x.max() for x in r.values()) / 5) * 5)
    fig, ax = plt.subplots(figsize=(3.5, 2.3))
    ax.axvspan(MATERIAL_PCT, hi, color=OPPOSITE, linewidth=0, zorder=0)
    ax.axvline(0, color=RULE, linewidth=0.7, zorder=1)
    for variant, (color, label) in lines.items():
        x = r[variant]
        ax.step(x, np.arange(1, len(x) + 1) / len(x), where="post", color=color, linewidth=1.3,
                label=label, zorder=2)
    ax.set_xlim(lo, hi)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("ESCR change under the outage [%]", fontsize=7, color=FAINT)
    ax.set_ylabel("share of outage-hours", fontsize=7, color=FAINT)
    ax.text(hi - 2, 0.45, "outage makes\nESCR stronger", fontsize=6.5, color=FAINT, ha="right", va="center")
    ax.legend(fontsize=6.5, frameon=False, loc="center left", bbox_to_anchor=(0.0, 0.6), labelcolor=INK)
    ax.grid(color=GRID_F2, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(length=0, labelsize=7, colors=FAINT)
    for side in ("top", "right", "left", "bottom"):
        ax.spines[side].set_visible(False)
    return fig


# =============================================================================
# numbers.tex: the paper's run numbers as LaTeX macros (never typed into the text)
# =============================================================================

def _first(value):
    """'18: |V| 18' -> '18'; '499/24192' -> '499'; '1046/24192, -63 %' -> '1046'."""
    return value.replace(":", "/").replace(",", "/").split("/")[0].strip()


def _range(i):
    """'14.3 % to 47.2 %' -> part i (0 lower, 1 upper)."""
    return lambda value: value.split(" to ")[i].strip()


def _of(value):
    """'472/490' -> '472 of 490', for prose."""
    return value.replace("/", " of ")


def _after_comma(value):
    """'1046/24192, -63 %' -> '-63 %'."""
    return value.split(",", 1)[1].strip()


# macro name -> (claim, item[, transform]) of analysis/summary.csv. The claim is matched by its
# prefix ('C1', 'design', 'captions'); a renamed or missing row fails loudly.
NUMBER_MACROS = {
    # design (Section III)
    "WeekStart": ("design", "week start (UTC)"),
    "LoadScaleMin": ("design", "load scale, minimum"),
    "LoadScalePeak": ("design", "load scale, peak"),
    "UnitsFewestNoIBR": ("design", "committed units under the largest-first rule, fewest [base]"),
    "UnitsMostNoIBR": ("design", "committed units under the largest-first rule, most [base]"),
    "UnitsFewestIBR": ("design", "committed units under the largest-first rule, fewest [ibr]"),
    "UnitsMostIBR": ("design", "committed units under the largest-first rule, most [ibr]"),
    "LinesOut": ("design", "outages included [line]"),
    "LinesAll": ("design", "candidates [line]"),
    "TransformersOut": ("design", "outages included [transformer]"),
    "TransformersAll": ("design", "candidates [transformer]"),
    "Corridors": ("design", "corridors"),
    "States": ("design", "states evaluated"),
    # C1: weakest hour under the largest-first rule
    "ConeIntact": ("C1", "weakest hour in the fewest-committed-units hours, intact", _of),
    "ConeAll": ("C1", "same, all cases (intact and outages)", _of),
    "ConeZonly": ("C1", "confirmation: Z_ii-only weakest hour in those hours", _of),
    "ConeExceptions": ("C1", "exceptions and their explanation", _first),
    "ConeExceptionHourFirst": ("C1", "exceptions: weakest hours", _range(0)),
    "ConeExceptionHourLast": ("C1", "exceptions: weakest hours", _range(1)),
    "ConeExceptionUnits": ("C1", "exceptions: committed units at their weakest hour", _range(0)),
    "ConeExceptionGap": ("C1", "exceptions: largest gap below the minimum-commitment hours"),
    "ConeRangeLow": ("C1", "week range per metric-site (max / min - 1)", _range(0)),
    "ConeRangeHigh": ("C1", "week range per metric-site (max / min - 1)", _range(1)),
    # C2: every unit online
    "CtwoLoadLow": ("C2", "load scale at the weakest hour", _range(0)),
    "CtwoLoadHigh": ("C2", "load scale at the weakest hour", _range(1)),
    "CtwoZonlyHour": ("C2", "Z_ii-only weakest hour"),
    "CtwoDeltaVLow": ("C2", "Δln |V| part, Z_ii-only to actual weakest hour", _range(0)),
    "CtwoDeltaVHigh": ("C2", "Δln |V| part, Z_ii-only to actual weakest hour", _range(1)),
    "CtwoDeltaZLow": ("C2", "Δln Z_ii part, same", _range(0)),
    "CtwoDeltaZHigh": ("C2", "Δln Z_ii part, same", _range(1)),
    "CtwoRangeLow": ("C2", "week range per metric-site (max / min - 1)", _range(0)),
    "CtwoRangeHigh": ("C2", "week range per metric-site (max / min - 1)", _range(1)),
    # C3: planning snapshots
    "CthreeGapEone": ("C3", "largest gap to the weekly minimum, E1"),
    "CthreeGapEfiveMin": ("C3", "largest gap, E5 minimum load"),
    "CthreeGapEfiveWorse": ("C3", "largest gap, E5 worse of minimum and peak load"),
    # C4: agreement between metrics
    "CfourHoursEone": ("C4", "E1 weakest hours"),
    "CfourSpanEfive": ("C4", "E5 weakest hours span"),
    # C5: the network under an outage
    "CfiveNetMax": ("C5", "network-only change, most strengthening"),
    "CfiveNetMin": ("C5", "network-only change, most weakening"),
    # C6: SCR and WSCR
    "CsixCorrSCR": ("C6", "SCR: correlation of metric and network change"),
    "CsixCorrWSCR": ("C6", "WSCR: correlation of metric and network change"),
    # C7: ESCR and SDSCR
    "CsevenCorrESCR": ("C7", "ESCR: correlation of metric and network change"),
    "CsevenCorrSDSCR": ("C7", "SDSCR: correlation of metric and network change"),
    "CsevenOppESCR": ("C7", "ESCR: points > 1 % stronger while the network weakens"),
    "CsevenOppSDSCR": ("C7", "SDSCR: points > 1 % stronger while the network weakens"),
    "CsevenShareESCR": ("C7", "ESCR: share of points > 1 % stronger"),
    "CsevenShareSDSCR": ("C7", "SDSCR: share of points > 1 % stronger"),
    "CsevenMaxESCR": ("C7", "ESCR: largest change toward stronger"),
    "CsevenMaxSDSCR": ("C7", "SDSCR: largest change toward stronger"),
    "CsevenVoneESCR": ("C7", "ESCR: points > 1 % stronger, |V| = 1 pu", _first),
    "CsevenVoneSDSCR": ("C7", "SDSCR: points > 1 % stronger, |V| = 1 pu", _first),
    "CsevenNoIntESCR": ("C7", "ESCR: points > 1 % stronger, interaction term removed", _first),
    "CsevenNoIntSDSCR": ("C7", "SDSCR: points > 1 % stronger, interaction term removed", _first),
    # C8: ESCR's interaction factor
    "CeightImpShare": ("C8", "impedance: share of points > 1 % stronger"),
    "CeightImpMax": ("C8", "impedance: largest change toward stronger"),
    "CeightProbeHalfShare": ("C8", "probe_0.005: share of points > 1 % stronger"),
    "CeightProbeOneShare": ("C8", "probe_0.01: share of points > 1 % stronger"),
    "CeightProbeTwoShare": ("C8", "probe_0.02: share of points > 1 % stronger"),
    "CeightProbeOneMax": ("C8", "probe_0.01: largest change toward stronger"),
    "CeightProbeOneSameHour": ("C8", "probe_0.01: weakest hour as under impedance", _of),
    # C9: checks
    "CnineParallel": ("C9", "parallel circuits: same weakest hour as circuit 1", _of),
    "CnineParallelDln": ("C9", "parallel circuits: max |Δln| against circuit 1"),
    # captions
    "FthreeBeyondSCR": ("captions", "F3: SCR points beyond the window (of all, lowest value)", _first),
    "FthreeBeyondWSCR": ("captions", "F3: WSCR points beyond the window (of all, lowest value)", _first),
    "FthreeBeyondESCR": ("captions", "F3: ESCR points beyond the window (of all, lowest value)", _first),
    "FthreeBeyondSDSCR": ("captions", "F3: SDSCR points beyond the window (of all, lowest value)", _first),
    "FthreeLowestSCR": ("captions", "F3: SCR points beyond the window (of all, lowest value)", _after_comma),
}


def _latex(value):
    """Summary text -> LaTeX: '%' escaped with a thin space, a leading minus as math minus."""
    value = str(value).strip().replace(" %", "\\,\\%")
    return "$-$" + value[1:] if value.startswith("-") else value


def numbers_tex(analysis_dir=ANALYSIS_DIR, validation_dir=VALIDATION_DIR):
    """numbers.tex: one \\newcommand per run number the paper quotes, from the analysis
    summary and the PowerWorld validation summary (largest fault-current error, %)."""
    summary = pd.read_csv(Path(analysis_dir) / "summary.csv", dtype=str, keep_default_na=False)
    values = {}
    for name, (claim, item, *transform) in NUMBER_MACROS.items():
        rows = summary[summary["claim"].str.startswith(claim) & (summary["item"] == item)]
        if len(rows) != 1:
            raise KeyError(f"\\Num{name}: {len(rows)} summary rows for ({claim!r}, {item!r})")
        value = rows["result"].iloc[0]
        values[name] = _latex(transform[0](value) if transform else value)
    validation = pd.read_csv(Path(validation_dir) / "summary.csv")
    states = validation[validation["comparison"] == "state"]
    values["PowerWorldMaxErrorPct"] = f"{states['I_max_abs_error_pct'].max():.2g}"
    values["Outages"] = str(int(values["LinesOut"]) + int(values["TransformersOut"]))  # included branches
    values["Cases"] = str(int(values["Corridors"]) + 1)  # the intact case and one per corridor
    lines = ["% Generated by experiments/glover37/report.py from analysis/summary.csv and",
             "% validation/summary.csv. Do not edit; rerun the report stage."]
    lines += [f"\\newcommand{{\\Num{name}}}{{{value}}}" for name, value in values.items()]
    return "\n".join(lines) + "\n"


# =============================================================================
# report.md: the claims and their numbers (text requirements, C1-C9)
# =============================================================================

def build_report(analysis_dir=ANALYSIS_DIR, raw_dir=RAW_DIR):
    """report.md: the claim and caption rows of summary.csv (written by the analysis stage),
    grouped by claim. No number is computed here."""
    summary = pd.read_csv(Path(analysis_dir) / "summary.csv", dtype=str, keep_default_na=False)
    rows = summary[summary["claim"].str.match(r"^(C\d|captions)")]
    raw_m, ana_m = manifest.read(raw_dir) or {}, manifest.read(analysis_dir) or {}
    lines = ["# System-strength study: report (generated)", "",
             f"- raw run `{raw_m.get('run_id')}`, analysis run `{ana_m.get('run_id')}`, commit "
             f"`{(raw_m.get('git_commit') or '')[:7]}`, of record: **{raw_m.get('of_record')}**",
             "- every number below is a row of `analysis/summary.csv` (claim, item, result, source table)",
             "- floats: F1a/F1b network and plant, F2a/F2b intact week (E1, E5), F3 outages (E2), F4 interaction factor (E3), T1 conventions"]
    for claim, g in rows.groupby("claim", sort=False):
        lines += ["", f"## {claim}"]
        lines += [f"- {r.item}: **{r.result}** ({r.table})" for r in g.itertuples()]
    return "\n".join(lines) + "\n"


# =============================================================================
# Driver
# =============================================================================

FLOATS = {"f1a_network": plot_network, "f1b_plant": plot_plant_chain, "f2_week": plot_week,
          "f3_outage": plot_outage,
          "f4_interaction_factor": plot_interaction_factor}


def run(png=False):
    started = manifest.start()
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    written = []
    for stem, draw in FLOATS.items():
        fig = draw()
        for suffix in (".pdf", ".png") if png else (".pdf",):
            path = (FIGURES_DIR / stem).with_suffix(suffix)
            # no creation date: the tracked PDFs are byte-identical when the inputs are
            fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.02,
                        metadata={"CreationDate": None} if suffix == ".pdf" else None)
            written.append(path)
        plt.close(fig)
    table, latex = table_conventions()
    table.to_csv(FIGURES_DIR / "t1_conventions.csv", index=False)
    (FIGURES_DIR / "t1_conventions.tex").write_text(latex, encoding="utf-8")
    (FIGURES_DIR / "numbers.tex").write_text(numbers_tex(), encoding="utf-8")
    written.append(FIGURES_DIR / "numbers.tex")
    (REPORT_DIR / "report.md").write_text(build_report(), encoding="utf-8")
    written += [FIGURES_DIR / "t1_conventions.csv", FIGURES_DIR / "t1_conventions.tex",
                REPORT_DIR / "report.md"]
    analysis_m = manifest.read(ANALYSIS_DIR) or {}
    manifest.write(REPORT_DIR, "report", started=started,
                   inputs=sorted(ANALYSIS_DIR.glob("*.csv")), outputs=written,
                   extra={"analysis_run_id": analysis_m.get("run_id")})
    for path in written:
        print("Saved", path)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--png", action="store_true", help="Also save PNG copies of the floats.")
    run(parser.parse_args().png)


if __name__ == "__main__":
    main()
