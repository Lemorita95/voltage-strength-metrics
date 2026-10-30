"""
Report stage: the paper's floats and the generated report of the system-strength study.

    F1a  network, with each host bus's plant converter          (case, AXD geometry)
    F1b  one plant's connection and each metric's evaluation point
    F2a  load scale over the intact week, minimum and peak load  (D1, RQ2, D5)
    F2b  where each metric is lowest, E1 against E5              (D1, RQ2, D5)
    F3   metric change against the network's change, per outage (D4)
    T1   metric conventions, generated from config.py
    report.md  the claims C1-C9 with their numbers, computed from the analysis tables

Usage
-----
    python -m experiments.glover37.report
    python -m experiments.glover37.report --png

Inputs:  results/glover37/raw/ and results/glover37/analysis/
Outputs: figures/glover37/{f1a,f1b,f2a,f2b,f3}*.pdf, t1_conventions.{tex,csv};
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
    ALL_ONLINE, ANALYSIS_DIR, AS_RUN, F3_FLOOR_PCT, EVAL_BUS, EVAL_POINT, FIGURES_DIR, INTACT, METRIC_DEFINITIONS,
    METRICS, NETWORK, RAW_DIR, REFERENCE_VARIANT, RESULTS_ROOT, SEQUENTIAL, SITES,
)

REPORT_DIR = RESULTS_ROOT / "report"


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


def _week_axis(ax, hours, stamps, labels=True):
    """Day gridlines and labels, recessive frame, as in the explorer."""
    days = [h for h in hours if stamps.loc[h].hour == 0]
    ax.set_xlim(hours[0] - 3, hours[-1] + 3)
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
    nodes = [(0.0, "host bus", "69 kV", ("SCR", "WSCR")), (1.6, "collector", "34.5 kV", ()),
             (3.2, "padmount", "34.5 kV", ("ESCR",)), (4.8, "terminal", "0.69 kV", ("SDSCR",))]
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


NETWORK_NAME = {"base": "37-bus (no IBR)", "ibr": "46-bus (with IBR)"}


def table_conventions():
    """T1: what each metric measures, where and with what, from config (cannot drift).
    Returns (DataFrame, LaTeX)."""
    rows = []
    for metric in METRICS:
        point, kv = EVAL_POINT[metric]
        d = METRIC_DEFINITIONS[metric]
        rows.append({"metric": metric, "network": NETWORK_NAME[NETWORK[metric]],
                     "evaluated_at": f"{point}, {kv:g} kV", "definition": d["formula"],
                     "interaction": d["interaction"], "denominator": f"{AS_RUN[metric]} MW",
                     "source": d["source"]})
    t = pd.DataFrame(rows)
    body = "\n".join(
        f"{r.metric} & {r.definition} \\\\\n"
        f" & \\footnotesize {r.network}, {r.evaluated_at}; {r.interaction}; {r.source} \\\\[2pt]"
        for r in t.itertuples())
    # needs \\usepackage{array,booktabs} in the paper
    latex = ("\\begin{tabular}{@{}l >{\\raggedright\\arraybackslash}p{0.82\\linewidth}@{}}\n\\toprule\n"
             "Metric & Definition; network, evaluation point; interaction; source \\\\\n\\midrule\n"
             f"{body}\n\\bottomrule\n\\end{{tabular}}\n")
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
             "- floats: F1a/F1b network and plant, F2a/F2b intact week (E1, E5), F3 outages (E2), T1 conventions"]
    for claim, g in rows.groupby("claim", sort=False):
        lines += ["", f"## {claim}"]
        lines += [f"- {r.item}: **{r.result}** ({r.table})" for r in g.itertuples()]
    return "\n".join(lines) + "\n"


# =============================================================================
# Driver
# =============================================================================

FLOATS = {"f1a_network": plot_network, "f1b_plant": plot_plant_chain, "f2a_load": plot_load,
          "f2b_minima": plot_minima, "f3_outage": plot_outage}


def run(png=False):
    started = manifest.now()
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
