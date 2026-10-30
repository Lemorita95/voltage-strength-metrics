"""Every parameter of the system-strength study, in one place.

Sites and equipment, the week and commitment rules, the contingency
rule, the short-circuit and interaction-factor conventions, the experiments
(E0-E5), the PowerWorld validation states (E0) and the analysis and text
thresholds. Nothing here runs a calculation; the other modules import from it.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

# =============================================================================
# 1. Paths
# =============================================================================

ROOT = Path(__file__).resolve().parents[2]
CASE_FILE = ROOT / "cases" / "glover37.json"
POWERWORLD_ROOT = ROOT / "data" / "powerworld" / "glover37"
GEN_LIMITS_FILE = POWERWORLD_ROOT / "base" / "generators.csv"
AXD_FILE = POWERWORLD_ROOT / "base" / "DesignCase2_2010.axd"
SE3_FILE = ROOT / "data" / "se3_2025.csv"

RESULTS_ROOT = ROOT / "results" / "glover37"
RAW_DIR = RESULTS_ROOT / "raw"
ANALYSIS_DIR = RESULTS_ROOT / "analysis"
VALIDATION_DIR = RESULTS_ROOT / "validation"
FIGURES_DIR = ROOT / "figures" / "glover37"

# =============================================================================
# 2. Network and week
# =============================================================================

S_BASE_MVA = 100.0
SLACK_BUS = 31

# Public SE3 demand is only a load-scaling signal: the 168 h window of YEAR
# with the largest load range, scaled to its own mean.
YEAR = 2025
WINDOW_HOURS = 168
SE3_TIMESTAMP_COLUMN = "timestamp"
SE3_LOAD_COLUMN = "SE3"

# Commitment rules (the `commitment` column of every raw file):
#   sequential  non-slack units filled in descending MWMax (tie-break bus, ID); stop
#               when the residual is below the next unit's MWMin; slack takes the rest.
#   all_online  rule (a): every non-slack unit online at max(MWMin, s * MWMax), s in
#               [0, 1] set to meet the target; the slack takes the rest. When the units'
#               MWMin alone exceeds the target (s = 0) the slack exports (slack_exporting).
# Both dispatch demand on the 37-bus network and demand - IBR_MW on the 46-bus one.
SEQUENTIAL, ALL_ONLINE = "sequential", "all_online"
COMMITMENT_RULES = (SEQUENTIAL, ALL_ONLINE)

# =============================================================================
# 3. IBR sites
# =============================================================================
# All branch impedances are on the 100 MVA system base. Transformer per-unit
# nominal taps are unity because the 69/34.5/0.69 kV voltage bases follow the
# respective bus sides.

HV_KV, MV_KV, LV_KV = 69.0, 34.5, 0.69
TRANSFORMER_Z_EQUIPMENT_BASE = 0.005 + 0.060j
FEEDER_Z_SYSTEM_BASE = 0.011 + 0.027j
FEEDER_B_SYSTEM_BASE = 0.069
IBR_MVA_MULTIPLIER = 1.1
TRANSFORMER_MVA_MULTIPLIER = 1.2
Q_CAPABILITY_MULTIPLIER = 0.436
# Converters in the fault calculation (models/FullConverter.py FAULT_MODELS): an ideal
# current source holding its pre-fault current, i.e. no fault contribution.
CONVERTER_FAULT_MODEL = "constant_current"


@dataclass(frozen=True)
class IBRSite:
    host_bus: int
    pmax_mw: float
    main_mv_bus: int
    padmount_hv_bus: int
    converter_lv_bus: int

    @property
    def poi_bus(self) -> int:
        """IEEE 2800 evaluation point (original Glover HV connection bus)."""
        return self.host_bus

    @property
    def escr_bus(self) -> int:
        """Fingrid-inspired ESCR evaluation point (padmount HV, 34.5 kV)."""
        return self.padmount_hv_bus

    @property
    def sdscr_bus(self) -> int:
        """Wu (2018) evaluation point (converter LV terminal)."""
        return self.converter_lv_bus

    @property
    def converter_mva(self) -> float:
        return IBR_MVA_MULTIPLIER * self.pmax_mw

    @property
    def transformer_mva(self) -> float:
        return TRANSFORMER_MVA_MULTIPLIER * self.converter_mva

    @property
    def q_limit_mvar(self) -> float:
        return Q_CAPABILITY_MULTIPLIER * self.converter_mva

    @property
    def transformer_z_pu(self) -> complex:
        return TRANSFORMER_Z_EQUIPMENT_BASE * (S_BASE_MVA / self.transformer_mva)


# Two MV buses + one LV bus at each site. Bus IDs do not overlap Glover IDs.
# Every plant runs at its rated output, the worst case: the largest P in every
# metric's denominator and the most synchronous generation displaced.
SITES = (
    IBRSite(21, 60.0, 2101, 2102, 2103),
    IBRSite(48, 45.0, 4801, 4802, 4803),
    IBRSite(16, 45.0, 1601, 1602, 1603),
)
IBR_MW = sum(site.pmax_mw for site in SITES)

# =============================================================================
# 4. Metrics, operating-state checks, contingencies
# =============================================================================

METRICS = ("SCR", "WSCR", "ESCR", "SDSCR")
# Network each metric is evaluated on: the 37-bus case or the 46-bus IBR case.
NETWORK = {"SCR": "base", "WSCR": "base", "ESCR": "ibr", "SDSCR": "ibr"}
# Denominator as run: every metric uses rated MW.
AS_RUN = {m: "rated" for m in METRICS}
EVAL_BUS = {
    "SCR": {str(s.host_bus): s.poi_bus for s in SITES},
    "WSCR": {str(s.host_bus): s.poi_bus for s in SITES},
    "ESCR": {str(s.host_bus): s.escr_bus for s in SITES},
    "SDSCR": {str(s.host_bus): s.sdscr_bus for s in SITES},
}
# Evaluation point and voltage level of each metric (T1)
EVAL_POINT = {"SCR": ("host bus", HV_KV), "WSCR": ("host bus", HV_KV),
              "ESCR": ("padmount", MV_KV), "SDSCR": ("converter terminal", LV_KV)}
# Definitions as computed (T1, LaTeX). S_i: SCMVA = S_base |I_f| (V_n = 1 pu) from a bolted
# fault (SCC_NR) at the actual pre-fault state; P: rated MW; Z: Y_SCC^-1 (builder).
METRIC_DEFINITIONS = {
    "SCR": {"formula": r"$S_i / P_i$", "interaction": "none",
            "source": "IEEE 2800-2022"},
    "WSCR": {"formula": r"$\sum_i S_i P_i \,/\, (\sum_i P_i)^2$", "interaction": r"weights $P_i$",
             "source": "IEEE 2800-2022"},
    "ESCR": {"formula": r"$S_i \,/\, (P_i + \sum_{j \ne i} \mathrm{MIIF}_{ji} P_j)$",
             "interaction": r"$\mathrm{MIIF}_{ji} = |Z_{ji}/Z_{ii}|$",
             "source": "CIGRE multi-infeed brochure, eq. (1-4), $Q_f = 0$"},
    "SDSCR": {"formula": r"$|V_i|^2 \,/\, (|Z_{ii}|\,|P_i + \sum_{j \ne i} w_{ij} P_j|)$",
              "interaction": r"$w_{ij} = (Z_{ij}/Z_{ii})\,(V_i/V_j)^*$",
              "source": "Wu et al. (2018)"},
}

# Short circuit (SCC_NR): SCMVA = S_BASE_MVA * V_n * |I_f| with V_n = 1 pu, from a
# bolted fault (fault impedance FAULT_Z_PU, the solver default) started from the
# actual pre-fault state. Impedances come from the solver's Y_SCC builder.
FAULT_Z_PU = 1e-6

# ESCR is CIGRE's multi-infeed ESCR (eq. 1-4), with Qf = 0 (no filters or capacitors):
#   ESCR_i = SCC_i / (P_i + sum_j MIIF_ji * P_j),   MIIF_ji = dV_j / dV_i, disturbance at i.
# Each MIIF construction is one ESCR `variant`:
#   impedance        |Z_ji / Z_ii|, the instant response (CIGRE): the reference
#   probe_<step>     power-flow voltage step of <step> at bus i (E3 robustness)
IF_STEPS = (0.005, 0.01, 0.02)
IMPEDANCE_VARIANT = "impedance"


def probe_variant(step):
    return f"probe_{step:g}"


PROBE_VARIANTS = tuple(probe_variant(s) for s in IF_STEPS)
REFERENCE_VARIANT = IMPEDANCE_VARIANT
ESCR_VARIANTS = (IMPEDANCE_VARIANT, *PROBE_VARIANTS)
# The probe whose audit terms (probe Q, target/actual V, dV, generator state changes)
# are written; the other probes write only their interaction factors.
AUDITED_PROBE = probe_variant(0.01)

# VoltageLimits preset (models/VoltageLimits.py); bands only flag states.
VOLTAGE_PRESET = "default"

# N-1 rule: every line whose outage islands no bus and whose base and IBR power
# flows converge in every hour, commitment and setpoints held at the intact schedule.
INTACT = "intact"

# =============================================================================
# 5. Analysis definitions
# =============================================================================

REL_TIE = 1e-9
# Text and caption numbers (analyze writes them, report formats them)
F3_FLOOR_PCT = -20.0      # F3 display window: weakenings beyond it are not drawn, only counted
NEAR_MIN_PCT = 1.0        # "within x % of the weekly minimum" (E5 flatness, C2)
MATERIAL_PCT = 1.0        # "> x % stronger" (C6-C8)
RECONSTRUCTION_TOL = 1e-9

# =============================================================================
# 6. Experiments, each tied to the research questions it serves. Each raw row's experiment follows from its commitment
# rule and case (experiment_of); E3 is the ESCR variants other than the reference.
# RQ/D is looked up here, never stored per raw row.
# =============================================================================


@dataclass(frozen=True)
class Experiment:
    id: str
    name: str
    states: str
    serves: tuple[str, ...]


EXPERIMENTS = {
    "E0": Experiment("E0", "Benchmark validation",
                     "PowerWorld agreement: base + A1-A3, B1-B3", ("method validity",)),
    "E1": Experiment("E1", "Intact week, unit-commitment logic",
                     "168 h, SE3 load scaling, sequential commitment by MWMax",
                     ("RQ1-D1", "RQ1-D5", "RQ2")),
    "E2": Experiment("E2", "N-1 line contingencies",
                     "every included line outage x 168 h, commitment held at E1",
                     ("RQ1-D4", "RQ1-D1", "RQ1-D5")),
    "E3": Experiment("E3", "MIIF robustness (ESCR)",
                     "ESCR on every state with probe MIIF at 0.5, 1 and 2 %",
                     ("RQ1-D4", "RQ3")),
    "E5": Experiment("E5", "Intact week, dispatch only",
                     "168 h, every unit online, rule (a)", ("RQ1-D1", "RQ1-D5", "RQ2")),
}


def experiment_of(commitment, case):
    """E1, E2 or E5 for a raw row's (commitment, case)."""
    if commitment == SEQUENTIAL:
        return "E1" if case == INTACT else "E2"
    if commitment == ALL_ONLINE and case == INTACT:
        return "E5"
    raise ValueError(f"No experiment for commitment={commitment!r}, case={case!r}")


# =============================================================================
# 7. E0: PowerWorld validation states
# =============================================================================


class Action(str, Enum):
    SET_GENERATOR_P = "set_generator_p"
    DISCONNECT = "disconnect"


@dataclass(frozen=True)
class Change:
    action: Action
    element: str
    value: float | None = None


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    changes: tuple[Change, ...]


E0_SCENARIOS = {
    "A1": Scenario("A1", "Generator 28 dispatched to 0 MW while remaining "
                         "connected and synchronized.",
                   (Change(Action.SET_GENERATOR_P, "generator.g28-1", 0.0),)),
    "A2": Scenario("A2", "Generator 28 disconnected.",
                   (Change(Action.DISCONNECT, "generator.g28-1"),)),
    "A3": Scenario("A3", "Generator 14 disconnected.",
                   (Change(Action.DISCONNECT, "generator.g14-1"),)),
    "B1": Scenario("B1", "Line 14-34 circuit 1 disconnected.",
                   (Change(Action.DISCONNECT, "line.line 14-34 ckt 1"),)),
    "B2": Scenario("B2", "Line 21-48 circuit 1 disconnected.",
                   (Change(Action.DISCONNECT, "line.line 21-48 ckt 1"),)),
    "B3": Scenario("B3", "Transformer 28-29 circuit 1 disconnected.",
                   (Change(Action.DISCONNECT, "transformer.transformer 28-29 ckt 1"),)),
}
E0_STATES = ("base", *E0_SCENARIOS)
