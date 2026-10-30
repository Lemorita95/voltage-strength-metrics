"""
E0: validate the Python short-circuit results against PowerWorld.

Solves the validated 37-bus baseline and the six static cases A1--B3 (config
E0_SCENARIOS): pre-fault power flow, then the short-circuit current for a
bolted three-phase fault at every bus. Compares each state with the PowerWorld
reference calculation, and the SCL change fields relative to the baseline and
A2--A1.

Usage
-----
    python -m experiments.glover37.validate

Inputs
------
cases/glover37.json (rebuilt from the PowerWorld exports by pw2json.py)
data/powerworld/glover37/<state>/shortcircuit.csv

Outputs (results/glover37/validation/)
-------
<state>/results.csv                                   Python bus results
<state>/python_vs_powerworld.csv                      state comparison
<scenario>/delta_from_base_python_vs_powerworld.csv   ΔSCL comparison
A2_vs_A1/python_vs_powerworld.csv                     A2--A1 comparison
summary.csv                                           one line per comparison
"""

from __future__ import annotations

import cmath
import csv
import math
import re

import numpy as np
import pandas as pd

from models.CustomNetwork import CustomNetwork
from models.PowerFlow import PowerFlow
from models.ShortCircuit import ShortCircuit as SC
from experiments.glover37 import manifest
from experiments.glover37.config import (
    CASE_FILE, E0_SCENARIOS, E0_STATES, POWERWORLD_ROOT, S_BASE_MVA, VALIDATION_DIR, Action,
)

CSV_HEADER = [
    "bus", "pre_fault_V_mag", "pre_fault_V_ang_deg", "P_net", "Q_net", "P_gen", "Q_gen",
    "P_conv", "Q_conv", "P_load", "Q_load", "P_shunt", "Q_shunt",
    "I_SCC_mag_pu", "I_SCC_ang_deg", "SCL_pu", "SCL_MVA",
]


# =============================================================================
# 1. Python short-circuit results
# =============================================================================

def apply_scenario(network: CustomNetwork, scenario) -> None:
    """Apply one predefined change set (None: baseline)."""
    for change in scenario.changes if scenario else ():
        if change.action == Action.SET_GENERATOR_P:
            key = change.element.lower()
            if key not in network.generators:
                raise KeyError(f"Generator not found: {change.element}")
            if change.value is None:
                raise ValueError(f"{change.action.value} requires a numerical value.")
            network.generators[key].P = float(change.value)
        elif change.action == Action.DISCONNECT:
            network.change_element_status(change.element, False)
        else:
            raise ValueError(f"Unsupported scenario action: {change.action}")


def solve_state(state: str) -> list[list]:
    """
    Load a fresh 37-bus case, apply one scenario, solve the pre-fault power
    flow, and calculate short-circuit current for a fault at every bus.

    SCL_pu = V_nominal_pu * I_SCC_pu with V_nominal_pu = 1.0; SCL_MVA = S_BASE_MVA * SCL_pu.
    """
    network = CustomNetwork(str(CASE_FILE))
    network.create_network()
    apply_scenario(network, None if state == "base" else E0_SCENARIOS[state])
    # Build passive network after any topology changes.
    network.build_ybus()
    passive_ybus = network.YBus

    pf_solver = PowerFlow()
    pf_result = pf_solver.solve_power_flow(network)
    if pf_result is None:
        raise RuntimeError(f"{state}: power flow did not converge.")
    theta, voltage = pf_result
    pf_state = np.concatenate((theta, voltage))
    v_complex = np.array([v * np.exp(1j * angle) for angle, v in zip(theta, voltage)])

    active_elements = network.get_active_elements(
        v_complex,
        pf_solver.get_active_elements_flow(network, pf_state),
        pf_solver.get_active_element_status(network),
    )
    case_data, _ = pf_solver.power_flow_summary(network, pf_state)

    failed = []
    for row, bus in zip(case_data, network.buses.keys()):
        i_scc = SC.SCC_NR(network.idx[bus], v_complex, passive_ybus, active_elements)
        if i_scc is None:
            failed.append(bus)
            row.extend([None, None, None, None])
            continue
        scl_pu = 1.0 * abs(i_scc)
        row.extend([abs(i_scc), cmath.phase(i_scc) * 180.0 / cmath.pi, scl_pu, S_BASE_MVA * scl_pu])
    if failed:
        raise RuntimeError(f"{state}: short circuit did not converge for bus(es) {failed}.")
    return case_data


def save_state(state: str, case_data: list[list]) -> None:
    path = VALIDATION_DIR / state / "results.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(CSV_HEADER)
        writer.writerows(case_data)
    print(f"Wrote: {path}")


# =============================================================================
# 2. Comparison with PowerWorld
# =============================================================================

def pw_float(value: str | None) -> float:
    if value is None or not str(value).strip():
        raise ValueError("Missing numeric value in PowerWorld CSV.")
    return float(str(value).strip().replace(",", "."))


def parse_bus_id(whoami: str) -> int:
    match = re.fullmatch(r"\s*Bus\s+'(\d+)'\s*", whoami or "")
    if not match:
        raise ValueError(f"Could not parse PowerWorld bus from WhoAmI={whoami!r}")
    return int(match.group(1))


def angle_diff(a: float, b: float) -> float:
    return (a - b + 180.0) % 360.0 - 180.0


def pct(delta: float, reference: float) -> float:
    return math.nan if abs(reference) < 1e-15 else 100.0 * delta / reference


def rmse(series: pd.Series) -> float:
    return float((series.pow(2).mean()) ** 0.5)


def assert_same_buses(label: str, *frames: pd.DataFrame) -> None:
    bus_sets = [set(df.index) for df in frames]
    if not all(buses == bus_sets[0] for buses in bus_sets[1:]):
        raise ValueError(f"{label}: bus sets do not match.")


def read_python(state: str) -> pd.DataFrame:
    path = VALIDATION_DIR / state / "results.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Python result file not found: {path}")

    df = pd.read_csv(path)
    required = {
        "bus",
        "pre_fault_V_mag",
        "pre_fault_V_ang_deg",
        "I_SCC_mag_pu",
        "I_SCC_ang_deg",
        "SCL_pu",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")

    if df["bus"].duplicated().any():
        raise ValueError(f"{path} contains duplicate bus rows.")

    return (
        df.rename(
            columns={
                "pre_fault_V_mag": "V_mag_pu",
                "pre_fault_V_ang_deg": "V_angle_deg",
                "I_SCC_mag_pu": "I_mag_pu",
                "I_SCC_ang_deg": "I_angle_deg",
            }
        )
        .set_index("bus")
        [["V_mag_pu", "V_angle_deg", "I_mag_pu", "I_angle_deg", "SCL_pu"]]
        .sort_index()
    )


def read_powerworld(state: str) -> pd.DataFrame:
    path = POWERWORLD_ROOT / state / "shortcircuit.csv"
    if not path.is_file():
        raise FileNotFoundError(f"PowerWorld result file not found: {path}")

    with path.open("r", encoding="utf-8-sig", newline="") as f:
        if not f.readline().strip():
            raise ValueError(f"{path} is empty.")

        reader = csv.DictReader(f)
        required = {
            "WhoAmI",
            "FaultType",
            "FaultImpedance",
            "FaultImpedance:1",
            "ABCPhaseI",
            "ABCPhaseAngle",
            "FaultThevImp",
            "FaultThevImp:1",
        }
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")

        rows = []
        for row in reader:
            bus = parse_bus_id(row["WhoAmI"])

            if row["FaultType"].strip() != "3PB":
                raise ValueError(
                    f"{state}, bus {bus}: expected 3PB fault, "
                    f"found {row['FaultType']!r}."
                )

            fault_r = pw_float(row["FaultImpedance"])
            fault_x = pw_float(row["FaultImpedance:1"])
            if abs(fault_r) > 1e-12 or abs(fault_x) > 1e-12:
                raise ValueError(
                    f"{state}, bus {bus}: expected bolted fault, "
                    f"found R={fault_r}, X={fault_x}."
                )

            i_mag = pw_float(row["ABCPhaseI"])
            i_angle = pw_float(row["ABCPhaseAngle"])
            r_th = pw_float(row["FaultThevImp"])
            x_th = pw_float(row["FaultThevImp:1"])

            i_complex = i_mag * cmath.exp(1j * math.radians(i_angle))
            v_complex = i_complex * complex(r_th, x_th)

            rows.append(
                {
                    "bus": bus,
                    "V_mag_pu": abs(v_complex),
                    "V_angle_deg": math.degrees(cmath.phase(v_complex)),
                    "I_mag_pu": i_mag,
                    "I_angle_deg": i_angle,
                    "SCL_pu": i_mag,
                    "R_th_pu": r_th,
                    "X_th_pu": x_th,
                }
            )

    df = pd.DataFrame(rows)
    if df["bus"].duplicated().any():
        raise ValueError(f"{path} contains duplicate bus rows.")
    return df.set_index("bus").sort_index()


def compare_state(state: str) -> pd.DataFrame:
    py = read_python(state)
    pw = read_powerworld(state)
    assert_same_buses(state, py, pw)

    out = pd.DataFrame(index=py.index)
    out["python_I_SCC_pu"] = py["I_mag_pu"]
    out["powerworld_I_SCC_pu"] = pw["I_mag_pu"]
    out["delta_I_SCC_pu"] = py["I_mag_pu"] - pw["I_mag_pu"]
    out["abs_delta_I_SCC_pu"] = out["delta_I_SCC_pu"].abs()
    out["delta_I_SCC_pct_of_PW"] = 100.0 * out["delta_I_SCC_pu"] / pw["I_mag_pu"]
    out["abs_delta_I_SCC_pct_of_PW"] = out["delta_I_SCC_pct_of_PW"].abs()

    out["python_I_angle_deg"] = py["I_angle_deg"]
    out["powerworld_I_angle_deg"] = pw["I_angle_deg"]
    out["delta_I_angle_deg"] = [
        angle_diff(a, b)
        for a, b in zip(py["I_angle_deg"], pw["I_angle_deg"])
    ]
    out["abs_delta_I_angle_deg"] = out["delta_I_angle_deg"].abs()

    out["python_V_pre_pu"] = py["V_mag_pu"]
    out["powerworld_V_pre_reconstructed_pu"] = pw["V_mag_pu"]
    out["delta_V_pre_pu"] = py["V_mag_pu"] - pw["V_mag_pu"]
    out["abs_delta_V_pre_pu"] = out["delta_V_pre_pu"].abs()

    out["python_V_pre_angle_deg"] = py["V_angle_deg"]
    out["powerworld_V_pre_angle_reconstructed_deg"] = pw["V_angle_deg"]
    out["delta_V_pre_angle_deg"] = [
        angle_diff(a, b)
        for a, b in zip(py["V_angle_deg"], pw["V_angle_deg"])
    ]
    out["abs_delta_V_pre_angle_deg"] = out["delta_V_pre_angle_deg"].abs()

    out["powerworld_R_th_pu"] = pw["R_th_pu"]
    out["powerworld_X_th_pu"] = pw["X_th_pu"]

    return out.reset_index()


def compare_change(
    left: str,
    right: str,
    *,
    left_label: str,
    right_label: str,
    delta_label: str,
) -> pd.DataFrame:
    py_left, py_right = read_python(left), read_python(right)
    pw_left, pw_right = read_powerworld(left), read_powerworld(right)
    assert_same_buses(delta_label, py_left, py_right, pw_left, pw_right)

    py_delta = py_right["SCL_pu"] - py_left["SCL_pu"]
    pw_delta = pw_right["SCL_pu"] - pw_left["SCL_pu"]
    py_pct = 100.0 * py_delta / py_left["SCL_pu"]
    pw_pct = 100.0 * pw_delta / pw_left["SCL_pu"]

    out = pd.DataFrame(index=py_left.index)
    out[f"python_{left_label}_SCL_pu"] = py_left["SCL_pu"]
    out[f"python_{right_label}_SCL_pu"] = py_right["SCL_pu"]
    out[f"python_{delta_label}_SCL_pu"] = py_delta
    out[f"python_{delta_label}_pct"] = py_pct
    out[f"powerworld_{left_label}_SCL_pu"] = pw_left["SCL_pu"]
    out[f"powerworld_{right_label}_SCL_pu"] = pw_right["SCL_pu"]
    out[f"powerworld_{delta_label}_SCL_pu"] = pw_delta
    out[f"powerworld_{delta_label}_pct"] = pw_pct

    out["delta_field_error_pu"] = py_delta - pw_delta
    out["abs_delta_field_error_pu"] = out["delta_field_error_pu"].abs()
    out["delta_field_error_pct_points"] = py_pct - pw_pct
    out["abs_delta_field_error_pct_points"] = (
        out["delta_field_error_pct_points"].abs()
    )
    return out.reset_index()


def compare_change_from_base(scenario: str) -> pd.DataFrame:
    return compare_change(
        "base",
        scenario,
        left_label="base",
        right_label="scenario",
        delta_label="delta_SCL",
    )


def compare_a2_minus_a1() -> pd.DataFrame:
    return compare_change(
        "A1",
        "A2",
        left_label="A1",
        right_label="A2",
        delta_label="A2_minus_A1",
    )


def state_summary(state: str, df: pd.DataFrame) -> dict:
    worst_i = df.loc[df["abs_delta_I_SCC_pct_of_PW"].idxmax()]
    worst_v = df.loc[df["abs_delta_V_pre_pu"].idxmax()]
    return {
        "comparison": "state",
        "case": state,
        "buses": len(df),
        "I_MAE_pu": df["abs_delta_I_SCC_pu"].mean(),
        "I_RMSE_pu": rmse(df["delta_I_SCC_pu"]),
        "I_mean_abs_error_pct": df["abs_delta_I_SCC_pct_of_PW"].mean(),
        "I_max_abs_error_pct": worst_i["abs_delta_I_SCC_pct_of_PW"],
        "I_max_abs_error_bus": int(worst_i["bus"]),
        "I_angle_MAE_deg": df["abs_delta_I_angle_deg"].mean(),
        "V_MAE_pu": df["abs_delta_V_pre_pu"].mean(),
        "V_max_abs_error_pu": worst_v["abs_delta_V_pre_pu"],
        "V_max_abs_error_bus": int(worst_v["bus"]),
        "V_angle_MAE_deg": df["abs_delta_V_pre_angle_deg"].mean(),
    }


def change_summary(label: str, df: pd.DataFrame) -> dict:
    worst = df.loc[df["abs_delta_field_error_pct_points"].idxmax()]
    return {
        "comparison": "change_field",
        "case": label,
        "buses": len(df),
        "delta_SCL_MAE_pu": df["abs_delta_field_error_pu"].mean(),
        "delta_SCL_RMSE_pu": rmse(df["delta_field_error_pu"]),
        "delta_SCL_MAE_pct_points": df[
            "abs_delta_field_error_pct_points"
        ].mean(),
        "delta_SCL_RMSE_pct_points": rmse(df["delta_field_error_pct_points"]),
        "delta_SCL_max_abs_error_pct_points": worst[
            "abs_delta_field_error_pct_points"
        ],
        "delta_SCL_max_abs_error_bus": int(worst["bus"]),
    }


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"Wrote: {path}")


def print_summary(summary: dict) -> None:
    print(f"\n{summary['case']}")
    print("-" * 48)

    if summary["comparison"] == "state":
        print(f"Buses: {summary['buses']}")
        print(
            f"Fault current: mean |error| = {summary['I_mean_abs_error_pct']:.4f} %, "
            f"max = {summary['I_max_abs_error_pct']:.4f} % "
            f"(bus {summary['I_max_abs_error_bus']})"
        )
        print(
            f"Pre-fault V: mean |error| = {summary['V_MAE_pu']:.8f} pu, "
            f"max = {summary['V_max_abs_error_pu']:.8f} pu "
            f"(bus {summary['V_max_abs_error_bus']})"
        )
    else:
        print(f"Buses: {summary['buses']}")
        print(
            f"ΔSCL field: mean |error| = "
            f"{summary['delta_SCL_MAE_pct_points']:.6f} pp, "
            f"RMSE = {summary['delta_SCL_RMSE_pct_points']:.6f} pp, "
            f"max = {summary['delta_SCL_max_abs_error_pct_points']:.6f} pp "
            f"(bus {summary['delta_SCL_max_abs_error_bus']})"
        )


def validate_state(state: str) -> dict:
    df = compare_state(state)
    write_csv(df, VALIDATION_DIR / state / "python_vs_powerworld.csv")
    summary = state_summary(state, df)
    print_summary(summary)
    return summary


def validate_change_from_base(scenario: str) -> dict:
    df = compare_change_from_base(scenario)
    write_csv(
        df,
        VALIDATION_DIR
        / scenario
        / "delta_from_base_python_vs_powerworld.csv",
    )
    summary = change_summary(f"{scenario} vs base", df)
    print_summary(summary)
    return summary


def validate_a1_a2() -> dict:
    df = compare_a2_minus_a1()
    write_csv(df, VALIDATION_DIR / "A2_vs_A1" / "python_vs_powerworld.csv")
    summary = change_summary("A2 vs A1", df)
    print_summary(summary)
    return summary


def validate_all() -> None:
    summaries = [validate_state(state) for state in E0_STATES]
    summaries += [validate_change_from_base(s) for s in E0_SCENARIOS]
    summaries.append(validate_a1_a2())
    write_csv(pd.DataFrame(summaries), VALIDATION_DIR / "summary.csv")


def main() -> None:
    started = manifest.now()
    for state in E0_STATES:
        print(f"Solving {state}")
        save_state(state, solve_state(state))
    validate_all()
    manifest.write(
        VALIDATION_DIR, "validate", started=started,
        inputs=[CASE_FILE] + [POWERWORLD_ROOT / s / "shortcircuit.csv" for s in E0_STATES],
        outputs=sorted(VALIDATION_DIR.rglob("*.csv")))


if __name__ == "__main__":
    main()
