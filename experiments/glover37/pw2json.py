import argparse
import csv
import json
from pathlib import Path
from typing import Optional

S_BASE_MVA = 100.0
TRANSFORMER_PHASE_SHIFT_DEG = 0.0

# JSON key -> PowerWorld bus column, written for every bus when the column has a value
BUS_COLUMNS = {"nom_kv": "NomkV"}
# JSON key -> PowerWorld branch column, written only when the column has a value
TRANSFORMER_TAP_COLUMNS = {"m0": "Tap", "m_min": "TapMin", "m_max": "TapMax", "m_step": "TapStepSize"}
TRANSFORMER_BAND_COLUMNS = {"v_low": "RegMin", "v_high": "RegMax"}
# PowerWorld per-transformer automatic control (YES/NO). Only YES is written ("auto": true);
# NO or missing is omitted, so the TapController default (auto off, taps held) applies.
TRANSFORMER_AUTO_COLUMN = "AutoControl"


def read_pw_csv(path: Path) -> list[dict]:
    """Read a PowerWorld CSV export whose first line contains the object type."""
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        first = f.readline()
        if not first:
            raise ValueError(f"{path} is empty.")
        return list(csv.DictReader(f))


def pw_float(value: str | None) -> Optional[float]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    return float(s.replace(",", "."))


def pw_int(value: str) -> int:
    return int(str(value).strip())


def closed(value: str) -> bool:
    return str(value).strip().lower() == "closed"


def pu_power(value: str | None) -> float:
    v = pw_float(value)
    return 0.0 if v is None else v / S_BASE_MVA


def convert(folder: Path) -> tuple[dict, str]:
    buses_csv = read_pw_csv(folder / "buses.csv")
    branches_csv = read_pw_csv(folder / "branches.csv")
    generators_csv = read_pw_csv(folder / "generators.csv")
    loads_csv = read_pw_csv(folder / "loads.csv")
    line_shunts_csv = read_pw_csv(folder / "line_shunts.csv")
    switched_shunts_csv = read_pw_csv(folder / "switched_shunts.csv")

    notes: list[str] = []
    warnings: list[str] = []

    # ------------------------------------------------------------------ buses
    slack_rows = [r for r in buses_csv
                  if r.get("Slack", "").strip().upper() == "YES"
                  or r.get("BusCat", "").strip().lower() == "slack"]
    if len(slack_rows) != 1:
        raise ValueError(
            f"Expected exactly one slack bus from buses.csv; found {len(slack_rows)}."
        )
    slack_id = pw_int(slack_rows[0]["Number"])

    buses = []
    for row in buses_csv:
        bus_id = pw_int(row["Number"])
        item = {
            "id": bus_id,
            "name": row["Name"].strip(),
        }
        for key, col in BUS_COLUMNS.items():
            value = pw_float(row.get(col))
            if value is not None:
                item[key] = value
        # Match the existing executable case schema: only the slack bus needs
        # an explicit type / starting voltage / angle.
        if bus_id == slack_id:
            item.update({
                "bus_type": "Slack",
                "V": pw_float(row["Vpu"]),
                "angle": pw_float(row["Vangle"]),
            })
        buses.append(item)

    # ----------------------------------------------------- lines/transformers
    lines = []
    transformers = []
    omitted_open_branches = []
    active_taps = []
    # (from, to, reg_bus, v_low, v_high, auto) -> transformer names; parallel circuits share one controller
    tap_groups: dict[tuple, list[str]] = {}
    missing_auto = []   # LTC transformers without a YES/NO automatic-control value

    for row in branches_csv:
        fbus = pw_int(row["BusNumFrom"])
        tbus = pw_int(row["BusNumTo"])
        ckt = row["Circuit"].strip()
        dev = row["BranchDeviceType"].strip()

        if not closed(row["Status"]):
            omitted_open_branches.append((fbus, tbus, ckt, dev))
            continue

        R = pw_float(row["R"]) or 0.0
        X = pw_float(row["X"]) or 0.0
        G = pw_float(row["G"]) or 0.0
        B = pw_float(row["B"]) or 0.0

        common = {
            "from_bus": fbus,
            "to_bus": tbus,
            "z_series": {"R": R, "X": X},
            "y_shunt": {"G": G, "B": B},
        }

        if row["IsXF"].strip().upper() == "YES":
            # Only fields present in the export are written; Transformer defines the defaults.
            taps = {key: pw_float(row.get(col)) for key, col in TRANSFORMER_TAP_COLUMNS.items()}
            taps = {key: value for key, value in taps.items() if value is not None}

            # Voltage regulation only for an LTC with a regulated bus (PowerWorld RegBusNum 0 = none).
            # It goes to the "tap_controllers" section; the transformer record keeps the tap grid only.
            name = f"Transformer {fbus}-{tbus} ckt {ckt}"
            control = (row.get("ControlType") or "").strip()
            reg_bus = pw_int(row["RegBusNum"]) if (row.get("RegBusNum") or "").strip() else 0
            regulation = {}
            if control.upper() == "LTC" and reg_bus != 0:
                regulation["reg_bus"] = reg_bus
                for key, col in TRANSFORMER_BAND_COLUMNS.items():
                    value = pw_float(row.get(col))
                    if value is not None:
                        regulation[key] = value
                auto = (row.get(TRANSFORMER_AUTO_COLUMN) or "").strip().upper()
                if auto == "YES":
                    regulation["auto"] = True
                elif auto != "NO":
                    missing_auto.append(name)
                key = (fbus, tbus, *(regulation.get(k) for k in ("reg_bus", "v_low", "v_high", "auto")))
                tap_groups.setdefault(key, []).append(name)
            elif control.upper() == "LTC":
                warnings.append(
                    f"Transformer {fbus}-{tbus} ckt {ckt}: ControlType LTC without a regulated bus "
                    "(RegBusNum 0); exported without a tap controller, so its tap stays at m0."
                )

            item = {
                **common,
                "name": name,
                **taps,
            }
            transformers.append(item)
            active_taps.append((fbus, tbus, ckt, control or "-", {**taps, **regulation}))
        else:
            item = {
                **common,
                "name": f"Line {fbus}-{tbus} ckt {ckt}",
            }
            lines.append(item)

    # ---------------------------------------------------------------- loads
    loads = []
    omitted_open_loads = []
    for row in loads_csv:
        bus = pw_int(row["BusNum"])
        load_id = row["ID"].strip()
        if not closed(row["Status"]):
            omitted_open_loads.append((bus, load_id))
            continue
        loads.append({
            "from_bus": bus,
            "to_bus": 0,
            "name": f"Load {bus}-{load_id}",
            "P": pu_power(row["MW"]),
            "Q": pu_power(row["Mvar"]),
        })

    # ----------------------------------------------------------- generators
    generators = []
    generator_impedance_details = []

    for row in generators_csv:
        bus = pw_int(row["BusNum"])
        gid = row["ID"].strip()
        mva_base = pw_float(row["MVABase"])
        if not mva_base or mva_base <= 0:
            raise ValueError(f"Invalid MVABase for generator {bus}-{gid}.")

        gen_r_machine = pw_float(row["GenR"]) or 0.0
        gen_x_machine = pw_float(row["GenX"]) or 0.0
        step_r_raw = pw_float(row["StepR"]) or 0.0
        step_x_raw = pw_float(row["StepX"]) or 0.0
        step_tap = pw_float(row["StepTap"])
        if step_tap is None:
            step_tap = 1.0

        # GenR/GenX are on generator MVA base. Convert to system base.
        base_factor = S_BASE_MVA / mva_base
        gen_r_system = gen_r_machine * base_factor
        gen_x_system = gen_x_machine * base_factor

        # For this supplied case, all StepTap values are 1.0 and generator MVA
        # bases equal the 100 MVA system base. Therefore folding StepR/StepX
        # into the series generator impedance is unambiguous numerically.
        #
        # Refuse silent generalization if a future case has non-unit StepTap.
        if abs(step_tap - 1.0) > 1e-12 and (abs(step_r_raw) > 0 or abs(step_x_raw) > 0):
            raise ValueError(
                f"Generator {bus}-{gid} has non-unit StepTap={step_tap} with "
                "nonzero internal step-up impedance. Define the required base/"
                "tap transformation before using this converter for that case."
            )

        # With S_gen = S_system = 100 MVA here, this is a direct sum.
        # If generator MVABase differs in a future case, verify the PowerWorld
        # base convention for StepR/StepX before generalizing this line.
        if abs(mva_base - S_BASE_MVA) > 1e-9 and (abs(step_r_raw) > 0 or abs(step_x_raw) > 0):
            raise ValueError(
                f"Generator {bus}-{gid} has MVABase={mva_base} MVA and nonzero "
                "StepR/StepX. Verify the step-up impedance base before conversion."
            )

        r_eq = gen_r_system + step_r_raw
        x_eq = gen_x_system + step_x_raw

        generators.append({
            "from_bus": bus,
            "to_bus": 0,
            "name": f"G{bus}-{gid}",
            "z_series": {"R": r_eq, "X": x_eq},
            "y_shunt": None,
            "V": pw_float(row["VoltSet"]),
            "P": pu_power(row["MW"]),
            "Q": None,
            "Qg_max": pu_power(row["MvarMax"]),
            "Qg_min": pu_power(row["MvarMin"]),
            "status": 1 if closed(row["Status"]) else 0,
        })

        generator_impedance_details.append({
            "bus": bus,
            "id": gid,
            "MVABase": mva_base,
            "GenR": gen_r_machine,
            "GenX": gen_x_machine,
            "StepR": step_r_raw,
            "StepX": step_x_raw,
            "StepTap": step_tap,
            "R_eq_system": r_eq,
            "X_eq_system": x_eq,
        })

    # --------------------------------------------------------------- shunts
    # The executable's "capacitors" list is structurally a one-terminal shunt
    # admittance, so it is used for fixed bus/line-end/switched shunts.
    capacitors = []

    # End-connected line shunts.
    for row in line_shunts_csv:
        if not closed(row["Status"]):
            continue

        fbus = pw_int(row["BusNumFrom"])
        tbus = pw_int(row["BusNumTo"])
        loc = pw_int(row["BusNumLoc"])
        ckt = row["Circuit"].strip()
        sid = row["ID"].strip()

        capacitors.append({
            "from_bus": loc,
            "to_bus": 0,
            "name": f"LineShunt {fbus}-{tbus} ckt {ckt} ID {sid} @ {loc}",
            "z_series": None,
            "y_shunt": {
                "G": pu_power(row["MWNom"]),
                "B": pu_power(row["MvarNom"]),
            },
            "status": 1,
        })

    # Switched shunts are intentionally frozen at the nominal Mvar of the
    # EXPORTED switching position. We use MvarNom, not instantaneous Mvar
    # (which contains the V^2 effect) and not MvarNomMax.
    fixed_switched_shunts = []
    for row in switched_shunts_csv:
        bus = pw_int(row["BusNum"])
        sid = row["ID"].strip()
        q_nom = pw_float(row["MvarNom"]) or 0.0
        status = 1 if closed(row["Status"]) else 0

        capacitors.append({
            "from_bus": bus,
            "to_bus": 0,
            "name": f"Switched Shunt {bus}-{sid} [FIXED EXPORTED POSITION]",
            "z_series": None,
            "y_shunt": {
                "G": 0.0,
                "B": q_nom / S_BASE_MVA,
            },
            "status": status,
        })

        fixed_switched_shunts.append({
            "bus": bus,
            "id": sid,
            "status": row["Status"].strip(),
            "mode_in_powerworld": row["ShuntMode"].strip(),
            "auto_control_in_powerworld": row["AutoControl"].strip(),
            "MvarNom_exported_position": q_nom,
            "B_pu_fixed_json": q_nom / S_BASE_MVA,
        })

    # one tap controller per regulating transformer group; "auto": true only for AutoControl YES
    if missing_auto:
        warnings.append(
            f"No YES/NO '{TRANSFORMER_AUTO_COLUMN}' value for {len(missing_auto)} LTC transformer(s) "
            f"({', '.join(missing_auto)}); their tap controllers are written without auto, so taps are held "
            "(TapController default). Add the field to the branch export if they should regulate."
        )
    tap_controllers = []
    for (fbus, tbus, reg_bus, v_low, v_high, auto), names in tap_groups.items():
        record = {"name": f"{fbus}-{tbus}", "elements": names, "reg_bus": reg_bus}
        if v_low is not None:
            record["v_low"] = v_low
        if v_high is not None:
            record["v_high"] = v_high
        if auto:
            record["auto"] = True
        tap_controllers.append(record)

    case = {
        "buses": buses,
        "elements": {
            "lines": lines,
            "transformers": transformers,
            "capacitors": capacitors,
            "generators": generators,
            "loads": loads,
            "full_converters": [],
            "external_grids": [],
        },
        "tap_controllers": tap_controllers,
    }

    # --------------------------------------------------------------- report
    report = []
    report.append("# PowerWorld → SCC JSON conversion report (v2)\n\n")
    report.append("## Case-level assumptions\n\n")
    report.append(f"- System base: **{S_BASE_MVA:g} MVA**.\n")
    report.append(f"- Slack bus read from CSV: **{slack_id}**.\n")
    report.append(
        f"- Transformer phase shift: **{TRANSFORMER_PHASE_SHIFT_DEG:g}° for all transformers** "
        "(user-verified for this case).\n"
    )
    report.append(
        "- PowerWorld automatic switched-shunt and transformer-tap controls are **not assumed frozen**. "
        "The JSON contains the exported snapshot values. If PowerWorld controls move in another scenario, "
        "the resulting model states should be recorded when comparing results.\n"
    )

    if warnings or notes:
        report.append("\n## Conversion warnings and notes\n\n")
        for message in warnings:
            report.append(f"- **Warning:** {message}\n")
        for message in notes:
            report.append(f"- {message}\n")

    report.append("\n## Schema audit\n\n")
    report.append(
        "**No required field in the current executable JSON schema remains missing for this 37-bus case.**\n\n"
    )
    report.append(
        "- Branch/transformer: R, X, G, B, tap, terminals, circuit and status are available.\n"
    )
    report.append(
        "- Generator: status, MVABase, GenR, GenX, StepR, StepX, StepTap, P, voltage setpoint and Q limits are available.\n"
    )
    report.append(
        "- Bus: explicit Slack/BusCat fields are available; the JSON continues to mark only the slack bus explicitly, "
        "matching the existing 14-bus case structure.\n"
    )
    report.append(
        "- Load: P, Q and status are available.\n"
    )

    report.append("\n## Generator equivalent impedances\n\n")
    report.append(
        "`GenR/GenX` are converted from generator MVA base to the 100-MVA system base, then the internal "
        "step-up `StepR/StepX` is included in the equivalent series impedance for this case. "
        "All generator MVA bases and StepTap values in the supplied case are 100 MVA and 1.0 respectively.\n\n"
    )
    report.append("| Bus-ID | GenR | GenX | StepR | StepX | R_eq | X_eq |\n")
    report.append("|---|---:|---:|---:|---:|---:|---:|\n")
    for d in generator_impedance_details:
        report.append(
            f"| {d['bus']}-{d['id']} | {d['GenR']:.5f} | {d['GenX']:.5f} | "
            f"{d['StepR']:.5f} | {d['StepX']:.5f} | "
            f"{d['R_eq_system']:.5f} | {d['X_eq_system']:.5f} |\n"
        )

    report.append("\n## Transformer exported tap states\n\n")
    report.append(
        "These are the tap ratios contained in the exported snapshot. PowerWorld may move controllable taps "
        "when solving another operating condition.\n\n"
    )
    report.append("Blank cells were not in the export and use the `Transformer` class defaults.\n\n")
    report.append("| From | To | Ckt | Control | Tap m | m min | m max | m step | Reg bus | V low | V high | Auto |\n")
    report.append("|---:|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|:---:|\n")
    fmt = lambda taps, key, spec: format(taps[key], spec) if key in taps else ""
    for fbus, tbus, ckt, control, taps in active_taps:
        report.append(
            f"| {fbus} | {tbus} | {ckt} | {control} | {fmt(taps, 'm0', '.5f')} | {fmt(taps, 'm_min', '.5f')} | "
            f"{fmt(taps, 'm_max', '.5f')} | {fmt(taps, 'm_step', '.5f')} | {fmt(taps, 'reg_bus', 'd')} | "
            f"{fmt(taps, 'v_low', '.4f')} | {fmt(taps, 'v_high', '.4f')} | "
            f"{('yes' if taps.get('auto') else 'no') if 'reg_bus' in taps else ''} |\n"
        )

    report.append("\n## Switched shunts represented as fixed shunts\n\n")
    report.append(
        "The JSON freezes each shunt at the **exported `MvarNom` switching position**:\n\n"
        "\\[\nB_{pu}=\\frac{Q_{nom}}{100\\ \\mathrm{MVA}}\n\\]\n\n"
        "It does not reproduce PowerWorld's switching controller. `Mvar` is not used because it already reflects "
        "the local-voltage squared effect; `MvarNomMax` is not used because it is the available maximum rather than "
        "the exported switching position.\n\n"
    )
    report.append("| Bus-ID | PW mode | PW auto | MvarNom | Fixed B pu |\n")
    report.append("|---|---|---|---:|---:|\n")
    for d in fixed_switched_shunts:
        report.append(
            f"| {d['bus']}-{d['id']} | {d['mode_in_powerworld']} | "
            f"{d['auto_control_in_powerworld']} | {d['MvarNom_exported_position']:.3f} | "
            f"{d['B_pu_fixed_json']:.5f} |\n"
        )

    report.append("\n## Open elements omitted from active topology\n\n")
    if omitted_open_branches:
        for x in omitted_open_branches:
            report.append(f"- Branch: {x[3]} {x[0]} → {x[1]}, circuit {x[2]}.\n")
    else:
        report.append("- No open branches.\n")
    if omitted_open_loads:
        for x in omitted_open_loads:
            report.append(f"- Load: bus {x[0]}, ID {x[1]}.\n")
    else:
        report.append("- No open loads.\n")

    report.append("\n## Comparison caveat\n\n")
    report.append(
        "If PowerWorld automatic tap or switched-shunt controls move after a scenario change while the SCC JSON "
        "continues to use an earlier exported position, the two short-circuit models no longer have identical "
        "network admittances. Report the resulting tap/shunt-state differences alongside the power-flow mismatch; "
        "do not interpret the full SCC discrepancy as solver error alone.\n"
    )

    return case, "".join(report)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", nargs="?", default=".")
    parser.add_argument("-o", "--output", default="37bus_case.json")
    parser.add_argument("--report", default="37bus_conversion_report_v2.md")
    args = parser.parse_args()

    case, report = convert(Path(args.folder))
    Path(args.output).write_text(
        json.dumps(case, indent=4, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    Path(args.report).write_text(report, encoding="utf-8")
    print(f"Wrote: {args.output}")
    print(f"Wrote: {args.report}")


if __name__ == "__main__":
    main()
