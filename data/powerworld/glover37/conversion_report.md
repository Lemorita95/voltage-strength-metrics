# PowerWorld → SCC JSON conversion report (v2)

## Case-level assumptions

- System base: **100 MVA**.
- Slack bus read from CSV: **31**.
- Transformer phase shift: **0° for all transformers** (user-verified for this case).
- PowerWorld automatic switched-shunt and transformer-tap controls are **not assumed frozen**. The JSON contains the exported snapshot values. If PowerWorld controls move in another scenario, the resulting model states should be recorded when comparing results.

## Conversion warnings and notes

- **Warning:** Transformer 54-53 ckt 1: ControlType LTC without a regulated bus (RegBusNum 0); exported without a tap controller, so its tap stays at m0.

## Schema audit

**No required field in the current executable JSON schema remains missing for this 37-bus case.**

- Branch/transformer: R, X, G, B, tap, terminals, circuit and status are available.
- Generator: status, MVABase, GenR, GenX, StepR, StepX, StepTap, P, voltage setpoint and Q limits are available.
- Bus: explicit Slack/BusCat fields are available; the JSON continues to mark only the slack bus explicitly, matching the existing 14-bus case structure.
- Load: P, Q and status are available.

## Generator equivalent impedances

`GenR/GenX` are converted from generator MVA base to the 100-MVA system base, then the internal step-up `StepR/StepX` is included in the equivalent series impedance for this case. All generator MVA bases and StepTap values in the supplied case are 100 MVA and 1.0 respectively.

| Bus-ID | GenR | GenX | StepR | StepX | R_eq | X_eq |
|---|---:|---:|---:|---:|---:|---:|
| 28-1 | 0.00000 | 0.05000 | 0.00000 | 0.00000 | 0.00000 | 0.05000 |
| 14-1 | 0.00163 | 0.09500 | 0.00000 | 0.00000 | 0.00163 | 0.09500 |
| 31-1 | 0.00000 | 0.05000 | 0.00000 | 0.00000 | 0.00000 | 0.05000 |
| 44-1 | 0.00400 | 0.12700 | 0.00000 | 0.00000 | 0.00400 | 0.12700 |
| 28-2 | 0.00000 | 0.05000 | 0.00000 | 0.00000 | 0.00000 | 0.05000 |
| 50-1 | 0.00350 | 0.20000 | 0.00000 | 0.00000 | 0.00350 | 0.20000 |
| 53-1 | 0.00132 | 0.10000 | 0.00000 | 0.00000 | 0.00132 | 0.10000 |
| 54-1 | 0.00112 | 0.08800 | 0.00000 | 0.00000 | 0.00112 | 0.08800 |
| 48-1 | 0.00156 | 0.11300 | 0.00000 | 0.01500 | 0.00156 | 0.12800 |

## Transformer exported tap states

These are the tap ratios contained in the exported snapshot. PowerWorld may move controllable taps when solving another operating condition.

Blank cells were not in the export and use the `Transformer` class defaults.

| From | To | Ckt | Control | Tap m | m min | m max | m step | Reg bus | V low | V high | Auto |
|---:|---:|:---:|:---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 12 | 40 | 1 | LTC | 1.03125 | 0.90000 | 1.10000 | 0.00625 | 12 | 1.0150 | 1.0250 | yes |
| 12 | 40 | 2 | LTC | 1.03125 | 0.90000 | 1.10000 | 0.00625 | 12 | 1.0150 | 1.0250 | yes |
| 10 | 39 | 1 | LTC | 1.01250 | 0.90000 | 1.10000 | 0.00625 | 10 | 1.0150 | 1.0250 | yes |
| 54 | 53 | 1 | LTC | 1.00000 | 0.99000 | 1.10000 | 0.00625 |  |  |  |  |
| 48 | 47 | 1 | LTC | 1.05000 | 0.90000 | 1.10000 | 0.00625 | 48 | 1.0150 | 1.0250 | no |
| 44 | 41 | 2 | LTC | 1.02125 | 0.99000 | 1.10000 | 0.00625 | 44 | 1.0000 | 1.0200 | no |
| 44 | 41 | 1 | LTC | 1.02125 | 0.99000 | 1.10000 | 0.00625 | 44 | 1.0000 | 1.0200 | no |
| 39 | 38 | 2 | Fixed | 1.00000 | 0.51000 | 1.50000 | 0.00625 |  |  |  |  |
| 39 | 38 | 1 | Fixed | 1.00000 | 0.51000 | 1.50000 | 0.00625 |  |  |  |  |
| 33 | 32 | 1 | LTC | 1.00000 | 0.90000 | 1.10000 | 0.00625 | 33 | 0.9900 | 1.0000 | no |
| 35 | 31 | 1 | LTC | 1.00000 | 0.90000 | 1.10000 | 0.00625 | 35 | 0.9900 | 1.0000 | no |
| 28 | 29 | 2 | LTC | 1.00000 | 0.90000 | 1.10000 | 0.00625 | 28 | 0.9900 | 1.0000 | no |
| 28 | 29 | 1 | LTC | 1.00000 | 0.90000 | 1.10000 | 0.00625 | 28 | 0.9900 | 1.0000 | no |
| 1 | 40 | 1 | LTC | 1.00000 | 0.90000 | 1.10000 | 0.00625 | 1 | 0.9900 | 1.0000 | no |

## Switched shunts represented as fixed shunts

The JSON freezes each shunt at the **exported `MvarNom` switching position**:

\[
B_{pu}=\frac{Q_{nom}}{100\ \mathrm{MVA}}
\]

It does not reproduce PowerWorld's switching controller. `Mvar` is not used because it already reflects the local-voltage squared effect; `MvarNomMax` is not used because it is the available maximum rather than the exported switching position.

| Bus-ID | PW mode | PW auto | MvarNom | Fixed B pu |
|---|---|---|---:|---:|
| 13-1 | Discrete | YES | 8.000 | 0.08000 |
| 14-1 | Discrete | YES | 7.200 | 0.07200 |
| 15-1 | Discrete | YES | 12.600 | 0.12600 |
| 16-1 | Discrete | YES | 28.800 | 0.28800 |
| 17-1 | Discrete | YES | 15.600 | 0.15600 |
| 18-1 | Fixed | YES | 18.000 | 0.18000 |
| 20-1 | Discrete | YES | 7.200 | 0.07200 |
| 44-1 | Discrete | YES | 0.000 | 0.00000 |

## Open elements omitted from active topology

- Branch: Line 20 → 48, circuit 1.
- No open loads.

## Comparison caveat

If PowerWorld automatic tap or switched-shunt controls move after a scenario change while the SCC JSON continues to use an earlier exported position, the two short-circuit models no longer have identical network admittances. Report the resulting tap/shunt-state differences alongside the power-flow mismatch; do not interpret the full SCC discrepancy as solver error alone.
