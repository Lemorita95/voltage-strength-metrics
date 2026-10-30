# System-Strength Screening Metrics over Time

Reproducible code, public data, and results for a time-ordered comparison of four system-strength screening metrics (SCR, WSCR, ESCR, SDSCR) at three IBR sites added to the Glover 37-bus benchmark, over a 168-hour week and the full N-1 line set.

Continues [`Lemorita95/spatial-short-circuit-level-field`](https://github.com/Lemorita95/spatial-short-circuit-level-field) at commit `3fed033`, whose history holds the earlier development and the previous results.

## Reproduce

Developed with Python 3.14.

```bash
pip install -r requirements.txt
make strength
```

Results are not in git: run `make strength` first (about 8 min) before `make report` or `make explorer`. Every stage writes a `manifest.json` (commit, uncommitted-changes flag, configuration, SHA-256 of inputs and outputs); a run is of record only from a clean, committed tree.

`make strength` runs the stages in order (about 8 min); each is also a target of its own:

```bash
make validate          # E0: PowerWorld agreement
make screen            # N-1 line set
make simulate          # 168 h metrics run: every state and ESCR MIIF variant
make analyze
make report
```

Run the checks, and browse the results:

```bash
make test
make explorer
```

`make explorer` serves the repository on localhost and opens [`tools/explorer.html`](tools/explorer.html), which reads `results/glover37/raw/` each time it loads (stop with Ctrl-C). The explorer needs internet to load its chart library.

Two variables can be set on any target: `PYTHON` (for example `PYTHON=.venv/bin/python`) and `VOLTAGE_PRESET`, which selects the voltage band used to flag out-of-band operating states.

## Repository map

- [`models/`](models/): network model, Newton–Raphson power flow (with generator reactive limits and tap controllers), short-circuit solver, voltage limits, and the system-strength metrics
- [`experiments/glover37/`](experiments/glover37/): one module per stage
  - [`config.py`](experiments/glover37/config.py): every study parameter
  - [`network.py`](experiments/glover37/network.py): week, dispatch, outages, IBR augmentation
  - [`simulate.py`](experiments/glover37/simulate.py), [`analyze.py`](experiments/glover37/analyze.py), [`report.py`](experiments/glover37/report.py)
  - [`validate.py`](experiments/glover37/validate.py): PowerWorld agreement; [`pw2json.py`](experiments/glover37/pw2json.py) builds the case it validates
  - [`layout.py`](experiments/glover37/layout.py): network diagram from the PowerWorld display file
- [`cases/glover37.json`](cases/glover37.json): benchmark network model, generated from the PowerWorld exports
- [`data/powerworld/glover37/`](data/powerworld/glover37/): PowerWorld reference data
- [`data/se3_2025.csv`](data/se3_2025.csv): public temporal scaling signal
- `results/glover37/`: generated results (git-ignored)
- [`figures/glover37/`](figures/glover37/): generated figures
- [`tools/`](tools/): the results explorer
- [`tests/`](tests/): targeted checks

The SE3 demand series is used only as a temporal scaling signal; the Glover 37-bus benchmark is not an electrical model of SE3.

License: [`LICENSE`](LICENSE)
