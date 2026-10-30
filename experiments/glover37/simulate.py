"""Simulation stage: N-1 screening and the temporal SCR/WSCR/ESCR/SDSCR run.

`screen` fixes the N-1 set before any metric is computed: a line outage is
included only if it islands no bus and both pre-fault power flows (37-bus base,
46-bus with IBRs) converge in every hour of the week, with commitment and
setpoints held at the intact schedule. No strength metric is evaluated there.

`run` evaluates the four metrics, hour by hour, for every operating state:
the intact network and every included line outage under the sequential
commitment rule (E1, E2), and the intact network with every unit online (E5).
ESCR is evaluated with every MIIF variant (config ESCR_VARIANTS; E3).

Run from repository root:
    python -m experiments.glover37.simulate screen
    python -m experiments.glover37.simulate screen --hours 2        # smoke test
    python -m experiments.glover37.simulate run --hours 2
    python -m experiments.glover37.simulate run \
        --contingencies results/glover37/raw/contingency_set.csv
    # intact + line 39-47 only:
    python -m experiments.glover37.simulate run \
        --contingencies results/glover37/raw/contingency_set.csv --case line_39_47_out

Outputs under results/glover37/raw/ by default. Every row carries its state
(hour_index, timestamp, case, commitment); metric rows also carry `variant`
(ESCR's MIIF construction, blank for the other metrics).
- `contingency_set.csv` (screen): one row per line;
- `metrics_timeseries.csv`: one row per metric / variant / evaluation bus / state;
- `metric_variables.csv`: the underlying voltage, SCMVA, impedance, MIIF and power terms;
- `run_summary.csv`, `dispatch_schedule.csv`: per-state audit (incl. slack power) and dispatch;
- `manifest.json`: commit, uncommitted-changes flag, config, SHA-256 and timings.
"""
from __future__ import annotations

import argparse
import contextlib
import functools
import io
import os
import shutil
import tempfile
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from models.PowerFlow import PowerFlow
from models.SystemStrength import SCR, WSCR, ESCR, SDSCR, fault_scmva
from models.VoltageLimits import PRESETS, VoltageLimits
from experiments.glover37 import manifest
from experiments.glover37.config import (
    ALL_ONLINE, AUDITED_PROBE, CASE_FILE, GEN_LIMITS_FILE, IBR_MW, IF_STEPS, IMPEDANCE_VARIANT,
    INTACT, RAW_DIR, REFERENCE_VARIANT, S_BASE_MVA, SE3_FILE, SEQUENTIAL, SITES, SLACK_BUS,
    VOLTAGE_PRESET, WINDOW_HOURS, probe_variant,
)
from experiments.glover37.network import (
    augment_network, baseline_load_mw, operating_state_audit, prepare_network,
    read_gen_limits, read_se3, schedule, select_week, sequential_dispatch,
)

CONTINGENCY_FILE = RAW_DIR / 'contingency_set.csv'
RUN_FILES = ('metrics_timeseries.csv', 'metric_variables.csv', 'run_summary.csv',
             'dispatch_schedule.csv')
# Worker processes; any count gives the same output (see parallel_map).
WORKERS = os.cpu_count() or 1


def load_week(se3_file=SE3_FILE, hours=WINDOW_HOURS):
    return select_week(read_se3(se3_file)).iloc[:hours]


@contextlib.contextmanager
def parallel_map(fn, items, workers):
    """Results of fn over items, in input order. Every item is computed independently,
    so the output does not depend on `workers`; workers=1 runs in this process."""
    if workers <= 1:
        yield map(fn, items)
        return
    # One BLAS thread per worker process; the workers are the parallelism.
    for var in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                'VECLIB_MAXIMUM_THREADS'):
        os.environ.setdefault(var, '1')
    pool = ProcessPoolExecutor(max_workers=workers)
    try:
        yield pool.map(fn, items, chunksize=max(1, len(items) // (8 * workers)))
    finally:
        pool.shutdown(cancel_futures=True)


@functools.cache
def schedule_inputs():
    """(generator limits, baseline load MW), read once per process."""
    return read_gen_limits(GEN_LIMITS_FILE), baseline_load_mw()


# =============================================================================
# 1. N-1 screening
# =============================================================================

def case_labels(lines):
    """{line key: case label}; the circuit number is kept only for parallel lines."""
    pairs = Counter(tuple(sorted((ln.bus_1, ln.bus_2))) for ln in lines.values())
    labels = {}
    for key, ln in lines.items():
        a, b = sorted((ln.bus_1, ln.bus_2))
        ckt = key.rsplit('ckt', 1)[-1].strip()
        labels[key] = f'line_{a}_{b}_ckt{ckt}_out' if pairs[(a, b)] > 1 else f'line_{a}_{b}_out'
    return labels


def islands(net, outaged):
    """True if removing `outaged` leaves any bus disconnected from the rest."""
    adjacency = {b: set() for b in net.buses}
    for key, branch in (net.lines | net.transformers).items():
        if key == outaged or not branch.status:
            continue
        adjacency[branch.bus_1].add(branch.bus_2)
        adjacency[branch.bus_2].add(branch.bus_1)
    start = next(iter(adjacency))
    seen, stack = {start}, [start]
    while stack:
        for other in adjacency[stack.pop()] - seen:
            seen.add(other)
            stack.append(other)
    return len(seen) != len(adjacency)


def converges(net):
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        return PowerFlow().solve_power_flow(net) is not None


@functools.cache
def reference_network():
    """Intact 37-bus network at load scale 1, the source of the candidate line set."""
    limits, base_load_mw = schedule_inputs()
    return prepare_network(1.0, sequential_dispatch(base_load_mw, limits), limits)


def screen_line(line, week):
    """One row of contingency_set.csv: islanding, then convergence in every hour."""
    limits, base_load_mw = schedule_inputs()
    reference = reference_network()
    ln = reference.lines[line]
    row = {'line': line, 'case': case_labels(reference.lines)[line],
           'from_bus': ln.bus_1, 'to_bus': ln.bus_2,
           'islands': islands(reference, line), 'n_hours': len(week),
           'n_converged_base': 0, 'n_converged_ibr': 0, 'failed_hours': ''}
    if not row['islands']:
        failed = []
        for r in week.itertuples(index=False):
            demand = base_load_mw * float(r.load_scale)
            ok_base = converges(prepare_network(
                r.load_scale, sequential_dispatch(demand, limits), limits, line))
            aug_net = prepare_network(r.load_scale, sequential_dispatch(demand - IBR_MW, limits),
                                      limits, line)
            augment_network(aug_net)  # rebuilds the Y-bus
            ok_ibr = converges(aug_net)
            row['n_converged_base'] += ok_base
            row['n_converged_ibr'] += ok_ibr
            if not (ok_base and ok_ibr):
                failed.append(int(r.hour_index))
        row['failed_hours'] = ';'.join(map(str, failed))
    row['included'] = (not row['islands'] and row['n_converged_base'] == len(week)
                       and row['n_converged_ibr'] == len(week))
    row['reason'] = ('islands a bus' if row['islands'] else
                     'power flow fails in some hours' if not row['included'] else '')
    return row


def screen(week, workers=WORKERS):
    lines = list(reference_network().lines)
    rows = []
    with parallel_map(functools.partial(screen_line, week=week), lines, workers) as results:
        for k, row in enumerate(results, 1):
            print(f"[{k:02d}/{len(lines)}] {row['case']}: "
                  f"{'included' if row['included'] else 'excluded (' + row['reason'] + ')'}",
                  flush=True)
            rows.append(row)
    return pd.DataFrame(rows)


def read_contingencies(path, cases=None):
    """[(case label, line key)] for the included rows of contingency_set.csv."""
    table = pd.read_csv(path)
    table = table[table['included'].astype(bool)]
    if cases:
        if unknown := set(cases) - set(table['case']):
            raise ValueError(f'Cases not in the included contingency set: {sorted(unknown)}')
        table = table[table['case'].isin(cases)]
    return list(zip(table['case'], table['line']))


# =============================================================================
# 2. Temporal metrics run
# =============================================================================

def solve_pf(network):
    '''complex bus voltages and the PowerFlow instance (generator states, control records).'''
    pf = PowerFlow()
    solved = pf.solve_power_flow(network)
    if solved is None:
        raise RuntimeError('Power flow failed')
    theta, magnitude = solved
    return magnitude * np.exp(1j * theta), pf


def slack_p_mw(net, v):
    """Active power of the slack bus's generation in the solved state, MW."""
    v = np.asarray(v, dtype=complex)
    pq_gen = PowerFlow().bus_flow(net, np.r_[np.angle(v), np.abs(v)])[1]
    return float(pq_gen[0, net.idx[SLACK_BUS]]) * S_BASE_MVA


def scalar_components(value):
    """Serialize a complex number as numeric real/imag/magnitude fields."""
    z = complex(value)
    return {'real': float(z.real), 'imag': float(z.imag), 'magnitude': float(abs(z))}


def append_record(rows, *, key, metric, variant, bus, variable, value, related_bus=None, unit=''):
    numeric = scalar_components(value)
    rows.append({
        **key, 'metric': metric, 'variant': variant, 'bus': bus, 'related_bus': related_bus,
        'variable': variable, 'unit': unit, **numeric,
    })


def emit_metric(rows, details, metric_name, record, key, variant=''):
    """Rows of one metric record. key: hour_index, timestamp, case, commitment."""
    values = record['value']
    items = values.items() if isinstance(values, dict) else [(None, values)]
    for bus, value in items:
        rows.append({
            **key, 'metric': metric_name, 'variant': variant,
            'bus': bus if bus is not None else 'group', 'value': float(value),
        })
    term = functools.partial(append_record, details, key=key, metric=metric_name, variant=variant)
    # Keep the inputs and intermediate terms needed to explain temporal changes.
    for name, unit in [('scmva', 'MVA'), ('rated_mw', 'MW'),
                       ('probe_reactive_power_mvar', 'MVAr'),
                       ('probe_pv_changes', 'count'),
                       ('target_voltage_pu', 'pu'), ('actual_voltage_pu', 'pu'),
                       ('z_ii', 'pu'), ('v_pf', 'pu')]:
        for bus, val in record.get(name, {}).items():
            term(bus=bus, variable=name, value=val, unit=unit)
    for name in ('if_magnitude', 'w_complex'):
        for (bus, other), val in record.get(name, {}).items():
            term(bus=bus, related_bus=other, variable=name, value=val, unit='ratio')
    for disturbed, responses in record.get('delta_v_pu', {}).items():
        for measured, dv in responses.items():
            term(bus=disturbed, related_bus=measured, variable='delta_v_pu', value=dv, unit='pu')
    if 'z_rr' in record:
        buses = tuple(record['rated_mw'])
        for i, bus in enumerate(buses):
            for j, other in enumerate(buses):
                term(bus=bus, related_bus=other, variable='z_rr', value=record['z_rr'][i, j],
                     unit='pu')


# Terms written per ESCR variant: the reference carries the state's shared terms, the
# audited probe its audit terms, every other variant only its value and MIIF.
ESCR_TERMS = {
    REFERENCE_VARIANT: ('value', 'if_magnitude', 'scmva', 'rated_mw', 'z_ii', 'v_pf'),
    AUDITED_PROBE: ('value', 'if_magnitude', 'probe_reactive_power_mvar', 'probe_pv_changes',
                    'target_voltage_pu', 'actual_voltage_pu', 'delta_v_pu'),
}


def variant_name(key):
    return IMPEDANCE_VARIANT if key == 'impedance' else probe_variant(key)


def evaluate_state(task, voltage_limits):
    """All outputs of one operating state (hour, commitment rule, case), computed
    independently of every other state. On failure, `error` is set and the outputs hold
    what was computed."""
    h, timestamp, scale, commitment, case, line = task
    limits, base_load_mw = schedule_inputs()
    demand = base_load_mw * scale
    # Crucial: independent schedules for the two networks, same commitment rule.
    base_dispatch, base_on = schedule(commitment, demand, limits)
    ibr_dispatch, ibr_on = schedule(commitment, demand - IBR_MW, limits)
    rated = {
        'SCR': {site.poi_bus: site.pmax_mw for site in SITES},
        'ESCR': {site.escr_bus: site.pmax_mw for site in SITES},
        'SDSCR': {site.sdscr_bus: site.pmax_mw for site in SITES},
    }
    metrics = {
        'SCR': SCR(rated['SCR']), 'WSCR': WSCR(rated['SCR']),
        'ESCR': ESCR(rated['ESCR'], probe_steps=IF_STEPS, impedance=True),
        'SDSCR': SDSCR(rated['SDSCR']),
    }
    key = {'hour_index': h, 'timestamp': str(timestamp), 'case': case, 'commitment': commitment}
    out = {'results': [], 'variables': [], 'dispatch': [], 'error': None, 'switches': {}}
    for network_name, dispatch, online in [('base', base_dispatch, base_on),
                                           ('ibr', ibr_dispatch, ibr_on)]:
        for gen in limits:
            out['dispatch'].append({
                **key, 'network': network_name, 'generator': gen.key,
                'bus': gen.bus, 'committed': int(online[gen.key]),
                'dispatch_MW': dispatch[gen.key],
                'system_demand_MW': demand,
                'ibr_injection_MW': IBR_MW if network_name == 'ibr' else 0,
            })
    base_net = prepare_network(scale, base_dispatch, limits, line, base_on)
    aug_net = prepare_network(scale, ibr_dispatch, limits, line, ibr_on)
    augment_network(aug_net)
    step = f'{commitment}:{case}:{h}'
    state = out['state'] = {
        **key, 'load_scale': scale, 'system_demand_MW': demand,
        'ibr_injection_MW': IBR_MW,
        'base_non_slack_MW': sum(base_dispatch.values()),
        'ibr_case_non_slack_MW': sum(ibr_dispatch.values()),
        'base_n_committed': sum(base_on.values()),
        'ibr_case_n_committed': sum(ibr_on.values()),
        'base_pf_ok': False, 'ibr_pf_ok': False,
        'error': '',
    }
    try:
        base_v, base_pf = solve_pf(base_net)
        state['base_pf_ok'] = True
        state.update(operating_state_audit(base_net, base_pf, np.abs(base_v),
                                           voltage_limits, prefix='base_'))
        aug_v, aug_pf = solve_pf(aug_net)
        state['ibr_pf_ok'] = True
        state.update(operating_state_audit(aug_net, aug_pf, np.abs(aug_v),
                                           voltage_limits, prefix='ibr_'))
        for prefix, net, v in [('base_', base_net, base_v), ('ibr_', aug_net, aug_v)]:
            state[f'{prefix}slack_P_MW'] = p_slack = slack_p_mw(net, v)
            state[f'{prefix}slack_exporting'] = p_slack < 0

        def evaluate(name, *args, **kwargs):
            try:
                return metrics[name].evaluate(step, *args, **kwargs)
            except Exception as exc:
                raise RuntimeError(f'{name}: {exc}') from exc

        # SCR and WSCR share the POIs' SCMVA: one set of fault solves.
        poi_scmva = fault_scmva(base_net, tuple(rated['SCR']), v_pf=base_v)
        for name in ('SCR', 'WSCR'):
            record = evaluate(name, base_net, base_v, scmva=poi_scmva)
            emit_metric(out['results'], out['variables'], name, record, key)
        # ESCR probes start from the generator states of this pre-fault solve.
        for variant_key, record in evaluate('ESCR', aug_net, aug_v,
                                            pv_state=aug_pf.pv_state).items():
            variant = variant_name(variant_key)
            if 'probe_pv_changes' in record:  # sites whose probe changed a generator's mode
                out['switches'][variant] = sum(n > 0 for n in record['probe_pv_changes'].values())
            terms = ESCR_TERMS.get(variant, ('value', 'if_magnitude'))
            emit_metric(out['results'], out['variables'], 'ESCR',
                        {t: record[t] for t in terms}, key, variant)
        emit_metric(out['results'], out['variables'], 'SDSCR',
                    evaluate('SDSCR', aug_net, aug_v), key)
    except Exception as exc:
        state['error'] = out['error'] = str(exc)
    return out


def state_tasks(week, contingencies):
    """(hour, timestamp, load scale, commitment, case, line) for every state: per hour,
    the sequential intact state and outages (E1, E2), then the all_online intact state (E5)."""
    scenarios = ([(SEQUENTIAL, INTACT, None)]
                 + [(SEQUENTIAL, case, line.lower()) for case, line in contingencies]
                 + [(ALL_ONLINE, INTACT, None)])
    return [(int(row.hour_index), row.timestamp, float(row.load_scale), commitment, case, line)
            for row in week.itertuples(index=False) for commitment, case, line in scenarios]


def evaluate_temporal(week, *, contingencies=(), out_dir=RAW_DIR,
                      voltage_limits=None, workers=WORKERS):
    # voltage bands only flag states in run_summary.csv; nothing is corrected
    voltage_limits = voltage_limits or VoltageLimits.preset()
    tasks = state_tasks(week, contingencies)
    results, variables, statuses, dispatch_rows = [], [], [], []
    switches = Counter()
    fn = functools.partial(evaluate_state, voltage_limits=voltage_limits)
    with parallel_map(fn, tasks, workers) as outcomes:
        for (h, timestamp, _, commitment, case, _), out in zip(tasks, outcomes):
            print(f'[{h + 1:03d}/{len(week):03d}] {timestamp} {commitment} case={case}',
                  flush=True)
            results += out['results']
            variables += out['variables']
            dispatch_rows += out['dispatch']
            statuses.append(out['state'])
            switches.update(out['switches'])
            if out['error'] is not None:
                # Fail visibly; never silently treat missing hours as valid.
                write_outputs(out_dir, results, variables, statuses, dispatch_rows)
                raise RuntimeError(f'Failed at hour={h}, {commitment}, case={case}: {out["error"]}')

    write_outputs(out_dir, results, variables, statuses, dispatch_rows)
    return {'states': len(statuses), 'probe_mode_switches': dict(sorted(switches.items()))}


KEY_COLUMNS = ['hour_index', 'timestamp', 'case', 'commitment']


def write_outputs(out_dir, results, variables, statuses, dispatch_rows):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results, columns=KEY_COLUMNS + ['metric', 'variant', 'bus', 'value']).to_csv(
        out_dir / 'metrics_timeseries.csv', index=False)
    pd.DataFrame(variables, columns=KEY_COLUMNS + ['metric', 'variant', 'bus', 'related_bus',
                                                   'variable', 'unit', 'real', 'imag',
                                                   'magnitude']).to_csv(
        out_dir / 'metric_variables.csv', index=False)
    pd.DataFrame(statuses).to_csv(out_dir / 'run_summary.csv', index=False)
    pd.DataFrame(dispatch_rows).to_csv(out_dir / 'dispatch_schedule.csv', index=False)


def run(week, output, *, contingencies=(), contingency_file=None, voltage_limits=None,
        workers=WORKERS):
    """evaluate_temporal() into a temporary directory, then publish to `output` with
    its manifest.

    The published result set is replaced only after every requested state has
    completed successfully. Failed runs leave the previous full result set intact.
    """
    started = manifest.now()
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    temp_output = Path(tempfile.mkdtemp(prefix='.strength-run-', dir=output.parent))
    try:
        counts = evaluate_temporal(week, contingencies=contingencies, out_dir=temp_output,
                                     voltage_limits=voltage_limits, workers=workers)
        output.mkdir(parents=True, exist_ok=True)
        for filename in RUN_FILES:
            source = temp_output / filename
            if source.exists():
                source.replace(output / filename)
        inputs = [CASE_FILE, GEN_LIMITS_FILE, SE3_FILE] + ([contingency_file] if contingency_file else [])
        manifest.write(output, 'simulate', started=started, inputs=inputs,
                       outputs=[output / f for f in RUN_FILES],
                       extra={'hours': int(len(week)), **counts, 'workers': workers,
                              'cases': [INTACT] + [c for c, _ in contingencies]})
        print('\nSaved results to', output)
        print('States evaluated:', counts['states'])
        print('Probe generator mode changes (site-states):', counts['probe_mode_switches'])
        return counts
    finally:
        shutil.rmtree(temp_output, ignore_errors=True)


# =============================================================================
# 3. Command line
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='stage', required=True)
    for name in ('screen', 'run'):
        p = sub.add_parser(name)
        p.add_argument('--se3-file', type=Path, default=SE3_FILE)
        p.add_argument('--hours', type=int, default=WINDOW_HOURS,
                       help='Run first N hours of the established 168-hour window (smoke test: 2).')
        p.add_argument('--workers', type=int, default=WORKERS,
                       help=f'Worker processes (default {WORKERS}); 1 runs serially. '
                            'The output does not depend on it.')
    screen_p, run_p = sub.choices['screen'], sub.choices['run']
    screen_p.add_argument('--output', type=Path, default=CONTINGENCY_FILE)
    run_p.add_argument('--contingencies', type=Path,
                       help='contingency_set.csv: also evaluate every included line outage.')
    run_p.add_argument('--case', action='append', dest='cases',
                       help='With --contingencies: evaluate only this case label (repeatable).')
    run_p.add_argument('--output', type=Path, default=RAW_DIR)
    run_p.add_argument('--voltage-preset', default=VOLTAGE_PRESET, choices=sorted(PRESETS),
                       help='VoltageLimits preset used to flag out-of-band states.')
    args = parser.parse_args()
    if not 1 <= args.hours <= WINDOW_HOURS:
        parser.error(f'--hours must be between 1 and {WINDOW_HOURS}')
    week = load_week(args.se3_file, args.hours)

    if args.stage == 'screen':
        table = screen(week, args.workers)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(args.output, index=False)
        print(f"\n{int(table['included'].sum())}/{len(table)} lines included; saved {args.output}")
        return

    if args.cases and not args.contingencies:
        parser.error('--case needs --contingencies')
    contingencies = read_contingencies(args.contingencies, args.cases) if args.contingencies else ()
    run(week, args.output, contingencies=contingencies, contingency_file=args.contingencies,
        voltage_limits=VoltageLimits.preset(args.voltage_preset), workers=args.workers)


if __name__ == '__main__':
    main()
