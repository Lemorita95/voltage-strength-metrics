"""Checks of the analysis stage (experiments.glover37.analyze) (stdlib unittest).

- planted answers for the D1 attribution and the exact log decomposition;
- an integration run on a fresh 3-hour simulation (intact, line 39-47 and both
  18-37 circuits, every ESCR variant, plus the all_online states) written to a
  temporary folder, never the results of record.

Run from the repository root:
    python -m unittest discover -s tests -v
"""
import contextlib
import io
import json
import tempfile
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.glover37 import analyze as asm
from experiments.glover37 import simulate as rsm
from experiments.glover37.config import PROBE_VARIANTS, REFERENCE_VARIANT, SE3_FILE, SEQUENTIAL
from experiments.glover37.network import read_se3, select_week

LINES = {  # case -> (line key, from, to)
    'line_39_47_out': ('line.line 39-47 ckt 1', 39, 47),
    'line_18_37_ckt1_out': ('line.line 18-37 ckt 1', 18, 37),
    'line_18_37_ckt2_out': ('line.line 18-37 ckt 2', 18, 37),
}


def quiet(fn, *args, **kwargs):
    with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
        warnings.simplefilter('ignore')
        return fn(*args, **kwargs)


def synthetic_values(n_hours, overrides):
    """values[variant][metric] with every series equal to `base`, except overrides
    {(variant, metric, site): array}."""
    base = np.linspace(2.0, 3.0, n_hours)
    index = pd.MultiIndex.from_product([['intact'], range(n_hours)], names=['case', 'hour_index'])
    values = {}
    for v in asm.VARIANTS:
        values[v] = {}
        for metric in asm.METRICS:
            cols = [asm.GROUP] if metric == 'WSCR' else list(asm.SITE_IDS)
            frame = pd.DataFrame({c: base for c in cols}, index=index)
            for (ov, om, os_), arr in overrides.items():
                if ov == v and om == metric:
                    frame[os_] = arr
            values[v][metric] = frame
    return values


class TestPlantedLogic(unittest.TestCase):

    def setUp(self):
        hours = np.arange(6)
        self.ctx = {'load_scale': pd.Series(np.linspace(0.7, 1.2, 6), index=hours),
                    'b_count': {'base': {0, 1}, 'ibr': {0, 1}}}

    def d1_row(self, overrides, metric='ESCR', site='21'):
        d1 = asm.d1_weakest_hour(synthetic_values(6, overrides), self.ctx, ['intact'])
        return d1[(d1['metric'] == metric) & (d1['site'] == site)].iloc[0]

    def test_commitment(self):
        row = self.d1_row({})
        self.assertEqual(row['t_star'], '0')
        self.assertEqual(row['explanation'], 'commitment')

    def test_voltage_exception(self):
        dip = np.array([2.0, 2.1, 2.2, 2.3, 2.4, 1.5])  # |V| dip at hour 5
        row = self.d1_row({('m', 'ESCR', '21'): dip, ('noC', 'ESCR', '21'): dip})
        self.assertEqual(row['t_star'], '5')
        self.assertEqual(row['explanation'], '|V|')

    def test_coupling_exception(self):
        dip = np.array([2.0, 2.1, 2.2, 2.3, 2.4, 1.5])  # coupling dip at hour 5
        row = self.d1_row({('m', 'ESCR', '21'): dip, ('noV', 'ESCR', '21'): dip})
        self.assertEqual(row['explanation'], 'coupling')

    def test_commitment_not_confirmed(self):
        late = np.array([3.0, 2.9, 2.8, 2.7, 2.6, 2.5])
        row = self.d1_row({(v, 'SCR', '21'): late for v in asm.VARIANTS}, metric='SCR')
        self.assertTrue(row['explanation'].startswith('commitment not confirmed'))

    def test_split_is_exact(self):
        rng = np.random.default_rng(1)
        overrides = {(v, 'SDSCR', '48'): rng.uniform(1, 2, 6) for v in asm.VARIANTS}
        values = synthetic_values(6, overrides)
        parts = asm.split(values, 'SDSCR', '48', ('intact', 1), ('intact', 4))
        m = values['m']['SDSCR']['48']
        self.assertAlmostEqual(parts['dln_total'], np.log(m.loc[('intact', 4)] / m.loc[('intact', 1)]))
        self.assertAlmostEqual(parts['dln_total'],
                               parts['dln_V'] + parts['dln_coupling'] + parts['dln_Z_ii'], places=12)


class TestIntegration(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.raw = Path(cls._tmp.name) / 'raw'
        cls.raw.mkdir()
        pd.DataFrame([{'line': key, 'case': case, 'from_bus': a, 'to_bus': b, 'islands': False,
                       'n_hours': 3, 'n_converged_base': 3, 'n_converged_ibr': 3,
                       'failed_hours': '', 'included': True, 'reason': ''}
                      for case, (key, a, b) in LINES.items()]
                     ).to_csv(cls.raw / 'contingency_set.csv', index=False)
        cls.week = select_week(read_se3(SE3_FILE)).iloc[:3]
        contingencies = rsm.read_contingencies(cls.raw / 'contingency_set.csv')
        quiet(rsm.run, cls.week, cls.raw, contingencies=contingencies)
        cls.out = quiet(asm.run, cls.raw, Path(cls._tmp.name) / 'analysis')

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_scenarios_count_corridors(self):
        d1 = self.out['d1_weakest_hour']
        self.assertEqual(sorted(d1['case'].unique()),
                         ['intact', 'line_18_37_ckt1_out', 'line_39_47_out'])

    def test_reconstruction(self):
        self.assertLess(self.out['reconstruction_check']['max_rel_diff'].max(), 1e-12)

    def test_parallel_circuits_identical(self):
        par = self.out['parallel_circuits']
        self.assertEqual(set(par['case']), {'line_18_37_ckt2_out'})
        self.assertTrue(par['same_t_star'].all())
        self.assertLess(par['max_abs_dln'].max(), 1e-9)

    def test_a3_unchanged(self):
        self.assertTrue(self.out['a3_denominators']['unchanged'].all())

    def test_decompositions_add_up(self):
        for name in ('d1_weakest_hour', 'd5_snapshots'):
            t = self.out[name]
            np.testing.assert_allclose(t['dln_total'], t['dln_V'] + t['dln_coupling'] + t['dln_Z_ii'],
                                       atol=1e-12)
        d4 = self.out['d4_outage']
        scr = d4[d4['metric'].isin(['SCR', 'WSCR'])]
        self.assertTrue((scr['median_dln_coupling'].abs() < 1e-12).all())

    def test_miif_variants(self):
        rob = self.out['miif_variants']
        self.assertEqual(list(dict.fromkeys(rob['variant'])), [REFERENCE_VARIANT, *PROBE_VARIANTS])

    def test_coupling_load_definition(self):
        v = pd.read_csv(self.raw / 'metric_variables.csv', keep_default_na=False)
        v['bus'] = pd.to_numeric(v['bus'], errors='coerce')
        v['related_bus'] = pd.to_numeric(v['related_bus'], errors='coerce')
        intact0 = v[(v.case == 'intact') & (v.hour_index == 0) & (v.metric == 'ESCR')
                    & (v.commitment == SEQUENTIAL) & (v.variant == REFERENCE_VARIANT)]
        ifm = intact0[intact0.variable == 'if_magnitude']
        site = asm.SITE_IDS[0]
        i_bus = asm.EVAL_BUS['ESCR'][site]
        rated = asm.P_MW['rated']
        expected = sum(float(r.real) * rated[next(s for s, b in asm.EVAL_BUS['ESCR'].items() if b == r.bus)]
                       for r in ifm[ifm.related_bus == i_bus].itertuples()) / rated[site]
        values = asm.variant_values(asm.view(asm.read_run(self.raw))['variables'],
                                    {m: asm.P_MW[asm.AS_RUN[m]] for m in asm.METRICS})
        self.assertAlmostEqual(asm.coupling_load(values, 'ESCR', site, 'intact').loc[0], expected, places=12)

    def test_e5_and_t3(self):
        # E5 (every unit online) is analysed on the intact network only; T3 pairs it with E1
        t3 = self.out['t3_commitment']
        self.assertEqual(len(t3), len(asm.UNITS))
        self.assertTrue((t3['E5_n_committed'] == 7).all())
        self.assertEqual(set(self.out['e5_weakest_hour']['case']), {'intact'})
        for name in ('d5_snapshots', 'e5_snapshots'):
            t = self.out[name]
            self.assertTrue((t['g'] >= -1e-12).all() and (t['g_minimum'] >= -1e-12).all())

    def test_tables_tagged(self):
        # every output table names a known experiment and the RQ/dimension it serves
        from experiments.glover37.config import EXPERIMENTS
        analysis = Path(self._tmp.name) / 'analysis'
        for path in analysis.glob('*.csv'):
            if path.stem == 'summary':
                continue
            t = pd.read_csv(path)
            self.assertEqual(list(t.columns[:2]), ['experiment', 'serves'], path.stem)
            ids = {i for cell in t['experiment'].astype(str) for i in cell.split(';')}
            self.assertTrue(ids and ids <= set(EXPERIMENTS), (path.stem, ids))
            self.assertTrue(t['serves'].astype(str).str.len().gt(0).all(), path.stem)

    def test_report_text(self):
        # every claim and caption has rows in summary.csv, and report.md quotes exactly those
        from experiments.glover37 import report
        summary = self.out['summary']
        claims = summary[summary['claim'].str.match(r'^(C\d|captions)')]
        self.assertEqual({c.split(' ')[0] for c in claims['claim']},
                         {'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'captions'})
        text = report.build_report(Path(self._tmp.name) / 'analysis', self.raw)
        for r in claims.itertuples():
            self.assertIn(f'- {r.item}: **{r.result}**', text)

    def test_outputs_written(self):
        analysis = Path(self._tmp.name) / 'analysis'
        self.assertEqual({p.stem for p in analysis.glob('*.csv')}, set(self.out))
        self.assertTrue((analysis / 'manifest.json').is_file())

    def test_analysis_manifest_names_raw_run(self):
        raw = json.loads((self.raw / 'manifest.json').read_text())
        analysis = json.loads((Path(self._tmp.name) / 'analysis' / 'manifest.json').read_text())
        self.assertEqual(analysis['raw_run_id'], raw['run_id'])

    def test_rejects_raw_without_manifest(self):
        bare = Path(self._tmp.name) / 'bare'
        bare.mkdir()
        for f in asm.INPUT_FILES:
            (bare / f).write_bytes((self.raw / f).read_bytes())
        with self.assertRaises(ValueError):
            quiet(asm.run, bare, Path(self._tmp.name) / 'bare_out')

    def test_rejects_input_as_output(self):
        with self.assertRaises(ValueError):
            asm.run(self.raw, self.raw)


if __name__ == '__main__':
    unittest.main()
