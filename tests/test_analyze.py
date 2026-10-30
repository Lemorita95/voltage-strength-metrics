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
from experiments.glover37.config import SE3_FILE
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

    def test_reconstruction(self):
        self.assertLess(self.out['reconstruction_check']['max_rel_diff'].max(), 1e-12)

    def test_decompositions_add_up(self):
        for name in ('d1_weakest_hour', 'd5_snapshots'):
            t = self.out[name]
            np.testing.assert_allclose(t['dln_total'], t['dln_V'] + t['dln_coupling'] + t['dln_Z_ii'],
                                       atol=1e-12)
        d4 = self.out['d4_outage']
        scr = d4[d4['metric'].isin(['SCR', 'WSCR'])]
        self.assertTrue((scr['median_dln_coupling'].abs() < 1e-12).all())

    def test_analysis_manifest_names_raw_run(self):
        raw = json.loads((self.raw / 'manifest.json').read_text())
        analysis = json.loads((Path(self._tmp.name) / 'analysis' / 'manifest.json').read_text())
        self.assertEqual(analysis['raw_run_id'], raw['run_id'])


if __name__ == '__main__':
    unittest.main()
