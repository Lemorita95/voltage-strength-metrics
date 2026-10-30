"""Checks of the simulation outputs (experiments.glover37.simulate).

Guards traceability: every raw row belongs to a known experiment (commitment, case)
and a known ESCR variant, every state appears once, and a run writes its manifest.

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

import pandas as pd

from experiments.glover37 import simulate as rsm
from experiments.glover37.config import ESCR_VARIANTS, SE3_FILE, experiment_of
from experiments.glover37.network import read_se3, select_week


class TestRunOutputs(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls._tmp = tempfile.TemporaryDirectory()
        cls.raw = Path(cls._tmp.name) / 'raw'
        week = select_week(read_se3(SE3_FILE)).iloc[:2]
        with warnings.catch_warnings(), contextlib.redirect_stdout(io.StringIO()):
            warnings.simplefilter('ignore')
            rsm.run(week, cls.raw, contingencies=[('line_39_47_out', 'line.line 39-47 ckt 1')])
        read = lambda f: pd.read_csv(cls.raw / f, keep_default_na=False)
        cls.files = {f: read(f) for f in rsm.RUN_FILES}

    @classmethod
    def tearDownClass(cls):
        cls._tmp.cleanup()

    def test_every_row_has_an_experiment(self):
        for name, frame in self.files.items():
            experiments = {experiment_of(c, k) for c, k in zip(frame['commitment'], frame['case'])}
            self.assertEqual(experiments, {'E1', 'E2', 'E5'}, name)

    def test_variants(self):
        for name in ('metrics_timeseries.csv', 'metric_variables.csv'):
            frame = self.files[name]
            escr = frame['metric'] == 'ESCR'
            self.assertEqual(set(frame.loc[escr, 'variant']), set(ESCR_VARIANTS), name)
            self.assertEqual(set(frame.loc[~escr, 'variant']), {''}, name)

    def test_states_once(self):
        summary = self.files['run_summary.csv']
        self.assertFalse(summary.duplicated(['hour_index', 'case', 'commitment']).any())
        self.assertEqual(len(summary), 2 * 3)  # 2 hours x (intact, outage, all_online)

    def test_manifest(self):
        record = json.loads((self.raw / 'manifest.json').read_text())
        self.assertEqual(record['stage'], 'simulate')
        self.assertEqual({Path(p).name for p in record['outputs_sha256']}, set(rsm.RUN_FILES))
        for key in ('run_id', 'git_commit', 'uncommitted_changes', 'of_record', 'config'):
            self.assertIn(key, record)


if __name__ == '__main__':
    unittest.main()
