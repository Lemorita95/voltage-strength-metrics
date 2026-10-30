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
from unittest import mock

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

    def test_controls_settled_flag(self):
        summary = self.files['run_summary.csv']
        for network in ('base', 'ibr'):
            self.assertIn(f'{network}_controls_settled', summary.columns)
        record = json.loads((self.raw / 'manifest.json').read_text())
        self.assertEqual(set(record['controls_unsettled']), {'base', 'ibr'})


class TestFailedRun(unittest.TestCase):

    def test_failed_state_keeps_partial_outputs(self):
        week = select_week(read_se3(SE3_FILE)).iloc[:1]
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(rsm, 'augment_network', side_effect=ValueError('boom')), \
                contextlib.redirect_stdout(io.StringIO()):
            raw = Path(tmp) / 'raw'
            with self.assertRaisesRegex(RuntimeError, r'hour=0, .*case=intact: boom'):
                rsm.run(week, raw, workers=1)
            self.assertFalse(raw.exists())
            [failed] = Path(tmp).glob('raw_failed_*')
            summary = pd.read_csv(failed / 'run_summary.csv', keep_default_na=False)
        self.assertEqual(len(summary), 1)  # stops at the first failing state
        row = summary.iloc[0]
        self.assertEqual(row['error'], 'boom')
        self.assertFalse(row['base_pf_ok'])
        self.assertFalse(row['ibr_pf_ok'])

    def test_power_flow_error_names_network(self):
        net = rsm.reference_network()
        with mock.patch.object(rsm.PowerFlow, 'solve_power_flow', return_value=None):
            with self.assertRaisesRegex(RuntimeError, r'^base power flow failed \(37-bus network\)$'):
                rsm.solve_pf(net, 'base')


if __name__ == '__main__':
    unittest.main()
