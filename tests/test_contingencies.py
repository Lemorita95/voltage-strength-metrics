"""Checks of the N-1 branch set (experiments.glover37.simulate screen, analyze.scenarios).

Lines and transformers of the 37-bus case are both candidates; every branch gets one case
label, and parallel branches of one kind between two buses form one corridor.

Run from the repository root:
    python -m unittest discover -s tests -v
"""
import unittest

import pandas as pd

from experiments.glover37 import analyze as asm
from experiments.glover37 import simulate as rsm


class TestBranchSet(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.net = rsm.reference_network()
        cls.branches = rsm.candidate_branches(cls.net)
        cls.labels = rsm.case_labels(cls.branches)

    def test_lines_and_transformers_are_candidates(self):
        self.assertEqual(len(self.branches), len(self.net.lines) + len(self.net.transformers))

    def test_labels_unique_and_typed(self):
        self.assertEqual(len(set(self.labels.values())), len(self.labels))
        self.assertEqual(self.labels['line.line 39-47 ckt 1'], 'line_39_47_out')
        self.assertEqual(self.labels['transformer.transformer 12-40 ckt 1'], 'transformer_12_40_ckt1_out')
        self.assertEqual(self.labels['transformer.transformer 10-39 ckt 1'], 'transformer_10_39_out')

    def test_no_transformer_outage_islands(self):
        for key in self.net.transformers:
            self.assertFalse(rsm.islands(self.net, key), key)


class TestCorridors(unittest.TestCase):

    def test_parallel_branches_one_corridor_per_kind(self):
        rows = [('line.line 18-37 ckt 1', 'line_18_37_ckt1_out', 18, 37),
                ('line.line 18-37 ckt 2', 'line_18_37_ckt2_out', 18, 37),
                ('transformer.transformer 12-40 ckt 1', 'transformer_12_40_ckt1_out', 12, 40),
                ('transformer.transformer 12-40 ckt 2', 'transformer_12_40_ckt2_out', 12, 40),
                ('transformer.transformer 10-39 ckt 1', 'transformer_10_39_out', 10, 39)]
        table = pd.DataFrame([{'branch': k, 'case': c, 'from_bus': a, 'to_bus': b, 'included': True}
                              for k, c, a, b in rows])
        corridors, representative = asm.scenarios(table)
        self.assertEqual(corridors, ['line_18_37_ckt1_out', 'transformer_12_40_ckt1_out',
                                     'transformer_10_39_out'])
        self.assertEqual(representative['transformer_12_40_ckt2_out'], 'transformer_12_40_ckt1_out')


if __name__ == '__main__':
    unittest.main()
