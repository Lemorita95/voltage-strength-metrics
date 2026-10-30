"""Checks of the operating-state rules (experiments.glover37.network).

Guards E5's rule (a): every unit online, within its limits, meeting the target, in the
three regimes (slack exporting, interior, MWMax exhausted).

Run from the repository root:
    python -m unittest discover -s tests -v
"""
import unittest

from experiments.glover37.config import ALL_ONLINE, GEN_LIMITS_FILE, SEQUENTIAL
from experiments.glover37.network import read_gen_limits, schedule


class TestAllOnlineRule(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.limits = read_gen_limits(GEN_LIMITS_FILE)
        cls.mw_min = sum(g.mw_min for g in cls.limits)
        cls.mw_max = sum(g.mw_max for g in cls.limits)

    def check(self, target):
        dispatch, committed = schedule(ALL_ONLINE, target, self.limits)
        self.assertTrue(all(committed.values()))
        for g in self.limits:
            self.assertGreaterEqual(dispatch[g.key], g.mw_min - 1e-12)
            self.assertLessEqual(dispatch[g.key], g.mw_max + 1e-12)
        return sum(dispatch.values())

    def test_interior_meets_target(self):
        for target in (self.mw_min + 1.0, 446.8, 0.5 * (self.mw_min + self.mw_max), self.mw_max - 1.0):
            self.assertAlmostEqual(self.check(target), target, places=9)

    def test_below_min_slack_exports(self):
        self.assertAlmostEqual(self.check(self.mw_min - 50.0), self.mw_min, places=12)

    def test_above_max_slack_supplies(self):
        self.assertAlmostEqual(self.check(self.mw_max + 50.0), self.mw_max, places=12)

    def test_sequential_commits_only_dispatched_units(self):
        dispatch, committed = schedule(SEQUENTIAL, 300.0, self.limits)
        self.assertEqual(committed, {k: p > 0 for k, p in dispatch.items()})


if __name__ == '__main__':
    unittest.main()
