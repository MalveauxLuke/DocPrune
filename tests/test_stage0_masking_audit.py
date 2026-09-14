"""Small numerical checks for the read-only Stage 0 analysis."""
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from stage0_masking_audit import budget_order, coverage, rank, ridge_cv


class Stage0AuditTests(unittest.TestCase):
    def test_ties_and_unequal_cost_nomination(self):
        np.testing.assert_equal(rank([3, 1, 3, 5]), [1.5, 0, 1.5, 3])
        # A highly ranked action larger than the cap cannot be nominated;
        # smaller following actions still fit. This is not retained-mask search.
        selected, cost = budget_order([9, 8, 7], np.array([10, 4, 3]), 7)
        self.assertEqual((selected, cost), ([1, 2], 7))
        self.assertEqual(coverage(np.array([0., 2., 3.]), [1]), .4)
        self.assertIsNone(coverage(np.zeros(3), [1]))

    def test_held_out_linear_recovery_for_integer_and_float_inputs(self):
        rng = np.random.default_rng(184)
        x = rng.integers(2, size=(150, 6))
        g, s = x[:, 0] + 2*x[:, 2], x[:, 1] - x[:, 5]
        y = np.column_stack((g, s, g-s))
        pred, fits = ridge_cv(x, y)
        float_pred, float_fits = ridge_cv(x.astype(float), y.astype(float))
        self.assertLess(float(np.mean((pred-y)**2)), .002)
        np.testing.assert_allclose(pred, float_pred)
        np.testing.assert_allclose(fits, float_fits)
        np.testing.assert_allclose(pred[:, 2], pred[:, 0]-pred[:, 1], atol=1e-12)


if __name__ == '__main__':
    unittest.main()
