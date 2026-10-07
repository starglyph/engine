"""Protect identical objectives and prior-aware stationarity diagnostics."""
import copy
import unittest

import numpy as np

from robustness_orion_convergence import ARMS, METHODS, build_input, previous, read, stationarity


class ConvergenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original = read(previous('42-input'))
        cls.saved = read(previous('42-results'))
        cls.alternate = read(previous('41-trace'))['steps'][-1]['camera']

    def test_all_arms_keep_ordered_pairs_and_explicit_prior(self):
        bundle = build_input(self.original, self.saved, self.alternate)
        self.assertEqual(len(bundle['cases']), 9)
        for i, method in enumerate(METHODS):
            cases = bundle['cases'][i*3:i*3+3]
            old = next(c for c in self.original['cases'] if c['name'] == 'distributed16_'+method)
            saved = next(c for c in self.saved['cases'] if c['name'] == old['name'])
            for case, arm in zip(cases, ARMS):
                self.assertEqual(case['name'], arm+'_'+method)
                self.assertEqual(case['matches'], old['matches'])
                self.assertEqual(case['prior_weight'], saved['prior_weight'])
            self.assertEqual(cases[0]['initial'], old['initial'])
            self.assertEqual(cases[0]['expected'], saved['camera'])
            self.assertEqual(cases[1]['initial'], saved['camera'])
            self.assertEqual(cases[2]['initial'], self.alternate)
            cases[2]['matches'][0]['xy'][0] += 1
            self.assertEqual(cases[0]['matches'], old['matches'])

    def test_reject_holdout_scope(self):
        changed = copy.deepcopy(self.original)
        changed['split'] = 'holdout'
        with self.assertRaises(ValueError):
            build_input(changed, self.saved, self.alternate)

    def test_reject_different_pixel_system(self):
        changed = self.alternate | {'width':4485}
        with self.assertRaises(ValueError):
            build_input(self.original, self.saved, changed)

    @staticmethod
    def quadratic_case():
        j = np.zeros((33, 5)); j[:5] = np.eye(5); j[-1, 4] = 2
        r = np.zeros(33); r[4] = -2; r[-1] = 1
        jp = np.zeros((95, 5)); jp[:5] = np.eye(5)
        return dict(jacobian=j, residuals=r, probe_jacobian=jp, camera=dict(width=1600, height=1042))

    def test_nonzero_prior_cancels_data_gradient_at_optimum(self):
        result = stationarity(self.quadratic_case())
        self.assertTrue(result['stationary_diagnostic'])
        np.testing.assert_allclose(result['delta_parameters'], 0, atol=1e-14)
        self.assertAlmostEqual(result['objective'], 5)

    def test_unfinished_fit_has_linear_descent(self):
        case = self.quadratic_case(); case['residuals'][0] = 1
        result = stationarity(case)
        self.assertFalse(result['stationary_diagnostic'])
        self.assertAlmostEqual(result['delta_parameters'][0], -1)
        self.assertAlmostEqual(result['relative_linear_reduction'], 1/6)
        self.assertAlmostEqual(result['max_linear_probe_shift_original_px'], 4485/1600)

    def test_probe_derivative_shape_guard(self):
        case = self.quadratic_case(); case['probe_jacobian'] = np.zeros((94, 5))
        with self.assertRaises(ValueError):
            stationarity(case)


if __name__ == '__main__':
    unittest.main()
