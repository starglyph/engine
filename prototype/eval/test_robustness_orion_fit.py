"""Protect fixed selection, disjoint scoring and working-pixel conventions."""
import copy
import unittest

import numpy as np

from robustness_orion_fit import (WIDTH, HEIGHT, METHODS, adapted_harness, build_input,
    favourable, previous, read, selection, working_xy)
from robustness_refinement_report import lift


class OrionFitTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources = read(previous('40-results'))['sources']
        cls.candidates = read(previous('40-candidates'))['frames'][0]['points']
        cls.trace = read(previous('41-trace'))

    def test_selection_disjoint_and_residual_independent(self):
        expected = [14, 6, 4, 45, 23, 19, 1, 28, 2, 35, 8, 9, 38, 32, 0, 11]
        chosen = selection(self.sources, self.candidates)
        self.assertEqual(chosen['fit_ids'], expected)
        self.assertEqual(len(chosen['check_ids']), 31)
        self.assertFalse(set(chosen['fit_ids']) & set(chosen['check_ids']))
        changed = copy.deepcopy(self.sources)
        for s in changed:
            s['predicted_xy'] = {'unusable': None}
            s['groups'] = []
        self.assertEqual(selection(changed[::-1], self.candidates)['fit_ids'], expected)

    def test_missing_cell_stops_without_substitution(self):
        sources = [s for s in self.sources if not (s['coordinates']['external_centroid'][0] < WIDTH/4
            and s['coordinates']['external_centroid'][1] < HEIGHT/4)]
        with self.assertRaises(ValueError):
            selection(sources, self.candidates)

    def test_duplicate_review_stops(self):
        with self.assertRaises(ValueError):
            selection(self.sources+[self.sources[0]], self.candidates)

    def test_pixel_center_roundtrip_with_unequal_scales(self):
        cam = self.trace['input']['camera']
        coords = [[0., 0.], [WIDTH-1, HEIGHT-1], [2348.21, 2651.54]]
        np.testing.assert_allclose(lift(working_xy(coords, cam), cam, WIDTH, HEIGHT), coords, rtol=0, atol=1e-12)

    def test_fit_membership_and_objectives_frozen(self):
        chosen = selection(self.sources, self.candidates)
        inp = build_input(self.sources, chosen, self.trace)
        self.assertEqual(len(inp['cases']), 4)
        self.assertEqual(inp['cases'][0]['expected'], self.trace['steps'][0]['camera'])
        for c, method in zip(inp['cases'][1:], METHODS):
            self.assertEqual({m['source_id'] for m in c['matches']}, set(chosen['fit_ids']))
            self.assertEqual(c['initial'], inp['cases'][0]['initial'])
            self.assertIsNone(c['prior_weight'])
            for m in c['matches']:
                source = next(s for s in self.sources if s['id'] == m['source_id'])
                self.assertEqual(m['xy'], working_xy(source['coordinates'][method], c['initial']))

    def test_empty_group_allowed_but_populated_regression_rejected(self):
        before = dict(all_reviewed=dict(count=31, rms_px=100), left=dict(count=2, rms_px=10), right=dict(count=0))
        after = dict(all_reviewed=dict(count=31, rms_px=80), left=dict(count=2, rms_px=10.51), right=dict(count=0))
        self.assertFalse(favourable(after, before))
        after['left']['rms_px'] = 10.5
        self.assertTrue(favourable(after, before))

    def test_adapter_only_changes_fixed_frame_guard(self):
        source = 'assert_eq!(input.id, "wm_r_143159342");'
        self.assertEqual(adapted_harness(source), 'assert_eq!(input.id, "wm_r_154807046");')
        with self.assertRaises(ValueError):
            adapted_harness('missing guard')


if __name__ == '__main__':
    unittest.main()
