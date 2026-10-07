"""Protect manual selection geometry, top-K timing and failure attribution."""
import copy
from pathlib import Path
import tempfile
import unittest

from collection_run import write_json
from robustness_snow_foreground import make_cases, sky_indices, failure_stage, validate, RID, POSITIVE


class SnowForegroundTests(unittest.TestCase):
    def test_filter_precedes_top_k_and_preserves_values_and_order(self):
        points = [dict(x=i+.3, y=90. if i % 3 == 0 else 10., flux=200-i, rank=i) for i in range(90)]
        diag = [dict(width=100, height=100, tier='default', max_detections=40, result=dict(detections=points))]
        before = copy.deepcopy(diag); cases, selections = make_cases(diag, 100)
        self.assertEqual(diag, before)
        self.assertEqual(cases[0]['search'], points[:40])
        self.assertEqual(cases[1]['search'], [p for p in points if p['y'] == 10.][:40])
        self.assertTrue(selections[1]['newly_promoted_from_below_top_k'])
        self.assertEqual((cases[1]['width'], cases[1]['height']), (100, 100))

    def test_pixel_center_lift_at_boundary(self):
        # Original boundary 450; y=224.75 at half resolution maps exactly to450.
        self.assertEqual(sky_indices([dict(y=224.74), dict(y=224.75), dict(y=224.76)], 500, 1000), [0])
        with self.assertRaises(ValueError):
            sky_indices([], 0, 1000)

    def test_not_enough_points_is_not_silent_budget_change(self):
        d = dict(width=100, height=100, tier='default', max_detections=40,
                 result=dict(detections=[dict(y=99.) for _ in range(40)]))
        with self.assertRaisesRegex(ValueError, 'prefixes'):
            make_cases([d], 100)

    def test_stage_is_furthest_reached_and_timeout_separate(self):
        counts = dict(patterns=2, hash_candidates=10, fov_pass=3, ratios_pass=0,
                      verifications=0, probability_pass=0)
        self.assertEqual(failure_stage(dict(error='NoMatch'), counts), 'edge_ratios')
        counts.update(ratios_pass=1, verifications=1)
        self.assertEqual(failure_stage(dict(error='NoMatch'), counts), 'verification_probability')
        self.assertEqual(failure_stage(dict(error='Timeout'), counts), 'timeout')
        self.assertEqual(failure_stage(dict(matches=8), counts), 'candidate')

    def test_holdout_guard_precedes_artifact_reads(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder); write_json(out/'protocol.json', dict(ids=[RID, POSITIVE], split='holdout', holdout=True))
            with self.assertRaisesRegex(ValueError, 'development'):
                validate(out)


if __name__ == '__main__':
    unittest.main()
