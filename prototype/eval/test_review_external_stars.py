import copy
import unittest

import numpy as np

from review_external_stars import compare_frame, registration_check
from audit_sky_edges import edge_margin


class ExternalStarReviewTests(unittest.TestCase):
    def test_edge_strips_use_each_axis_and_reject_outside_pixels(self):
        self.assertEqual(edge_margin(0, 300, 1001, 601), 0.)
        self.assertEqual(edge_margin(500, 600, 1001, 601), 0.)
        self.assertAlmostEqual(edge_margin(100, 300, 1001, 601), .1)
        self.assertAlmostEqual(edge_margin(500, 60, 1001, 601), .1)
        self.assertEqual(edge_margin(500, 300, 1001, 601), .5)
        self.assertIsInstance(edge_margin(np.float64(500), np.float64(300), 1001, 601) <= .1, bool)
        with self.assertRaisesRegex(ValueError, 'invalid edge'):
            edge_margin(-1, 300, 1001, 601)

    def test_image_registration_does_not_fit_held_out_points(self):
        pixels = [[0, 0], [10, 0], [10, 10], [0, 10], [2, 3]]
        registration = {'points': [{'source_id': i, 'primary_xy': p,
                                     'role': 'anchor' if i < 4 else 'held_out'} for i, p in enumerate(pixels)]}
        reviewed = {'points': [{'id': i, 'x': 2*x+10, 'y': 2*y+20, 'label': 'visible_source'}
                                for i, (x, y) in enumerate(pixels)]}
        before = registration_check(registration, reviewed)
        self.assertLess(before['held_out_error']['rms_px'], 1e-10)
        reviewed['points'][4]['x'] += 3
        reviewed['points'][4]['y'] += 4
        after = registration_check(registration, reviewed)
        np.testing.assert_allclose(before['homography'], after['homography'])
        self.assertAlmostEqual(after['held_out_error']['rms_px'], 5.)

    def test_collinear_registration_anchors_are_rejected(self):
        registration = {'points': [{'source_id': i, 'primary_xy': [i, i],
                                     'role': 'anchor' if i < 4 else 'held_out'} for i in range(5)]}
        reviewed = {'points': [{'id': i, 'x': i, 'y': i, 'label': 'visible_source'} for i in range(5)]}
        with self.assertRaisesRegex(ValueError, 'degenerate'):
            registration_check(registration, reviewed)

    def setUp(self):
        self.frame = {'id': 'fixture', 'source_sha256': 'hash', 'width': 1000, 'height': 600,
                      'matches': [{'source_id': 1, 'star_id': 'external',
                                   'method': 'visual_pattern_transfer'}]}
        self.review = {**self.frame, 'points': [
            {'id': 1, 'x': 503., 'y': 304., 'label': 'visible_source'}]}
        self.stars = {'external': {'ra_deg': 0., 'dec_deg': 0.}}
        self.artifact = {**self.frame, 'pixel_convention': 'top_left_zero_based',
                         'report': {'status': 'solved'},
                         'camera': {'width': 1000, 'height': 600, 'focal_px': 900, 'k1': 0,
                                    'world_to_camera': [[0, -1, 0], [0, 0, 1], [1, 0, 0]]}}

    def compare(self):
        return compare_frame(self.frame, self.review, self.stars, self.artifact)

    def test_errors_are_measured_without_refitting_or_promoting_reference(self):
        before = copy.deepcopy(self.artifact)
        result = self.compare()
        self.assertEqual(result['external_star_error']['rms_px'], 5.)
        self.assertEqual(result['review_status'], 'diagnostic_only')
        self.assertEqual(self.artifact, before)
        self.assertEqual(result['points'][0]['predicted_x'], 500.)
        self.artifact['report']['status'] = 'failed'
        self.assertNotIn('external_star_error', self.compare())
        self.frame['matches'] = []
        self.assertEqual(self.compare()['reason'], 'no independently identified stars')

    def test_wrong_identity_and_duplicate_matches_are_rejected(self):
        self.artifact['source_sha256'] = 'another'
        with self.assertRaisesRegex(ValueError, 'source_sha256 mismatch'):
            self.compare()
        self.artifact['source_sha256'] = 'hash'
        self.frame['matches'].append(dict(self.frame['matches'][0]))
        with self.assertRaisesRegex(ValueError, 'one-to-one'):
            self.compare()

    def test_unreviewed_or_nonfinite_sources_cannot_become_reference_points(self):
        self.review['points'][0]['label'] = 'ambiguous_texture'
        with self.assertRaisesRegex(ValueError, 'invalid reviewed'):
            self.compare()
        self.review['points'][0]['label'] = 'visible_source'
        self.stars['external']['dec_deg'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'invalid reviewed'):
            self.compare()


if __name__ == '__main__':
    unittest.main()
