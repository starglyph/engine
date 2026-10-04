"""Guard development isolation and distinguish mask membership from solving."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from robustness_mask_screen import probe_metrics, records, retained, saved_detections, validate


class MaskScreenTests(unittest.TestCase):
    def test_holdout_rejected_before_image_access(self):
        with self.assertRaisesRegex(ValueError, 'outside selected split'):
            records(['wm_r_146925915'])

    def test_abstention_keeps_sources_in_unmasked_path(self):
        self.assertTrue(retained([80., 90.], None))
        self.assertEqual(saved_detections(None, {}, None)['status'], 'missing_saved_report')

    def test_group_losses_preserve_edge_and_outside_membership(self):
        sources = [dict(source_id=0, hyg_id=1, name='a', xy=[5., 5.], outside_both_inputs=True),
                   dict(source_id=1, hyg_id=2, name='b', xy=[95., 95.], outside_both_inputs=False)]
        mask = dict(width=100, height=100, sky_polygon=[[0, 0], [1, 0], [1, .5], [0, .5]])
        result = probe_metrics(sources, mask, 100, 100)
        self.assertEqual(result['groups']['all_reviewed'], dict(total=2, retained=1))
        self.assertEqual(result['groups']['bottom_10pct'], dict(total=1, retained=0))
        self.assertEqual(result['groups']['outside_both_inputs'], dict(total=1, retained=1))

    def test_saved_inliers_are_measured_without_new_solve_claim(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rec = dict(id='test', width=100, height=100, clean_sha256='abc')
            report = dict(id='test', width=100, height=100, source_sha256='abc',
                report=dict(status='solved', detections=[dict(x=20., y=20., inlier=True),
                                                        dict(x=20., y=80., inlier=True)]))
            (root/'saved.json').write_text(json.dumps(report))
            mask = dict(width=100, height=100, sky_polygon=[[0, 0], [1, 0], [1, .5], [0, .5]])
            with patch('robustness_mask_screen.ROOT', root):
                result = saved_detections('saved.json', rec, mask)
                self.assertEqual((result['inliers_total'], result['inliers_retained']), (2, 1))
                self.assertEqual(result['prior_solve_status'], 'solved')
                rec['clean_sha256'] = 'changed'
                with self.assertRaisesRegex(ValueError, 'source mismatch'):
                    saved_detections('saved.json', rec, mask)

    def test_invalid_split_stops_before_hash_reads(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'protocol.json').write_text(json.dumps(dict(split='holdout', holdout=True)))
            with self.assertRaisesRegex(ValueError, 'development required'):
                validate(root)


if __name__ == '__main__':
    unittest.main()
