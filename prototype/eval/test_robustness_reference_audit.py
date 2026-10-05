"""Selection independence, spatial exclusions and frozen-scope protection."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import numpy as np

from collection_run import digest, write_json
from robustness_reference_audit import RID, distributed, group_masks, validate
from robustness_reference_coverage import extent


class ReferenceAuditTests(unittest.TestCase):
    def test_selection_keeps_brightest_even_without_extraction(self):
        points = [dict(hyg_id=i, mag=m, review_center=xy, choices=choices)
                  for i, m, xy, choices in [(4, 4, [10, 10], [1]), (2, 2, [12, 12], []),
                     (3, 3, [11, 11], [1]), (1, 1, [13, 13], []), (5, 5, [70, 70], [])]]
        result = distributed(points, 100, 100)
        self.assertEqual([p['hyg_id'] for p in result], [1, 2, 3, 5])
        self.assertEqual(result, distributed(list(reversed(points)), 100, 100))
        self.assertNotIn('id', points[0])

    def test_outside_groups_exclude_all_external_rows_and_each_input(self):
        xy = np.array([[10, 10], [30, 30], [50, 50], [90, 90]], dtype=float)
        # Both solver lists are concatenated; external rows are not filtered by weight.
        masks = group_masks(xy, np.array([[10, 10], [30, 30]]), np.array([[30, 30], [50, 50]]), 100, 100)
        self.assertEqual(masks['outside_both_starglyph_inputs'].tolist(), [False, False, True, True])
        self.assertEqual(masks['outside_external_corr'].tolist(), [True, False, False, True])
        self.assertEqual(masks['outside_all_inputs'].tolist(), [False, False, False, True])

    def test_holdout_rejected_before_image_access(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            write_json(out/'protocol.json', dict(id=RID, split='holdout', holdout=True))
            with patch('robustness_reference_audit.selected') as select:
                with self.assertRaisesRegex(ValueError, 'development only'):
                    validate(out)
                select.assert_not_called()

    def test_mutated_camera_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            camera = out/'camera.json'
            camera.write_text('before')
            write_json(out/'protocol.json', dict(id=RID, split='development', holdout=False,
                hashes={str(camera): digest(camera)}))
            camera.write_text('after')
            with patch('robustness_reference_audit.selected'):
                with self.assertRaisesRegex(ValueError, 'frozen input changed'):
                    validate(out)

    def test_extent_does_not_imply_full_frame_coverage(self):
        d = extent([dict(x=60, y=10), dict(x=80, y=90)], 100, 100)
        self.assertEqual(d['bounds_xyxy'], [60, 10, 80, 90])
        self.assertAlmostEqual(d['width_fraction'], .2)
        self.assertIsNone(extent([], 100, 100)['bounds_xyxy'])
