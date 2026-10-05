"""Guard the single change, development selection and geometric hold-back."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import robustness_centroid_origin as run
from robustness_centroid_origin_report import geometry_groups


class CentroidOriginTests(unittest.TestCase):
    def test_patch_changes_only_two_origin_expressions_plus_test_module(self):
        source = 'prefix\n' + run.BEFORE + '\ntest\n' + run.TEST_BEFORE + '\nsuffix\n'
        self.assertEqual(run.patch_source(source), 'prefix\n' + run.AFTER + '\ntest\n' + run.TEST_AFTER + '\nsuffix\n' + run.MODULE)
        with self.assertRaises(ValueError):
            run.patch_source(source + run.BEFORE)
        with self.assertRaises(ValueError):
            run.patch_source('new source')

    def test_holdout_cannot_start_hash_or_image_access(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out/'protocol.json').write_text(json.dumps(dict(split='holdout', holdout=True)))
            with patch.object(run, 'digest', side_effect=AssertionError('must not read')):
                with self.assertRaisesRegex(ValueError, 'development only'):
                    run.validate(out)

    def test_membership_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out/'protocol.json').write_text(json.dumps(dict(split='development', holdout=False, ids=[])))
            with patch.object(run, 'records', return_value=[dict(id='development')]):
                with self.assertRaisesRegex(ValueError, 'selection changed'):
                    run.validate(out)

    def test_probe_exclusion_uses_union_of_both_current_input_lists(self):
        xy = np.array([[5., 5.], [50., 50.], [95., 95.]])
        masks = geometry_groups(xy, [np.array([[5., 5.]]), np.array([[50., 51.]])], 100, 100, 12)
        np.testing.assert_array_equal(masks['outside_current_inputs'], [False, False, True])
        np.testing.assert_array_equal(masks['left_10pct'], [True, False, False])
        self.assertTrue(masks['all_reviewed'].all())

    def test_empty_detection_list_does_not_remove_probes(self):
        masks = geometry_groups(np.array([[50., 50.]]), [np.empty((0, 2))], 100, 100, 12)
        self.assertTrue(masks['outside_current_inputs'][0])
        self.assertFalse(masks['left_10pct'].any())


if __name__ == '__main__':
    unittest.main()
