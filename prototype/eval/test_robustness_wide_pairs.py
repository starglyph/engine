"""Protect visual annotations and the one-source diagnostic intervention."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from robustness_geometry import validate_review
from robustness_wide_pairs import reviewed
from robustness_wide_replay import cases_for


class WidePairTests(unittest.TestCase):
    def input(self):
        points = [dict(id=0, hyg_id=81442, label='not_visible', detector_xy_working=[5.,6.]),
                  dict(id=1, hyg_id=96324, label='ambiguous', detector_xy_working=[7.,8.])]
        saved = dict(cases=[dict(name='drop_detection_all_stages',
            matches=[dict(xy=[1.,2.], world=[1.,0.,0.], hyg_ids=[1])],
            detections=[dict(x=1.,y=2.,rank=0),dict(x=5.,y=6.,rank=1),dict(x=7.,y=8.,rank=2)])])
        return points, saved

    def test_only_confirmed_source_removed_and_initial_fit_preserved(self):
        points, saved = self.input()
        before = copy.deepcopy(saved)
        cases, leaf = cases_for(points, saved)
        self.assertEqual(cases[0], saved['cases'][0])
        self.assertEqual(cases[1]['matches'], cases[0]['matches'])
        self.assertEqual(cases[1]['detections'], [cases[0]['detections'][0], cases[0]['detections'][2]])
        self.assertEqual(saved, before)
        self.assertEqual(leaf['hyg_id'], 81442)

    def test_ambiguous_or_additional_rejection_cannot_change_experiment(self):
        points, saved = self.input()
        points[0]['label'] = 'ambiguous'
        with self.assertRaisesRegex(ValueError, 'exactly'): cases_for(points, saved)
        points[0]['label'] = 'not_visible'; points[1]['label'] = 'not_visible'
        with self.assertRaisesRegex(ValueError, 'exactly'): cases_for(points, saved)

    def test_duplicate_detection_or_initial_fit_membership_rejected(self):
        points, saved = self.input()
        saved['cases'][0]['detections'].append(dict(x=5.,y=6.,rank=4))
        with self.assertRaisesRegex(ValueError, 'unique'): cases_for(points, saved)
        saved['cases'][0]['detections'].pop()
        saved['cases'][0]['matches'][0]['xy'] = [5.,6.]
        with self.assertRaisesRegex(ValueError, 'initial fit'): cases_for(points, saved)

    def test_labels_cannot_change_identity_or_pixel_coordinates(self):
        points, _ = self.input()
        original = dict(split='development', frames=[dict(id='test',points=points)])
        for key, value in [('hyg_id',123), ('detector_xy_working',[6.,7.])]:
            changed = copy.deepcopy(original); changed['frames'][0]['points'][0][key] = value
            with self.assertRaisesRegex(ValueError, 'frozen candidate'): validate_review(original, changed)

    def test_holdout_review_rejected_before_reading_inputs(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out/'protocol.json').write_text(json.dumps(dict(id='wm_r_143159342', split='holdout')))
            with patch('robustness_wide_pairs.selected') as selection:
                with self.assertRaisesRegex(ValueError, 'development'): reviewed(out)
                selection.assert_not_called()


if __name__ == '__main__': unittest.main()
