"""The full frozen matrix and clean-data gates must constrain progression."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import robustness_roll_synthetic as experiment


class RollSyntheticTests(unittest.TestCase):
    def test_case_matrix_contains_clean_controls_and_both_corruptions(self):
        rows=experiment.cases()
        self.assertEqual(len(rows),2592)
        self.assertEqual(len({r['id'] for r in rows}),2592)
        self.assertEqual(sum(r['sigma_px']==0 for r in rows),32)
        self.assertEqual(sum(r['sigma_px']>0 and r['bad_count']==0 for r in rows),512)
        for row in rows:
            self.assertIn(row['bad_count'],[0,1,row['pairs']//4])

    def test_scope_guard_precedes_hash_access(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            for changes in [dict(split='development'),dict(holdout=True),dict(real_images=True)]:
                p=dict(split='synthetic',holdout=False,real_images=False)
                p.update(changes)
                (out/'protocol.json').write_text(json.dumps(p))
                with patch.object(experiment,'digest',side_effect=AssertionError('access')):
                    with self.assertRaisesRegex(ValueError,'synthetic only'):
                        experiment.validate(out)

    def test_missing_cases_cannot_be_summarized(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            (out/'raw.json').write_text('{"cases":[]}')
            with patch.object(experiment,'validate',return_value={}):
                with self.assertRaisesRegex(ValueError,'incomplete'):
                    experiment.summarize(out)

    def test_clean_regression_blocks_despite_contamination_improvement(self):
        rows=[]
        for case in experiment.cases():
            control=.001 if case['sigma_px']==0 else 1.
            median=control if case['sigma_px']==0 else (.5 if case['bad_count'] else 1.2)
            rows.append(dict(case,arms=dict(control=dict(rms_px=control,max_px=control),
                                          median=dict(rms_px=median,max_px=median))))
        result=experiment.evaluate(rows,experiment.CRITERIA)
        self.assertTrue(result['gates']['contaminated_groups'])
        self.assertFalse(result['gates']['clean_groups'])
        self.assertFalse(result['eligible_for_development'])

    def test_fixed_criteria_cannot_change_silently(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            (out/'protocol.json').write_text(json.dumps(dict(split='synthetic',holdout=False,
                real_images=False,plan=experiment.PLAN,criteria={},case_count=2592)))
            with self.assertRaisesRegex(ValueError,'criteria changed'):
                experiment.validate(out)


if __name__=='__main__':
    unittest.main()
