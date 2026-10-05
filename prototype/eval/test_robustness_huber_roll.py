"""Fresh synthetic noise must remain separate and constrain progression."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import robustness_huber_roll as experiment


class HuberRollTests(unittest.TestCase):
    def test_disjoint_cohorts_and_complete_case_counts(self):
        rows=experiment.cases()
        self.assertEqual(len(rows),12864)
        self.assertEqual(len({r['id'] for r in rows}),12864)
        observed={experiment.case_key(r) for r in rows if r['cohort']=='observed'}
        previous={experiment.case_key(r) for r in experiment.previous.cases()}
        self.assertEqual(observed,previous)
        self.assertEqual(sum(r['cohort']=='fresh' for r in rows),10272)
        self.assertFalse(set(experiment.COHORTS['fresh']) & set(experiment.COHORTS['observed']))

    def test_fresh_failure_cannot_be_hidden_by_observed_improvement(self):
        rows=[]
        for case in experiment.cases():
            ideal=case['sigma_px']==0
            control=.001 if ideal else 1.
            candidate=control if ideal else (.5 if case['bad_count'] else
                (1.2 if case['cohort']=='fresh' else .8))
            rows.append(dict(case,arms=dict(control=dict(rms_px=control,max_px=control),
                                          huber=dict(rms_px=candidate,max_px=candidate))))
        result=experiment.evaluate(rows)
        self.assertTrue(result['cohorts']['observed']['eligible_for_development'])
        self.assertFalse(result['cohorts']['fresh']['eligible_for_development'])
        self.assertFalse(result['eligible_for_development'])

    def test_real_input_guard_precedes_hash_access(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            for changes in [dict(split='development'),dict(real_images=True),dict(holdout=True)]:
                p=dict(split='synthetic',real_images=False,holdout=False)
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

    def test_parameter_change_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            (out/'protocol.json').write_text(json.dumps(dict(split='synthetic',real_images=False,
                holdout=False,plan=experiment.previous.PLAN,cohorts=experiment.COHORTS,
                criteria=experiment.previous.CRITERIA,case_count=12864,parameters={})))
            with self.assertRaisesRegex(ValueError,'parameters'):
                experiment.validate(out)


if __name__=='__main__':
    unittest.main()
