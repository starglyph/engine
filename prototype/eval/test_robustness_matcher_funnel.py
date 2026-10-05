"""Balance and attribution guards for observation-only solver counters."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from collection_run import write_json
from robustness_matcher_funnel import COUNTERS, RID, POSITIVE, parse_trace, validate, validate_counts, aggregate, instrument


def event(stage,data):return 'SGTRACE '+json.dumps(dict(stage=stage,data=data))+'\n'


def fov():
    return dict(fov_deg=22.,retained=5,patterns=5,hash_candidates=8,fov_pass=4,ratios_pass=3,
        svd_pass=2,verifications=2,probability_pass=0,finalized=0,nonfinite_probability=0,
        matches_histogram={'3':1,'4':1},best_ratio=2.,best={'probability':.2,'threshold':.1})


class FunnelTests(unittest.TestCase):
    def test_pairs_fovs_and_aggregates_without_crossing_attempts(self):
        text='noise\n'
        for name in ['control','manual']:
            text+=event('attempt',dict(case=name,k=8))
            text+=event('thinning',dict(fov=22.,kept_input_indices=[0,1,2,3,4]))
            text+=event('funnel',fov())
        parsed=parse_trace(text)
        self.assertEqual([a['context']['case']for a in parsed],['control','manual'])
        summary=aggregate([a['fovs'][0]for a in parsed])
        self.assertEqual(summary['verifications'],4)
        self.assertEqual(summary['matches_histogram'],{'3':2,'4':2})
        self.assertEqual(summary['best']['best_ratio'],2.)

    def test_rejects_inconsistent_stages_and_histogram(self):
        for changes in [dict(fov_pass=9),dict(verifications=3),dict(finalized=1),dict(retained=3),dict(patterns=-1)]:
            with self.assertRaises(ValueError):validate_counts(fov()|changes)
        with self.assertRaisesRegex(ValueError,'histogram'):
            validate_counts(fov()|dict(matches_histogram={'3':1}))
        empty=dict.fromkeys(COUNTERS,0)|dict(retained=3,matches_histogram={})
        validate_counts(empty)

    def test_rejects_unpaired_or_wrong_fov_events(self):
        start=event('attempt',dict(case='control',k=8))
        thinning=event('thinning',dict(fov=22.,kept_input_indices=[0,1,2,3,4]))
        for text in [event('funnel',fov()),start+thinning,start+thinning+event('funnel',fov()|dict(fov_deg=23.)),
                     start+thinning+start,start+thinning+thinning]:
            with self.assertRaises(ValueError):parse_trace(text)

    def test_holdout_rejected_before_input_access(self):
        with tempfile.TemporaryDirectory()as folder:
            out=Path(folder);write_json(out/'protocol.json',dict(ids=[RID,POSITIVE],split='holdout',holdout=True))
            with patch('robustness_matcher_funnel.selected')as select:
                with self.assertRaisesRegex(ValueError,'development'):validate(out)
                select.assert_not_called()

    def test_unreviewed_instrumentation_anchor_rejected(self):
        with self.assertRaisesRegex(ValueError,'anchor must occur once'):instrument('changed upstream')
