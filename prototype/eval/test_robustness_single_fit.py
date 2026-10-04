"""Preserve reviewed membership and a common objective in the fixed-subset fit."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from robustness_single_fit import RID, build_cases, validate


class SingleFitTests(unittest.TestCase):
    def fixture(self):
        pairs = [dict(hyg_ids=[i+1], world=[1., i*.001, 0.], xy=[float(i), 20.]) for i in range(34)]
        points = [dict(hyg_id=i+1, world=p['world'], detector_xy_working=p['xy'],
                       label='visible_source' if i<25 else 'blend' if i<32 else 'ambiguous' if i==32 else 'not_visible')
                  for i, p in enumerate(pairs)]
        def case(name, subset):
            return dict(name=name, stages=[dict(stage='rematch_10_before', camera=dict(k1=0.), pairs=subset),
                dict(stage='rematch_10_after', camera=dict(k1=.1), pairs=subset)])
        return dict(id=RID, split='development', cases=[case('drop_detection_all_stages', pairs),
                    case('drop_second_leaf', pairs[:-1])]), points

    def test_only_visible_sources_selected_without_changing_data(self):
        trace, points = self.fixture(); before = copy.deepcopy(trace)
        cases = build_cases(trace, points)
        self.assertEqual([len(c['matches']) for c in cases], [34,33,25])
        self.assertEqual(cases[2]['matches'], cases[0]['matches'][:25])
        self.assertEqual([c['prior_weight'] for c in cases], [None]*3)
        self.assertEqual([c['initial'] for c in cases], [dict(k1=0.)]*3)
        self.assertIsNone(cases[2]['expected'])
        cases[2]['matches'][0]['xy'][0] = 500
        self.assertEqual(trace, before)

    def test_common_initial_pose_is_required(self):
        trace, points = self.fixture()
        trace['cases'][1]['stages'][0]['camera']['k1'] = .02
        with self.assertRaisesRegex(ValueError, 'common initial'): build_cases(trace, points)

    def test_coordinate_or_identity_substitution_rejected(self):
        for key, value in [('hyg_id', 90), ('detector_xy_working', [9.,9.]), ('world',[0.,0.,1.])]:
            trace, points = self.fixture(); points[0][key] = value
            with self.assertRaisesRegex(ValueError, 'identity mismatch'): build_cases(trace, points)

    def test_changed_review_count_cannot_silently_change_subset(self):
        trace, points = self.fixture(); points[0]['label'] = 'ambiguous'
        with self.assertRaisesRegex(ValueError, '25-source'): build_cases(trace, points)

    def test_holdout_rejected_before_hash_reads(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out/'protocol.json').write_text(json.dumps(dict(id=RID, split='holdout', holdout=True)))
            with self.assertRaisesRegex(ValueError, 'development'): validate(out)


if __name__ == '__main__': unittest.main()
