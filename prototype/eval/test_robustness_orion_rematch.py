"""Keep catalogue filtering isolated and independent evaluation uncontaminated."""
import copy
from pathlib import Path
import tempfile
import unittest

from collection_run import ROOT, write_json
from robustness_orion_rematch import replay_module, validate, RID
from robustness_orion_rematch_report import unused_sources, pair_summary, verdict, METHODS


class RematchTests(unittest.TestCase):
    def test_generated_path_preserves_refinement_and_final_full_catalogue(self):
        source = (ROOT/'prototype/crates/starglyph-core/src/solve.rs').read_text()
        code = replay_module(source)
        self.assertIn('let mut refined = refine_pose(initial, initial_matches);', code)
        self.assertIn('for radius in WIDE_REMATCH_RADII_PX', code)
        self.assertIn('pose_arm == "control" || *mag <= 5.5', code)
        self.assertIn('run(&initial, &rows, &rematch_stars, &dets)', code)
        self.assertIn('let v = match_predictions(&p, &stars, &dets, FINAL_RADIUS_PX);', code)
        self.assertIn(f'assert_eq!(plan["id"], "{RID}")', code)
        self.assertNotIn('for match_arm', code)
        with self.assertRaises(ValueError):
            replay_module(source.replace('for radius in WIDE_REMATCH_RADII_PX', 'for radius in []').replace(
                '                refined = refine_pose(&refined, &rematch.matches);', '                refined = other();'))

    def test_common_holdback_excludes_catalogue_or_any_centroid_overlap(self):
        sources = [dict(id=i, hyg_id=100+i, coordinates={m: [100.*i, 0.] for m in METHODS}) for i in range(4)]
        sources[2]['coordinates']['native_r12'] = [0., 0.]
        pairs = [dict(hyg_id=101, xy=[0., 0.])]
        self.assertEqual(unused_sources(sources, pairs), [3])

    def test_pair_extent_counts_only_matching_id_as_agreement(self):
        pairs = [dict(xy=[i*100., 0.], reviewed_identity_agrees=v) for i, v in enumerate([True, False, None, True])]
        result = pair_summary(pairs)
        self.assertEqual([result[k]['count'] for k in ['agree', 'disagree', 'unknown']], [2, 1, 1])
        self.assertEqual(result['agree']['bounds_xyxy'], [0., 0., 300., 0.])

    def test_verdict_requires_geometry_and_correct_pair_coverage(self):
        before = dict(pair_summary={'agree': dict(count=10, width_fraction=.2), 'disagree': dict(count=2)},
            geometry={m: dict(groups={'outside_common/all_reviewed': dict(count=20, rms_px=100.),
                                     'all47/left': dict(count=3, rms_px=200.)}) for m in METHODS})
        after = copy.deepcopy(before); after['pair_summary']['agree']['width_fraction'] = .4
        for m in METHODS:
            after['geometry'][m]['groups']['outside_common/all_reviewed']['rms_px'] = 80.
        self.assertTrue(verdict(before, after)['passed'])
        after['pair_summary']['disagree']['count'] = 3
        self.assertFalse(verdict(before, after)['passed'])
        after['pair_summary']['disagree']['count'] = 2
        after['geometry'][METHODS[0]]['groups']['all47/left']['rms_px'] = 201.
        self.assertFalse(verdict(before, after)['passed'])

    def test_holdout_rejected_before_artifacts(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder); write_json(out/'protocol.json', dict(id=RID, split='holdout', holdout=True))
            with self.assertRaisesRegex(ValueError, 'development'):
                validate(out)


if __name__ == '__main__':
    unittest.main()
