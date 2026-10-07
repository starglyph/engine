"""Guard experiment scope, input membership and field-level acceptance criteria."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

from collection_run import write_json
from robustness_orion_pattern_order import instrument, validate
from robustness_orion_pattern_report import METHODS, assess, attempted, compare_control, previous, read


class PatternOrderTests(unittest.TestCase):
    def test_projection_equivalence_zero_error_is_success(self):
        old = read(previous('41-native-report'))
        compare_control(old, copy.deepcopy(old))
        changed = copy.deepcopy(old); changed['camera']['k1'] += 1e-12
        with self.assertRaises(ValueError):
            compare_control(changed, old)

    def test_variant_changes_iterator_only_after_thinning(self):
        text = ('        let num_pattern_centroids = pattern_centroid_inds.len();\n'
                'BreadthFirstCombinations::<PATTERN_SIZE>::new(&pattern_centroid_inds)')
        result = instrument(text)
        self.assertIn('new(&pattern_order)', result)
        self.assertIn('farthest_first(&pattern_centroid_inds, &positions)', result)
        self.assertIn('pattern_centroid_inds.clone()', result)
        with self.assertRaises(ValueError):
            instrument(text+text)

    def test_one_centroid_group_regression_blocks_improvement(self):
        before = {m: dict(groups=dict(outside_both_starglyph_inputs=dict(count=18, rms_px=100.),
                                     left_10pct=dict(count=3, rms_px=20.))) for m in METHODS}
        after = copy.deepcopy(before)
        for m in METHODS:
            after[m]['groups']['outside_both_starglyph_inputs']['rms_px'] = 80.
        self.assertTrue(assess(before, after)['passed'])
        after['native_r8']['groups']['left_10pct']['rms_px'] = 20.6
        self.assertFalse(assess(before, after)['passed'])
        self.assertFalse(assess(before, None)['passed'])

    def test_attempt_parser_rejects_lost_survivor_and_dangling_trace(self):
        data = dict(fov_deg=68., retained=[0, 1, 2, 3], priority=[0, 3, 2, 1])
        with self.assertRaisesRegex(ValueError, 'unattributed'):
            attempted('SGORDER '+json.dumps(data))
        data['priority'] = [0, 3, 2]
        with self.assertRaisesRegex(ValueError, 'membership'):
            attempted('SGORDER '+json.dumps(data))

    def test_holdout_rejected_before_hash_access(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            write_json(out/'protocol.json', dict(id='wm_r_146925915', split='holdout', holdout=True,
                                                arms=['control', 'farthest']))
            with self.assertRaisesRegex(ValueError, 'development'):
                validate(out)


if __name__ == '__main__':
    unittest.main()
