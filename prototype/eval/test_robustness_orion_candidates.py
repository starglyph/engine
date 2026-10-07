"""Protect correspondence equality and saved candidate audit boundaries."""
import copy
from pathlib import Path
import tempfile
import unittest

from collection_run import write_json
from robustness_orion_candidates import bind_pairs, pair_key, pair_metrics, previous, read, validate


class OrionCandidateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.trace = read(previous('41-trace'))
        cls.saved = read(previous('41-results'))

    def test_full_pairs_equal_despite_order_and_camera_difference(self):
        a, b = self.trace['candidates']
        self.assertNotEqual(a['camera'], b['camera'])
        self.assertNotEqual(a['verification']['matches'], b['verification']['matches'])
        self.assertEqual({pair_key(p) for p in a['verification']['matches']},
                         {pair_key(p) for p in b['verification']['matches']})
        self.assertEqual(len(bind_pairs(a['verification']['matches'], self.trace, self.saved)), 16)

    def test_same_detection_with_changed_world_is_not_same_pair(self):
        p = copy.deepcopy(self.trace['candidates'][0]['pairs'][0])
        p['world'][0] += .001
        with self.assertRaises(ValueError):
            bind_pairs([p], self.trace, self.saved)

    def test_changed_centroid_or_duplicate_is_not_silently_bound(self):
        p = copy.deepcopy(self.trace['candidates'][0]['pairs'][0])
        with self.assertRaises(ValueError):
            bind_pairs([p, p], self.trace, self.saved)
        p['xy'][0] += .001
        with self.assertRaises(ValueError):
            bind_pairs([p], self.trace, self.saved)

    def test_existing_projection_replays_both_rust_candidates(self):
        for c in self.trace['candidates']:
            self.assertLess(pair_metrics(c)['projection_replay_max_error_px'], 1e-8)
        bound = bind_pairs(self.trace['candidates'][0]['verification']['matches'], self.trace, self.saved)
        gamma = next(p for p in bound if p['nearest_reviewed_id'] == 24)
        self.assertFalse(gamma['reviewed_identity_agrees'])
        self.assertLess(gamma['reviewed_distance_px'], 12)

    def test_holdout_rejected_before_hash_or_photo_access(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            write_json(out/'protocol.json', dict(id='wm_r_146925915', split='holdout', holdout=True))
            with self.assertRaisesRegex(ValueError, 'development only'):
                validate(out)


if __name__ == '__main__':
    unittest.main()
