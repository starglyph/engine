"""Causal replay must change only the named factors and never select holdout."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from centroid_replay import prepare_cases
from robustness_source_trace import replay
from robustness_concentration import checked_protocol
from collection_run import write_json


class SourceReplayTests(unittest.TestCase):
    def test_factorial_replay_preserves_membership_slots_and_input(self):
        a = [dict(x=x, y=0, rank=i, flux=10-i) for i, x in enumerate((0, 20, 40))]
        b = [dict(x=x, y=0, rank=i, flux=12-i) for i, x in enumerate((41, 1, 80))]
        original = copy.deepcopy([a,b])
        cases, pairs = prepare_cases([a,b], 2)
        self.assertEqual([a,b], original)
        self.assertEqual(set(pairs), {(0,1), (2,0)})
        by_name = {c['name']: c['detections'] for c in cases}
        swapped = by_name['control_swapped_common_coordinates']
        self.assertEqual([d['x'] for d in swapped], [1,20,41])
        self.assertEqual([d['flux'] for d in swapped], [10,9,8])
        swapped = by_name['control_swapped_common_order']
        self.assertEqual([d['x'] for d in swapped], [40,20,0])
        self.assertEqual([d['flux'] for d in swapped], [10,9,8])
        self.assertEqual([d['x'] for d in by_name['common_coordinates_0_order_1']], [40,0])

    def test_holdout_rejected_before_reading_images(self):
        with self.assertRaisesRegex(ValueError, 'development'):
            replay(Path('/nonexistent'), 'wm_r_146925915', 1600, 'default')

    def test_freeze_detects_code_change_and_wrong_split(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            write_json(out/'protocol.json', dict(hashes={'code': 'old'}, split='development'))
            with patch('robustness_concentration.source_hashes', return_value={'code': 'new'}):
                with self.assertRaisesRegex(ValueError, 'changed'):
                    checked_protocol(out)
            write_json(out/'protocol.json', dict(hashes={}, split='holdout'))
            with patch('robustness_concentration.source_hashes', return_value={}):
                with self.assertRaisesRegex(ValueError, 'development'):
                    checked_protocol(out)


class TraceReportTests(unittest.TestCase):
    def test_replay_tier_metadata_survives_compaction(self):
        try:
            from PIL import Image
        except ImportError:
            self.skipTest('Pillow is needed for trace report image dimensions')
        from robustness_trace_report import compact
        from robustness_source_trace import IDS
        from collection_run import digest
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            image = out/'source.png'
            Image.new('L', (40,30)).save(image)
            for rid, tier in zip(IDS, ('default','deep')):
                write_json(out/f'{rid}-trace-input.json', dict(split='development',
                    image=str(image), image_sha256=digest(image), provenance={}, probes=[]))
                write_json(out/f'{rid}-traces.json', dict(tiers=[]))
                write_json(out/f'{rid}-1600-{tier}-replay-input.json', dict(cases=[], provenance={}))
                write_json(out/f'{rid}-1600-{tier}-replay.json', dict(tier=tier, cases=[]))
            report = compact(out)
            self.assertEqual([f['replay']['tier'] for f in report['frames']], ['default','deep'])


if __name__ == '__main__':
    unittest.main()
