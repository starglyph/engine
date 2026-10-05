"""Trace attribution and split/freeze protection, without a solver launch."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from collection_run import digest, write_json
from robustness_thinning_trace import RID, POSITIVE, read_thinning, validate


def event(stage, data):
    return 'SGTRACE ' + json.dumps(dict(stage=stage, data=data)) + '\n'


class ThinningTests(unittest.TestCase):
    def test_preserves_attempt_and_sweep_identity(self):
        text = 'ignored\n'
        for name in ['native', 'reduced']:
            text += event('attempt', dict(case=name, k=8))
            for fov, indices in [(22., [0, 2, 3]), (23., [0, 2, 3, 5])]:
                text += event('thinning', dict(fov=fov, kept_input_indices=indices))
        result = read_thinning(text)
        self.assertEqual([r['context']['case'] for r in result], ['native', 'reduced'])
        self.assertEqual([len(s['kept_input_indices']) for s in result[0]['sweeps']], [3, 4])

    def test_rejects_missing_context_and_invalid_indices(self):
        thinning = lambda ids: event('thinning', dict(fov=22., kept_input_indices=ids))
        with self.assertRaisesRegex(ValueError, 'unattributed'):
            read_thinning(thinning([0]))
        for ids in [[0, 0], [-1], [8]]:
            with self.assertRaisesRegex(ValueError, 'invalid retained'):
                read_thinning(event('attempt', dict(k=8)) + thinning(ids))
        with self.assertRaisesRegex(ValueError, 'missing thinning'):
            read_thinning(event('attempt', dict(k=8)))

    def test_holdout_rejected_before_input_access(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            write_json(out / 'protocol.json', dict(split='holdout', ids=[RID, POSITIVE], holdout=True))
            with patch('robustness_thinning_trace.selection') as select:
                with self.assertRaisesRegex(ValueError, 'development'):
                    validate(out)
                select.assert_not_called()

    def test_mutated_frozen_input_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder); source = out / 'input'
            source.write_text('before')
            write_json(out / 'protocol.json', dict(split='development', ids=[RID, POSITIVE], holdout=False,
                       hashes={str(source): digest(source)}))
            source.write_text('after')
            with patch('robustness_thinning_trace.selection'):
                with self.assertRaisesRegex(ValueError, 'frozen input changed'):
                    validate(out)
