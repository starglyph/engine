"""Protect the experimental split, frozen inputs, and trace attribution."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from collection_run import digest, write_json
from tetra3_footprint import run
from tetra3_internal import insert, prepare
from tetra3_internal_report import read_trace


class FootprintTests(unittest.TestCase):
    def test_holdout_is_rejected_before_launch(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            write_json(out / 'protocol.json', {'split': 'holdout'})
            with patch('tetra3_footprint.subprocess.run') as launch:
                with self.assertRaisesRegex(ValueError, 'development'):
                    run(out)
                launch.assert_not_called()

    def test_mutated_frozen_input_prevents_launch(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)
            source = out / 'input'
            source.write_text('frozen')
            write_json(out / 'protocol.json', {
                'split': 'development', 'hashes': {str(source): digest(source)}})
            source.write_text('changed')
            with patch('tetra3_footprint.subprocess.run') as launch:
                with self.assertRaisesRegex(ValueError, 'frozen input changed'):
                    run(out)
                launch.assert_not_called()

    def test_unreviewed_upstream_is_rejected_before_copy(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            source = root / 'registry/src/solver/solve.rs'
            source.parent.mkdir(parents=True)
            source.write_text('different version')
            with self.assertRaisesRegex(ValueError, 'unexpected tetra3'):
                prepare(root / 'out', root / 'registry')
            self.assertFalse((root / 'out').exists())

    def test_missing_or_ambiguous_instrumentation_anchor_fails(self):
        for text in ('none', 'anchor anchor'):
            with self.assertRaisesRegex(ValueError, 'must occur once'):
                insert(text, 'anchor', 'observation')

    def test_trace_does_not_mix_attempts_or_rejected_lookups(self):
        rows = [
            ('attempt', {'case': 'a'}), ('lookup', {'id': 'discarded'}),
            ('ratios', {'pass': False}), ('lookup', {'id': 'used'}),
            ('ratios', {'pass': True}), ('verification', {'pass': False}),
            ('attempt', {'case': 'b'}), ('lookup', {'id': 'second'}),
            ('verification', {'pass': True}),
        ]
        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / 'trace.log'
            log.write_text('ordinary log\n' + '\n'.join(
                'SGTRACE ' + json.dumps({'stage': stage, 'data': data}) for stage, data in rows))
            result = read_trace(log)['target_verifications']
        self.assertEqual([r['lookup']['id'] for r in result], ['used', 'second'])
        self.assertEqual([r['attempt']['case'] for r in result], ['a', 'b'])
        self.assertNotIn('ratios', result[1])


if __name__ == '__main__':
    unittest.main()
