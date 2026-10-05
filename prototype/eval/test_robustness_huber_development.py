"""Protect acceptance logic, development scope and synthetic prerequisite."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import robustness_huber_development as experiment


class HuberDevelopmentTests(unittest.TestCase):
    def test_patch_changes_only_adapter_and_module_declarations(self):
        source=(experiment.ROOT/'prototype/crates/starglyph-core/src/solve.rs').read_text()
        result=experiment.patch_source(source)
        start=source.index('fn pose_from_solution(')
        end=source.index('/// Project verification stars',start)
        new_end=result.index('/// Project verification stars',start)
        self.assertEqual(source[:start],result[:start])
        self.assertEqual(source[end:],result[new_end:].removesuffix('\nmod robust_roll;\nmod huber_roll;\n'))
        self.assertIn('huber_roll::estimate(&rolls)?',result[start:new_end])

    def test_holdout_guard_precedes_selection_and_hash_reads(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            for plan in [dict(split='holdout',holdout=False),dict(split='development',holdout=True)]:
                (out/'protocol.json').write_text(json.dumps(plan))
                with patch.object(experiment,'records',side_effect=AssertionError('selection')):
                    with self.assertRaisesRegex(ValueError,'development selection'):
                        experiment.validate(out)

    def test_failed_synthetic_gates_prevent_build(self):
        with tempfile.TemporaryDirectory() as directory:
            synthetic=Path(directory)
            (synthetic/'results.json').write_text('{"eligible_for_development":false}')
            with patch.object(experiment,'validate_synthetic'), \
                    patch.object(experiment,'prepare_workspace',side_effect=AssertionError('build')):
                with self.assertRaisesRegex(ValueError,'synthetic gates'):
                    experiment.prepare(synthetic/'real',synthetic)

    def test_changed_frozen_binary_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            (out/'protocol.json').write_text(json.dumps(dict(split='development',holdout=False,
                ids=['dev'],hashes={'binary':'original'})))
            with patch.object(experiment,'records',return_value=[dict(id='dev')]), \
                    patch.object(experiment,'digest',return_value='changed'):
                with self.assertRaisesRegex(ValueError,'changed frozen input'):
                    experiment.validate(out)


if __name__=='__main__':
    unittest.main()
