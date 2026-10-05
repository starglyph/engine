"""Keep the diagnostic limited to frozen inputs and explicit fresh runs."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from robustness_roll_influence import FRAME, run, validate


class RollInfluenceTests(unittest.TestCase):
    def test_scope_rejected_before_file_access(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            for plan in [dict(id='other', holdout=False, cases=28),
                         dict(id=FRAME, holdout=True, cases=28),
                         dict(id=FRAME, holdout=False, cases=27)]:
                (out/'protocol.json').write_text(json.dumps(plan))
                with patch('robustness_roll_influence.digest', side_effect=AssertionError('access')):
                    with self.assertRaisesRegex(ValueError, 'fixed legacy'):
                        validate(out)

    def test_changed_frozen_input_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out/'protocol.json').write_text(json.dumps(dict(
                id=FRAME, holdout=False, cases=28, hashes={'input.json':'original'})))
            with patch('robustness_roll_influence.digest', return_value='changed'):
                with self.assertRaisesRegex(ValueError, 'changed frozen input'):
                    validate(out)

    def test_completed_attempt_not_overwritten(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            (out/'run.json').write_text('{}')
            with patch('robustness_roll_influence.validate', return_value={}), \
                    patch('robustness_roll_influence.command', side_effect=AssertionError('run')):
                with self.assertRaisesRegex(ValueError, 'already attempted'):
                    run(out)


if __name__ == '__main__':
    unittest.main()
