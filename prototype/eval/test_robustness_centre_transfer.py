"""Synthetic experiments must remain frozen and retain the complete case matrix."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import robustness_centre_transfer as audit


class CentreTransferTests(unittest.TestCase):
    def test_real_or_holdout_protocol_rejected_before_input_read(self):
        for changes in [dict(split='development'), dict(holdout=True), dict(real_images=True)]:
            with tempfile.TemporaryDirectory() as temp:
                out = Path(temp)
                protocol = dict(split='synthetic', holdout=False, real_images=False)
                protocol.update(changes)
                (out / 'protocol.json').write_text(json.dumps(protocol))
                with patch.object(audit, 'digest', side_effect=AssertionError('must not read')):
                    with self.assertRaisesRegex(ValueError, 'synthetic only'):
                        audit.validate(out)

    def test_changed_plan_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out / 'protocol.json').write_text(json.dumps(dict(
                split='synthetic', holdout=False, real_images=False, plan={})))
            with self.assertRaisesRegex(ValueError, 'plan changed'):
                audit.validate(out)

    def test_missing_cases_cannot_be_summarized_as_success(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out / 'raw.json').write_text(json.dumps(dict(cases=[])))
            with patch.object(audit, 'validate', return_value=dict(case_count=144)):
                with self.assertRaisesRegex(ValueError, 'incomplete'):
                    audit.summarize(out)

    def test_existing_run_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)
            (out / 'run.json').write_text('{}')
            with patch.object(audit, 'validate', return_value={}):
                with self.assertRaisesRegex(ValueError, 'already attempted'):
                    audit.run(out)


if __name__ == '__main__':
    unittest.main()
