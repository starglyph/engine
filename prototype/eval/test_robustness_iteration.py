import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import robustness_iteration as iteration


class ProtocolTests(unittest.TestCase):
    def test_holdout_requires_protocol_before_creating_output(self):
        with tempfile.TemporaryDirectory() as folder:
            out = Path(folder)/'holdout'
            with patch('sys.argv',['iteration','--split','holdout','--out-dir',str(out)]):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    iteration.main()
            self.assertEqual(error.exception.code,2)
            self.assertFalse(out.exists())

    def test_changed_frozen_code_is_rejected_before_creating_output(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            protocol = root/'protocol.json'
            protocol.write_text(json.dumps({'hashes':{'detector':'old'}}))
            out = root/'holdout'
            with patch('sys.argv',['iteration','--split','holdout','--out-dir',str(out),
                                   '--protocol',str(protocol)]), patch.object(iteration,'frozen_hashes',
                                                                            return_value={'detector':'new'}):
                with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
                    iteration.main()
            self.assertEqual(error.exception.code,2)
            self.assertFalse(out.exists())


if __name__ == '__main__':
    unittest.main()
