"""Guard registration diagnostics without turning contrast into an acceptance rule."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from robustness_image_transfer import contrast, project, validate, image_path, IDS


class ImageTransferTests(unittest.TestCase):
    def test_flat_background_and_known_peak(self):
        gray=np.full((100,100),30.)
        self.assertEqual(contrast(gray,[50,50],12)['peak_over_scale'],0.)
        gray[50,50]=80.
        self.assertEqual(contrast(gray,[50,50],12)['peak_over_scale'],50.)
        self.assertEqual(contrast(gray,[2,2],12)['status'],'outside_measurement_margin')

    def test_projective_denominator_guard(self):
        np.testing.assert_allclose(project(np.eye(3),[5,7]),[5,7])
        bad=np.eye(3);bad[2]=[0,0,0]
        with self.assertRaisesRegex(ValueError,'infinity'):
            project(bad,[5,7])

    def test_scope_checked_before_hashes(self):
        with tempfile.TemporaryDirectory() as directory:
            out=Path(directory)
            for data in (dict(ids=IDS,holdout=True),dict(ids=['another'],holdout=False)):
                (out/'protocol.json').write_text(json.dumps(data))
                with patch('robustness_image_transfer.digest',side_effect=AssertionError('must not read')):
                    with self.assertRaisesRegex(ValueError,'fixed legacy'):
                        validate(out)

    def test_arbitrary_image_path_rejected(self):
        with self.assertRaisesRegex(ValueError,'only frozen'):
            image_path('../holdout')


if __name__=='__main__':
    unittest.main()
