"""Check fixed membership and vector comparisons without refitting a camera."""
import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import robustness_single22_residuals as report


class Single22ResidualTests(unittest.TestCase):
    def test_alternate_centroids_do_not_change_group_membership(self):
        positions=dict(detector=[[9.,30.],[11.,30.]],external=[[11.,30.],[9.,30.]])
        prediction=np.array([[14.,30.],[16.,30.]])
        masks=dict(all_reviewed=np.array([True,True]),left_10pct=np.array([True,False]),empty=np.array([False,False]))
        result=report.describe(positions,prediction,masks,100,100,'detector')
        self.assertEqual(result['methods']['external']['groups']['left_10pct']['rms_px'],3.)
        self.assertEqual(result['methods']['detector']['groups']['left_10pct']['rms_px'],5.)
        self.assertEqual(result['methods']['external']['groups']['empty'],{'count':0})
        self.assertEqual(result['centroid_agreement']['external']['left_10pct']['field_cosine'],1.)

    def test_centroid_disagreement_is_not_camera_accuracy(self):
        positions=dict(detector=[[10.,20.]],external=[[10.,20.1]])
        result=report.describe(positions,np.array([[20.,20.]]),{'all':np.array([True])},100,100,'detector')
        agree=result['centroid_agreement']['external']['all']
        self.assertGreater(agree['field_cosine'],.999)
        self.assertAlmostEqual(agree['camera_disagreement_rms_px'],.1)
        self.assertEqual(result['methods']['detector']['groups']['all']['rms_px'],10.)

    def test_frozen_identities_checked_before_projection(self):
        review=report.read(report.prior('27-results'))
        inputs=report.read(report.prior('28-input'))
        geom=report.read(report.prior('13-geometry'))
        alternate=report.read(report.prior('5-geometry'))
        saved=report.read(report.prior('28-results'))
        before=copy.deepcopy((review,inputs,geom,alternate,saved))
        data=report.datasets(review,inputs,geom,alternate,saved)
        self.assertEqual(len(data['fit22']['sources']),22)
        self.assertEqual(len(data['probes35']['sources']),35)
        self.assertEqual((review,inputs,geom,alternate,saved),before)
        inputs['cases'][1]['matches'][0]['world'][0]+=1e-5
        with self.assertRaisesRegex(ValueError,'identity'):report.datasets(review,inputs,geom,alternate,saved)

    def test_holdout_rejected_before_hash_or_manifest_reads(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder)
            (out/'protocol.json').write_text(json.dumps(dict(id=report.RID,split='holdout',holdout=True,arms=list(report.ARMS))))
            with patch.object(report,'selected',side_effect=AssertionError('must not read')):
                with self.assertRaisesRegex(ValueError,'development'):report.validate(out)


if __name__=='__main__':
    unittest.main()
