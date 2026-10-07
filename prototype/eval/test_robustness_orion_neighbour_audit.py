"""Protect local review selection and keep the target outside registration fit."""
import copy
from pathlib import Path
import tempfile
import unittest

import numpy as np

from collection_run import write_json
from robustness_orion_neighbour_audit import tangent, registration, validate, RID, IDS


class NeighbourAuditTests(unittest.TestCase):
    def fixture(self):
        xy=[[-1,-1],[1,-1],[1,1],[-1,1],[.2,.3],[.5,-.4]]
        t=dict(hyg_id=99,axy_row=99,xy=[100.,200.],
            pattern=[dict(hyg_id=i+1,tangent_xy=p) for i,p in enumerate(xy)],
            extractions=[dict(row=i+1,xy=[100+40*p[0]+3*p[1],200-2*p[0]+45*p[1]]) for i,p in enumerate(xy)])
        r=dict(neighbours=[dict(hyg_id=i+1,label='visible_source',axy_row=i+1) for i in range(6)])
        return t,r

    def test_target_and_unused_neighbours_do_not_affect_fit(self):
        t,r=self.fixture();before=registration(t,r)
        altered=copy.deepcopy(t);altered['xy']=[1000.,-500.]
        altered['extractions'][-1]['xy']=[800.,400.]
        after=registration(altered,r)
        np.testing.assert_allclose(before['homography'],after['homography'],atol=0)
        self.assertEqual([p['source_id'] for p in before['points'] if p['role']=='anchor'],[1,2,3,4])
        self.assertEqual(before['points'][-1]['role'],'held_out')
        self.assertLess(before['held_out_error']['max_px'],1e-10)
        self.assertGreater(after['held_out_error']['max_px'],100)

    def test_insufficient_and_duplicate_correspondences(self):
        t,r=self.fixture()
        for p in r['neighbours'][1:]:p.update(label='ambiguous',axy_row=None)
        self.assertEqual(registration(t,r),dict(status='insufficient_visible_neighbours',count=1))
        r['neighbours'][1].update(label='visible_source',axy_row=1)
        with self.assertRaisesRegex(ValueError,'duplicate'):registration(t,r)

    def test_tan_orientation_and_center(self):
        xy=np.array(tangent([72.,7.],[[72.,7.],[72.1,7.],[72.,7.1]]))
        np.testing.assert_allclose(xy[0],[0,0],atol=1e-12)
        self.assertLess(xy[1,0],0);self.assertGreater(xy[2,1],0)

    def test_holdout_guard_before_data(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);write_json(out/'protocol.json',dict(id=RID,split='holdout',holdout=True,source_ids=list(IDS)))
            with self.assertRaisesRegex(ValueError,'development'):validate(out)


if __name__=='__main__':unittest.main()
