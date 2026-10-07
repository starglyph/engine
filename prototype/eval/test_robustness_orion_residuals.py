"""Protect fixed membership and meaning of the descriptive vector metrics."""
import copy
import unittest

import numpy as np

from robustness_orion_residuals import masks_for, previous, read
from robustness_single22_residuals import describe


class OrionResidualTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources=read(previous('40-results'))['sources']
        cls.selection=read(previous('43-results'))['selection']

    def test_partitions_preserve_every_original_source(self):
        masks=masks_for(self.sources,self.selection)
        self.assertEqual(sum(masks['fit16/all_reviewed']),16)
        self.assertEqual(sum(masks['check31/all_reviewed']),31)
        for name in [k.split('/',1)[1] for k in masks if k.startswith('all47/')]:
            np.testing.assert_array_equal(masks['fit16/'+name]|masks['check31/'+name],masks['all47/'+name])
            self.assertFalse((masks['fit16/'+name]&masks['check31/'+name]).any())
        for part in ('all47','fit16','check31'):
            cells=[v for k,v in masks.items() if k.startswith(part+'/cell_')]
            radii=[v for k,v in masks.items() if k.startswith(part+'/radius_')]
            np.testing.assert_array_equal(np.sum(cells,axis=0),masks[part+'/all_reviewed'])
            np.testing.assert_array_equal(np.sum(radii,axis=0),masks[part+'/all_reviewed'])

    def test_overlapping_fit_check_rejected(self):
        s=copy.deepcopy(self.selection);s['check_ids'][0]=s['fit_ids'][0]
        with self.assertRaises(ValueError):masks_for(self.sources,s)

    def test_no_groups_selected_by_residuals(self):
        sources=copy.deepcopy(self.sources)
        for s in sources:s['predicted_xy']={'unavailable':None}
        a,b=masks_for(sources,self.selection),masks_for(self.sources,self.selection)
        for k in a:np.testing.assert_array_equal(a[k],b[k])

    def test_legacy_edge_change_rejected(self):
        sources=copy.deepcopy(self.sources);sources[0]['groups'].append('left_10pct')
        with self.assertRaises(ValueError):masks_for(sources,self.selection)

    def test_empty_group_and_centroid_difference_interpretation(self):
        positions={'external':[[1.,1.],[3.,3.]],'native':[[2.,1.],[4.,3.]]}
        r=describe(positions,np.array([[4.,5.],[7.,8.]]),{'all':np.ones(2,bool),'empty':np.zeros(2,bool)},10,10,'external')
        self.assertEqual(r['methods']['external']['groups']['empty'],{'count':0})
        self.assertAlmostEqual(r['centroid_agreement']['native']['all']['camera_disagreement_rms_px'],1.)
        for data in r['methods'].values():
            g=data['groups']['all']
            self.assertAlmostEqual(g['rms_px']**2,g['radial_rms_px']**2+g['tangential_rms_px']**2)


if __name__=='__main__':unittest.main()
