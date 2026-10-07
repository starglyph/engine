import copy
import unittest

from robustness_snow_sources import (CASE, EXPECTED_RANKS, POSITIONS, RID,
    associated_source, make_probes)
from robustness_central_sources import probe_row


class SnowSourcesTests(unittest.TestCase):
    def fixture(self):
        detections=[{'x':float(i), 'y':i+.25} for i in range(42)]
        indices=list(range(24))
        for pos, rank in zip(POSITIONS, EXPECTED_RANKS):
            indices[pos]=rank
        saved=dict(split='development',cases=[dict(name=CASE,id=RID,height=1935,
            search=[detections[i] for i in indices])],selections=[dict(name=CASE,original_indices=indices)])
        diag=[dict(width=2772,tier='default',result=dict(detections=detections))]
        return saved, diag

    def test_exact_saved_sources_without_recentering(self):
        saved,diag=self.fixture(); probes=make_probes(saved,diag)
        self.assertEqual([p['native_default_rank'] for p in probes],list(EXPECTED_RANKS))
        self.assertEqual([p['xy'][0] for p in probes],list(EXPECTED_RANKS))

    def test_reject_changed_source_or_provenance(self):
        saved,diag=self.fixture()
        for field,value in [('split','holdout')]:
            bad=copy.deepcopy(saved);bad[field]=value
            with self.assertRaises(ValueError): make_probes(bad,diag)
        bad=copy.deepcopy(saved);bad['cases'][0]['search'][19]['x']+=1
        with self.assertRaises(ValueError): make_probes(bad,diag)
        bad=copy.deepcopy(saved);bad['selections'][0]['original_indices'][19]+=1
        with self.assertRaises(ValueError): make_probes(bad,diag)

    def test_nearby_component_does_not_prove_source_identity(self):
        meta=dict(native_default_rank=37)
        row=dict(component=dict(rank_before_top_k=37),centroid_offset_original_px=0)
        self.assertTrue(associated_source(meta,row))
        row['centroid_offset_original_px']=.01
        self.assertFalse(associated_source(meta,row))
        row['centroid_offset_original_px']=0;row['component']['rank_before_top_k']=38
        self.assertFalse(associated_source(meta,row))
        self.assertFalse(associated_source(meta,dict(component=None)))

    def test_pixel_centre_lift_uses_both_actual_dimensions(self):
        sx,sy=1600/2772,1117/1935; xy=[310.22129,515.10081]
        point=[(xy[0]+.5)*sx-.5,(xy[1]+.5)*sy-.5]
        row=probe_row(dict(id='P20',xy=xy,visual_class='ambiguous'),
            dict(point=point,distance_to_component=0,component=dict(centroid=point,outcome='selected')),sx,sy)
        self.assertLess(row['centroid_offset_original_px'],1e-10)


if __name__ == '__main__':
    unittest.main()
