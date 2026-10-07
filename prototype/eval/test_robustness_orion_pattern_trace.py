"""Protect preselected identities and distinguish database absence from lookup."""
import json
from pathlib import Path
import tempfile
import unittest

from collection_run import write_json
from robustness_orion_pattern_trace import RID, choose, previous, read, target_module, validate
from robustness_orion_pattern_trace_report import classify, trace_events


class PatternTraceTests(unittest.TestCase):
    def test_selection_uses_reviewed_survivors_without_residuals(self):
        prior=read(previous('48-control-trace')) | dict(id=RID,split='development',holdout=False)
        selected=choose(read(previous('41-trace')),read(previous('41-results')),prior)
        self.assertEqual(selected[0]['detection_indices'],[0,1,2,3])
        self.assertEqual(selected[1]['detection_indices'],[2,6,16,18])
        self.assertEqual(selected[0]['catalog_ids'],[32263,37173,24378,27919])
        with self.assertRaises(ValueError):
            choose({}, {}, prior | dict(split='holdout',holdout=True))

    def test_target_hook_keeps_both_catalog_sets(self):
        text='    ids == [73486, 74164, 76644, 78029]'
        output=target_module(text,[dict(catalog_ids=[4,3,2,1]),dict(catalog_ids=[8,7,6,5])])
        self.assertIn('[[1, 2, 3, 4], [5, 6, 7, 8]].contains(&ids)',output)
        with self.assertRaises(ValueError):target_module(text+text,[])

    def fixture(self):
        return (dict(target='distributed',image=dict(ratios=[.5]*5),
                     window=dict(key_min=[4]*5,key_max=[6]*5,ratio_min=[.49]*5,ratio_max=[.51]*5,tolerance=.01),lookups=[]),
                dict(entries=[],metrics=dict(key=[7]*5,ratios=[.52]*5)))

    def test_absent_lookup_is_not_absent_database(self):
        w,i=self.fixture()
        self.assertEqual(classify(w,i)['blocker'],'not_indexed')
        i['entries']=[dict(slot=7)]
        self.assertEqual(classify(w,i)['blocker'],'lookup_window')
        i['metrics']['key']=[5]*5
        self.assertEqual(classify(w,i)['blocker'],'lookup_not_observed')

    def test_observed_rejections_and_cross_target_lookup(self):
        w,i=self.fixture();i['entries']=[dict(slot=7)]
        w['lookups']=[dict(catalog_target='local_control',verification={'pass':True})]
        self.assertEqual(classify(w,i)['blocker'],'lookup_window')
        row=dict(catalog_target='distributed');w['lookups']=[row]
        self.assertEqual(classify(w,i)['blocker'],'fov')
        row['ratios']={'pass':False}
        self.assertEqual(classify(w,i)['blocker'],'edge_ratios')
        row['verification']={'pass':False}
        self.assertEqual(classify(w,i)['blocker'],'verification_rejected')
        row['verification']['pass']=True
        self.assertEqual(classify(w,i)['blocker'],'verification_passed')

    def test_unattributed_trace_rejected(self):
        with self.assertRaisesRegex(ValueError,'unattributed'):
            trace_events('SGTRACE '+json.dumps(dict(stage='lookup',data={})),[])

    def test_holdout_rejected_before_hashes(self):
        with tempfile.TemporaryDirectory() as folder:
            out=Path(folder);write_json(out/'protocol.json',dict(id=RID,split='holdout',holdout=True))
            with self.assertRaisesRegex(ValueError,'development'):validate(out)


if __name__=='__main__':unittest.main()
