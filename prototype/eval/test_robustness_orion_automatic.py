"""Verify automatic pair integrity and genuinely unused assessment membership."""
import copy
import unittest

from robustness_orion_automatic import assessment_groups, build_input, transfer, previous, read


class AutomaticOrionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sources=read(previous('40-results'))['sources']
        cls.pairs=read(previous('41-results'))['stages'][0]['pairs']
        cls.legacy=read(previous('44-input'))['groups']
        cls.trace=read(previous('41-trace'))
        cls.old_input=read(previous('45-nonlinear-input'))
        cls.old_results=read(previous('45-nonlinear-results'))

    def test_physical_component_excluded_even_with_different_identity(self):
        groups=assessment_groups(self.sources,self.pairs,self.legacy)
        self.assertEqual(groups['overlap_auto/all_reviewed'],[1,4,5,8,9,15,22,24,36])
        self.assertEqual(len(groups['outside_auto/all_reviewed']),38)
        # Gamma Lep has a different assigned HYG ID but the same measured center.
        self.assertNotIn(24,groups['outside_auto/all_reviewed'])

    def test_all_centroid_methods_participate_in_exclusion(self):
        sources=copy.deepcopy(self.sources)
        sources[0]['coordinates']['native_r8']=self.pairs[1]['xy']
        groups=assessment_groups(sources,self.pairs,self.legacy)
        self.assertNotIn(0,groups['outside_auto/all_reviewed'])

    def test_identity_excluded_even_if_measured_position_is_far(self):
        sources=copy.deepcopy(self.sources);sources[0]['hyg_id']=self.pairs[1]['hyg_id']
        groups=assessment_groups(sources,self.pairs,self.legacy)
        self.assertNotIn(0,groups['outside_auto/all_reviewed'])

    def test_automatic_input_is_exact_saved_verification(self):
        inp=build_input(self.trace,self.old_input,self.old_results)
        self.assertEqual(len(inp['cases']),9)
        replay,five,seven=inp['cases'][6:]
        for c in [replay,five,seven]:
            self.assertEqual(c['matches'],self.trace['input']['verification']['matches'])
        self.assertEqual(five['initial'],seven['initial'])
        self.assertEqual(five['prior_weight'],seven['prior_weight'])
        self.assertEqual(replay['initial'],self.trace['input']['camera'])
        self.assertEqual(replay['expected'],self.trace['steps'][0]['camera'])
        self.assertFalse(five['extra']);self.assertTrue(seven['extra'])
        changed=copy.deepcopy(self.trace);changed['steps'][0]['fit_matches'][0]['xy'][0]+=1
        with self.assertRaises(ValueError):build_input(changed,self.old_input,self.old_results)

    def test_holdout_bundle_rejected(self):
        old=copy.deepcopy(self.old_input);old['split']='holdout'
        with self.assertRaises(ValueError):build_input(self.trace,old,self.old_results)

    def test_all_reviewed_regression_blocks_outside_improvement(self):
        before={'outside_auto/all_reviewed':dict(count=38,rms_px=100),'all47/right_10pct':dict(count=3,rms_px=10),'outside_auto/empty':dict(count=0)}
        after={'outside_auto/all_reviewed':dict(count=38,rms_px=50),'all47/right_10pct':dict(count=3,rms_px=11),'outside_auto/empty':dict(count=0)}
        self.assertFalse(transfer(before,after)['passed'])
        after['all47/right_10pct']['rms_px']=10.4
        self.assertTrue(transfer(before,after)['passed'])


if __name__=='__main__':unittest.main()
