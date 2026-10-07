"""Keep the nonlinear adapter restricted to scope and frozen paired objectives."""
import copy
from pathlib import Path
import unittest

from robustness_orion_tangential_fit import METHODS, RID, adapted_harness, build_cases, previous, read


class OrionNonlinearTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.original=read(previous('43-input'))
        fits=read(previous('43-fits'))
        cls.frozen=dict(id=RID,split='development',cases=[dict(method=m,
            camera=next(c for c in fits['cases'] if c['name']=='replay_'+m)['camera'],
            prior_weight=next(c for c in fits['cases'] if c['name']=='replay_'+m)['prior_weight'],
            fit_pairs=next(c for c in cls.original['cases'] if c['name']=='replay_'+m)['matches']) for m in METHODS])

    def test_matched_starts_pairs_weights_and_separate_replay(self):
        inp=build_cases(self.frozen,self.original)
        self.assertEqual(len(inp['cases']),9)
        for i in range(3):
            replay,five,seven=inp['cases'][i*3:i*3+3]
            self.assertEqual(five['initial'],seven['initial'])
            self.assertEqual(five['initial'],replay['expected'])
            self.assertEqual(five['matches'],seven['matches'])
            self.assertEqual(five['matches'],replay['matches'])
            self.assertEqual(five['prior_weight'],seven['prior_weight'])
            self.assertFalse(five['extra']);self.assertTrue(seven['extra'])
            self.assertEqual(len(five['matches']),16)

    def test_holdout_or_changed_pairs_rejected(self):
        frozen=copy.deepcopy(self.frozen);frozen['split']='holdout'
        with self.assertRaises(ValueError):build_cases(frozen,self.original)
        frozen=copy.deepcopy(self.frozen);frozen['cases'][0]['fit_pairs'][0]['xy'][0]+=1
        with self.assertRaises(ValueError):build_cases(frozen,self.original)

    def test_only_three_scope_guards_change_in_rust(self):
        source=Path(__file__).with_name('tangential_refit.rs').read_text()
        adapted=adapted_harness(source)
        restored=adapted.replace(f'assert_eq!(input.id, "{RID}");','assert_eq!(input.id, "wm_r_132162731");').replace('assert_eq!(input.cases.len(), 9);','assert_eq!(input.cases.len(), 18);').replace('assert_eq!(matches.len(), 16);','assert_eq!(matches.len(), 22);')
        self.assertEqual(source,restored)
        with self.assertRaises(ValueError):adapted_harness('changed source')


if __name__=='__main__':unittest.main()
