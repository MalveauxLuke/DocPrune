import itertools
import sys
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from stage0_owned_overlap import owned_patch_fractions
from stage0_mask_banks import CostConditionalSampler, common_budget
from stage0_preference_yield import preference_yield


class MaskPreparationTests(unittest.TestCase):
    def test_fractional_ownership_keeps_small_action(self):
        grid=[[0,y,x,2,2] for y in range(2) for x in range(2)]
        fraction=owned_patch_fractions([[0,0,1,1]],grid,[0,0,0,1],2)
        np.testing.assert_allclose(fraction,[[.75,.25]])
        # Neither an enclosing box nor a single-center assignment is substituted.
        fractions=owned_patch_fractions([[0,0,.5,1],[.5,0,1,1]],grid,[0,1,1,0],2)
        np.testing.assert_allclose(fractions,[[.5,.5],[.5,.5]])

    def test_partition_matches_enumeration_and_cost_gauge(self):
        costs=np.array([1,1,2,3]);theta=np.array([-.4,.2,.8,-.1]);b=3
        masks=np.array([m for m in itertools.product([0,1],repeat=4) if costs@m==b])
        expected=np.exp(masks@theta).sum()
        sampler=CostConditionalSampler(costs,theta,b)
        self.assertAlmostEqual(np.exp(sampler.logz[-1,b]),expected,places=12)
        shifted=CostConditionalSampler(costs,theta+1.7*costs,b)
        self.assertAlmostEqual(shifted.logz[-1,b]-sampler.logz[-1,b],1.7*b,places=12)
        a=np.random.default_rng(1);z=np.random.default_rng(1)
        for _ in range(50):np.testing.assert_equal(sampler.draw(a),shifted.draw(z))
        self.assertEqual(common_budget([3,6],5),3)
        with self.assertRaises(ValueError):CostConditionalSampler([2,4],[0,0],3)

    def test_centering_changes_unequal_region_count_law(self):
        costs=np.array([1,1,2]);masks=np.array([[1,1,0],[0,0,1]])
        zero=CostConditionalSampler(costs,[0,0,0],2)
        centered=CostConditionalSampler(costs,[-1,-1,-1],2)
        probs=np.exp(masks@centered.theta-centered.logz[-1,2])
        self.assertAlmostEqual(np.exp(-zero.logz[-1,2]),.5)
        self.assertLess(probs[0],probs[1])

    def test_sampling_probabilities_on_small_exact_population(self):
        sampler=CostConditionalSampler([1,1,2],[0,0,1],2)
        rng=np.random.default_rng(184)
        empirical=np.mean([sampler.draw(rng)[2] for _ in range(10000)])
        self.assertLess(abs(empirical-np.exp(1)/(1+np.exp(1))),.015)

    def test_S_disagreements_require_two_admissible(self):
        # Two admissible alternatives reverse G/C; the third is inadmissible.
        gs=np.array([[0.,0.],[-.05,-1.],[-2.,-10.]])
        result=preference_yield(gs,[0,0],.1,.001)
        self.assertEqual(result['admissible_masks'],2)
        self.assertEqual(result['all_pairs_S_disagreement_upper_bound'],1)
        self.assertEqual(result['strict_G_goldaware_disagreements'],1)
        result=preference_yield(gs,[0,0],0.,.001)
        self.assertEqual(result['strict_G_goldaware_disagreements'],0)
        subset=preference_yield(gs,[0,0],.1,.001,pairs=[[0,2],[2,0],[1,2]])
        self.assertEqual(subset['comparisons'],2)
        self.assertEqual(subset['strict_G_goldaware_disagreements'],0)


if __name__=='__main__':unittest.main()
