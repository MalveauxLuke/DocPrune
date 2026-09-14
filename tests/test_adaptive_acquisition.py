import copy
from itertools import combinations, product
import math
import os
from pathlib import Path
import unittest
import numpy as np
from docprune.adaptive_acquisition import (CONFIG, ConditionalSampler, Controller, draw_quartet,
    make_groups, mask_id, preference, scores, split_plan, utility)
from docprune.acquisition_io import read_record, validate_package


def fixture():
    masks = []
    for selected in combinations(range(8), 4):
        masks.append([int(i in selected) for i in range(8)])
    public = {'case': 'Q01', 'source_ids': [str(i) for i in range(8)], 'costs': [1254]*8,
              'budget': 5016, 'owners': [i for i in range(8) for _ in range(1254)],
              'priority': list(range(8)), 'regions': [{'kind': 'residual', 'centroid': [i//2, i%2, 0, 57, 44]} for i in range(8)],
              'banks': {'R': masks[:32], 'A': masks[20:52]}}
    reader = {'targets': [[1], [2]], 'expected_generated_ids': [2], 'reference_likelihoods': [-1., -2.],
              'images': [{}, {}, {}, {}], 'repetition_penalty': 1.05}
    return {'public': public, 'reader': reader}


def drive(generator, response):
    requests = []
    try:
        request = next(generator)
        while True:
            requests.append(request)
            request = generator.send(response(request))
    except StopIteration:
        return requests


class PolicyTests(unittest.TestCase):
    def test_sampler_partition_matches_enumeration(self):
        costs = [1, 2, 2, 3, 5]
        feasible = [z for z in product((0, 1), repeat=5) if sum(c*x for c, x in zip(costs, z)) == 5]
        sampler = ConditionalSampler(costs, 5)
        self.assertAlmostEqual(math.exp(sampler.logz[-1, 5]), len(feasible))
        rng = np.random.default_rng(3)
        counts = {z: 0 for z in feasible}
        for _ in range(12000):
            counts[tuple(sampler.draw(rng).astype(int))] += 1
        self.assertLess(max(abs(v-12000/len(feasible)) for v in counts.values()), 180)
        with self.assertRaises(ValueError):
            ConditionalSampler([2, 4], 3)

    def test_group_opening_restores_exchange_and_fixed_background(self):
        costs = np.array([2, 3, 1, 4, 7, 9])
        u, v = np.array([1, 1, 0, 0, 1, 0]), np.array([0, 0, 1, 1, 1, 0])
        # Partial costs differ, so these original actions truly cannot recombine.
        self.assertIsNone(split_plan(u, v, costs, [[0,1], [2,3], [4], [5]]))
        costs = np.array([2, 3, 2, 3, 7, 9])
        plan = split_plan(u, v, costs, [[0,1], [2,3], [4], [5]])
        self.assertEqual(plan['cost'], 2)
        self.assertEqual(len(plan['opened']), 2)
        w, x, chosen = draw_quartet(u, v, costs, plan, np.random.default_rng(1))
        self.assertEqual(costs@w, costs@u)
        self.assertEqual(costs@x, costs@v)
        self.assertTrue(np.array_equal(w[4:], u[4:]))
        self.assertTrue(np.array_equal(x[4:], u[4:]))
        self.assertTrue(np.array_equal(w.astype(int)+x, u+v))

    def test_seed_can_bisect_group_without_rewriting_parent(self):
        u, v = np.array([1,0,1,0]), np.array([0,1,0,1])
        plan = split_plan(u, v, np.ones(4, dtype=int), [[0,1], [2,3]])
        self.assertEqual(plan['units'], [[[0],[2]], [[1],[3]]])
        self.assertEqual(u.tolist(), [1,0,1,0])

    def controller_pair(self, a_y, b_y):
        p = {'case': 'Q12', 'costs': [1]*4, 'budget': 2, 'priority': [0]*4,
             'regions': [{'kind': 'residual', 'centroid': [0, i, 0, 4, 1]} for i in range(4)],
             'banks': {'A': [[1,1,0,0], [0,0,1,1], [1,0,1,0], [0,1,0,1], [1,0,0,1], [0,1,1,0]]}}
        c = Controller(p, (-1, -2))
        u, v = np.array([1,1,0,0], dtype=bool), np.array([0,0,1,1], dtype=bool)
        a, b = mask_id(u), mask_id(v)
        c.observed = {a: a_y, b: b_y}
        c.masks = {a: u, b: v}
        c.scale = np.array([.01, .01])
        c._edge(a, b)
        return c

    def test_flat_parents_still_get_audited_and_reveal_cancellation(self):
        c = self.controller_pair((-1, -2), (-1, -2))
        self.assertEqual(c._candidates('focus', 0), [])
        self.assertEqual(len(c._candidates('audit', 0)), 1)
        def outcome(request):
            return (-.5, -2) if request['mask'][0] else (-1.5, -2)
        requests = drive(c._followup(0, 1, 'audit'), outcome)
        self.assertEqual(len(requests), 2)
        self.assertTrue(c.diagnostics[0]['productive'])
        self.assertTrue(any(abs(v) > .01 for v in c.diagnostics[0]['delta_first']))

    def test_C_flat_G_S_comovement_is_responsive(self):
        c = self.controller_pair((-1, -2), (-.5, -1.5))
        self.assertEqual(len(c._candidates('focus', 0)), 1)

    def test_background_reversal_remains_eligible(self):
        c = self.controller_pair((-1, -2), (-1, -2))
        drive(c._followup(0, 1, 'audit'), lambda request: (-1, -3))
        self.assertTrue(c.diagnostics[0]['reversal'])
        self.assertEqual(c.diagnostics[0]['descendant_discount'], 1.)

    def test_consistent_exhausted_pair_discounts_without_removing_children(self):
        c = self.controller_pair((-1, -2), (-1, -3))
        drive(c._followup(0, 0, 'focus'), lambda request: (-1, -2.5))
        self.assertFalse(c.diagnostics[0]['productive'])
        self.assertEqual(c.diagnostics[0]['descendant_discount'], .5)
        descendants = [e for e in c.edges.values() if e['depth'] == 1]
        self.assertEqual(len(descendants), 4)
        self.assertTrue(all(e['discount'] == .5 for e in descendants))
        self.assertTrue(c._candidates('audit', 0))

    def test_scores_and_guarded_preference(self):
        self.assertEqual(scores([-3, -1, -2]), (-1, -2))
        self.assertGreater(utility((-.99, -1), -1), utility((-1.2, -100), -1))
        self.assertEqual(preference((-1.099, -1), (-1.101, -10), -1), 0)
        with self.assertRaises(ValueError):
            scores([float('nan'), -1])

    def test_all_quiet_has_explicit_fallbacks_but_still_32_distinct(self):
        c = Controller(fixture()['public'], (-1, -2))
        requests = drive(c.stream(), lambda _: (-1, -2))
        self.assertEqual(len(requests), 32)
        self.assertEqual(len({r['mask_id'] for r in requests}), 32)
        self.assertTrue(any(r['reason']['kind'] == 'fallback_A' for r in requests))
        self.assertTrue(any(r['reason']['kind'] == 'audit' for r in requests))

    def test_unrevealed_futures_cannot_change_first_adaptive_proposal(self):
        p = fixture()['public']
        c1, c2 = Controller(p, (-1, -2)), Controller(copy.deepcopy(p), (-1, -2))
        # A controller's constructor has no argument for another arm's score cache.
        g1, g2 = c1.stream(), c2.stream()
        r1, r2 = next(g1), next(g2)
        for i in range(10):
            self.assertEqual(r1, r2)
            values = (-1+i*.03, -2-i*.04)
            r1, r2 = g1.send(values), g2.send(values)
        self.assertEqual(r1, r2)


class PilotGeometryTests(unittest.TestCase):
    @unittest.skipUnless(os.environ.get('DOCPRUNE_ACQUISITION_PACKAGE'), 'Set package path to test all 17 frozen cases')
    def test_real_17_geometry_and_controller(self):
        package = Path(os.environ['DOCPRUNE_ACQUISITION_PACKAGE'])
        result = validate_package(package)
        self.assertEqual(result['image_count'], 68)
        self.assertEqual(sum(r['actions'] for r in result['cases']), 1659)
        for info in result['cases']:
            case = read_record(package/f"cases/{info['case']}.json")
            p = case['public']
            groups = make_groups(p)
            self.assertTrue(all(len(g) == 1 or sum(p['costs'][i] for i in g) <= 501 for g in groups))
            sampler = ConditionalSampler([sum(p['costs'][i] for i in g) for g in groups], 5016)
            self.assertTrue(np.isfinite(sampler.logz[-1,-1]))
            for outcome in (lambda r: (-1., -2.), lambda r: (-1.+sum(r['mask'][::2])*.03, -2.+sum(r['mask'][1::2])*.05)):
                c = Controller(p, (-1, -2))
                requests = drive(c.stream(), outcome)
                self.assertEqual(len(requests), 32)
                self.assertEqual(len({r['mask_id'] for r in requests}), 32)
                self.assertTrue(all(np.dot(p['costs'], r['mask']) == 5016 for r in requests))
                for diagnostic in c.diagnostics:
                    a, b = [c.masks[key] for key in diagnostic['parents']]
                    w, x = [c.masks[key] for key in diagnostic['children']]
                    self.assertTrue(np.array_equal(a.astype(int)+b, w.astype(int)+x))
                    self.assertTrue(np.array_equal(w[a == b], a[a == b]))


if __name__ == '__main__':
    unittest.main()
