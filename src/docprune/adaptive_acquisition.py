"""Frozen v1 acquisition policy. This module receives no historical mask scores."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from itertools import combinations
import numpy as np


@dataclass(frozen=True)
class Config:
    version: str = 'adaptive-acquisition-v1'
    seed: int = 20260913
    observations: int = 32
    initial: int = 8
    rounds: int = 3
    epsilon: float = .1
    margin: float = .01
    scale_floor: float = .01
    individual_fraction: float = .2
    group_fraction: float = .05
    tiles: int = 3
    attempts: int = 128


CONFIG = Config()


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def rng_for(*parts):
    return np.random.default_rng(int(digest([CONFIG.seed, *parts])[:16], 16))


def mask_id(mask):
    return digest([int(v) for v in mask])


def scores(values):
    a = np.asarray(values, dtype=float)
    if a.ndim != 1 or len(a) < 2 or not np.isfinite(a).all():
        raise ValueError('Expected finite reference likelihoods followed by fixed-self likelihood')
    return float(max(a[:-1])), float(a[-1])


def utility(y, reference_g):
    g, s = y
    return (1, g - s, g, -s) if g >= reference_g - CONFIG.epsilon else (0, g, g - s, -s)


def preference(before, after, reference_g):
    """Margin-aware preference for after over before, not statistical confidence."""
    threshold = reference_g - CONFIG.epsilon
    a, b = before[0] >= threshold, after[0] >= threshold
    if a != b:
        if min(abs(before[0] - threshold), abs(after[0] - threshold)) < CONFIG.margin:
            return 0
        return 1 if b else -1
    delta = ((after[0] - after[1]) - (before[0] - before[1])) if a else after[0] - before[0]
    return int(delta > CONFIG.margin) - int(delta < -CONFIG.margin)


class ConditionalSampler:
    """Uniform exact-cost subsets; log-DP from the existing Stage 0 bank sampler."""
    def __init__(self, costs, budget):
        self.costs = np.asarray(costs, dtype=int)
        self.budget = int(budget)
        if self.costs.ndim != 1 or np.any(self.costs <= 0) or self.budget < 0:
            raise ValueError('Invalid subset costs or budget')
        self.logz = np.full((len(self.costs) + 1, self.budget + 1), -np.inf)
        self.logz[0, 0] = 0
        for i, c in enumerate(self.costs):
            self.logz[i + 1] = self.logz[i]
            if c <= self.budget:
                self.logz[i + 1, c:] = np.logaddexp(self.logz[i, c:], self.logz[i, :self.budget-c+1])
        if not np.isfinite(self.logz[-1, self.budget]):
            raise ValueError('Infeasible exact cost')

    def draw(self, rng):
        result = np.zeros(len(self.costs), dtype=bool)
        remaining = self.budget
        for i in range(len(self.costs)-1, -1, -1):
            c = int(self.costs[i])
            logp = self.logz[i, remaining-c] - self.logz[i+1, remaining] if remaining >= c else -np.inf
            if rng.random() < min(1., float(np.exp(logp))):
                result[i] = True
                remaining -= c
        if remaining:
            raise RuntimeError('Exact-cost sampler failure')
        return result


def reachable(costs):
    bits = 1
    for c in costs:
        bits |= bits << int(c)
    return bits


def make_groups(public):
    costs = np.asarray(public['costs'], dtype=int)
    cap = math.floor(CONFIG.group_fraction * sum(costs))
    n = len(costs)
    priority = public['priority']
    individual = set(sorted(range(n), key=lambda i: (-priority[i], i))[:math.ceil(CONFIG.individual_fraction*n)])
    individual.update(i for i, c in enumerate(costs) if c > cap)
    groups = [[i] for i in sorted(individual)]
    buckets = {}
    for i, region in enumerate(public['regions']):
        if i in individual:
            continue
        page, row, col, h, w = region['centroid']
        key = (page, int(CONFIG.tiles*row/h), int(CONFIG.tiles*col/w), region['kind'])
        buckets.setdefault(key, []).append(i)
    for key in sorted(buckets):
        ordered = sorted(buckets[key], key=lambda i: (*public['regions'][i]['centroid'][1:3], i))
        current, used = [], 0
        for i in ordered:
            if current and used + costs[i] > cap:
                groups.append(current)
                current, used = [], 0
            current.append(i)
            used += costs[i]
        if current:
            groups.append(current)
    return sorted(groups, key=lambda g: min(g))


def split_plan(u, v, costs, groups, fine=False):
    """Find a proper balanced partial exchange, opening temporary groups as needed."""
    sides = [[i for i in range(len(u)) if bool(u[i]) == retained and bool(v[i]) != retained]
             for retained in (True, False)]
    total = int(sum(costs[i] for i in sides[0]))
    if total != sum(costs[i] for i in sides[1]):
        raise ValueError('Parent costs differ')
    units = []
    for side in sides:
        allowed = set(side)
        units.append([[i] for i in side] if fine else
                     [part for group in groups if (part := [i for i in group if i in allowed])])
    opened = []
    while total:
        unit_costs = [[int(sum(costs[i] for i in g)) for g in side] for side in units]
        common = reachable(unit_costs[0]) & reachable(unit_costs[1])
        proper = [t for t in range(1, total) if (common >> t) & 1]
        if proper:
            t = min(proper, key=lambda t: (abs(2*t-total), t))
            return {'units': units, 'cost': t, 'total': total, 'opened': opened}
        candidates = [(s, j, g) for s, side in enumerate(units) for j, g in enumerate(side) if len(g) > 1]
        if not candidates:
            return None
        s, j, group = min(candidates, key=lambda x: (-sum(costs[i] for i in x[2]), -len(x[2]), min(x[2]), x[0]))
        opened.append({'side': s, 'members': group})
        units[s][j:j+1] = [[i] for i in group]
    return None


def draw_quartet(u, v, costs, plan, rng):
    chosen = []
    for side in plan['units']:
        sampled = ConditionalSampler([sum(costs[i] for i in g) for g in side], plan['cost']).draw(rng)
        chosen.append([i for g, keep in zip(side, sampled) if keep for i in g])
    w, x = np.asarray(u, dtype=bool).copy(), np.asarray(v, dtype=bool).copy()
    w[chosen[0]], w[chosen[1]] = False, True
    x[chosen[0]], x[chosen[1]] = True, False
    return w, x, chosen


class Controller:
    """Generator asks for one mask at a time; send only that mask's (G,S)."""
    def __init__(self, public, reference):
        self.public = public
        self.costs = np.asarray(public['costs'], dtype=int)
        self.budget = public['budget']
        self.reference = tuple(reference)
        self.groups = make_groups(public)
        self.observed, self.masks, self.edges = {}, {}, {}
        self.audit_counts = np.zeros(len(self.costs), dtype=int)
        self.diagnostics = []
        self.scale = None
        self._plans = {}

    def _edge(self, a, b, depth=0, discount=1.):
        if a == b:
            return
        key = tuple(sorted((a, b)))
        # First discovery owns lineage; rediscovery cannot reset an expanded edge.
        if key not in self.edges:
            self.edges[key] = {'a': a, 'b': b, 'depth': depth, 'discount': discount, 'used': []}

    def _ask(self, mask, reason, connect=False):
        mask = np.asarray(mask, dtype=bool)
        key = mask_id(mask)
        if key in self.observed or int(self.costs @ mask) != self.budget:
            raise ValueError('Duplicate or off-budget logical observation')
        request = {'mask_id': key, 'mask': mask.astype(int).tolist(), 'reason': reason,
                   'history_sha256': digest(list(self.observed.items()))}
        value = yield request
        if len(value) != 2 or not np.isfinite(value).all():
            raise ValueError('Expected finite G/S')
        previous = list(self.observed)
        self.observed[key] = tuple(map(float, value))
        self.masks[key] = mask
        if connect:
            for other in previous:
                self._edge(other, key)
        return key

    def _fallback(self, reason):
        for mask in self.public['banks']['A']:
            if mask_id(mask) not in self.observed:
                return (yield from self._ask(mask, {'kind': 'fallback_A', **reason}, connect=True))
        raise RuntimeError('A bank exhausted before 32 distinct observations')

    def _broad(self, round_index, fine):
        units = [[i] for i in range(len(self.costs))] if fine else self.groups
        rng = rng_for(self.public['case'], 'broad', round_index, fine)
        try:
            sampler = ConditionalSampler([sum(self.costs[i] for i in group) for group in units], self.budget)
        except ValueError:
            yield from self._fallback({'reason': 'grouped_budget_infeasible', 'round': round_index})
            return
        for attempt in range(CONFIG.attempts):
            picked = sampler.draw(rng)
            mask = np.zeros(len(self.costs), dtype=bool)
            for group, keep in zip(units, picked):
                mask[group] = keep
            if mask_id(mask) not in self.observed:
                yield from self._ask(mask, {'kind': 'broad_original' if fine else 'broad_grouped',
                                          'round': round_index, 'attempt': attempt+1}, connect=True)
                return
        yield from self._fallback({'reason': 'broad_duplicate_cap', 'round': round_index})

    def _candidates(self, mode, round_index):
        result = []
        for key, e in self.edges.items():
            if mode in e['used']:
                continue
            a, b = e['a'], e['b']
            delta = np.asarray(self.observed[b]) - self.observed[a]
            quiet = bool(np.all(np.abs(delta) <= CONFIG.margin))
            if mode == 'focus' and quiet:
                continue
            cachekey = (key, mode)
            if cachekey not in self._plans:
                self._plans[cachekey] = split_plan(self.masks[a], self.masks[b], self.costs, self.groups, mode == 'audit')
            plan = self._plans[cachekey]
            if plan is None:
                continue
            different = np.flatnonzero(self.masks[a] != self.masks[b])
            d = len(different)
            response = float(max(abs(delta) / self.scale))
            priority = e['discount'] * response / (math.sqrt(d) * (1 + e['depth']))
            better = max(utility(self.observed[x], self.reference[0]) for x in (a, b))
            coverage = float(sum(1/(1+self.audit_counts[i]) for i in different)) / d
            result.append((key, e, plan, response, priority, better, d, coverage))
        if mode == 'focus':
            admitted = [item for item in result if item[5][0]]
            if admitted:
                result = admitted
                result.sort(key=lambda t: (-t[4], tuple(-x for x in t[5]), t[6], t[0]))
            else:
                result.sort(key=lambda t: (-max(self.observed[t[1][x]][0] for x in ('a', 'b')),
                                           -t[4], tuple(-x for x in t[5]), t[6], t[0]))
        else:
            result.sort(key=lambda t: ((-t[7], t[3]/math.sqrt(t[6])) if round_index == 1 else
                                       (t[3]/math.sqrt(t[6]), -t[7]), t[6], t[0]))
        return result

    def _followup(self, round_index, phase, mode):
        candidates = self._candidates(mode, round_index)
        if not candidates:
            for _ in range(2):
                yield from self._fallback({'reason': 'no_eligible_pair', 'mode': mode, 'round': round_index, 'phase': phase})
            return
        key, e, plan, response, priority, _, _, _ = candidates[0]
        e['used'].append(mode)
        a, b = e['a'], e['b']
        rng = rng_for(self.public['case'], round_index, phase, mode, key)
        incumbent = max(self.observed.values(), key=lambda y: utility(y, self.reference[0]))
        for attempt in range(CONFIG.attempts):
            w, x, chosen = draw_quartet(self.masks[a], self.masks[b], self.costs, plan, rng)
            children = [mask_id(w), mask_id(x)]
            if any(child not in self.observed for child in children):
                break
        else:
            for _ in range(2):
                yield from self._fallback({'reason': 'quartet_duplicate_cap', 'parents': list(key), 'mode': mode})
            return
        before = len(self.observed)
        metadata = {'kind': mode, 'round': round_index, 'phase': phase, 'parents': [a, b],
                    'children': children, 'exchanged': chosen, 'cost': plan['cost'],
                    'opened_groups': plan['opened'], 'depth': e['depth'], 'response': response,
                    'priority': priority, 'attempt': attempt+1}
        for child, mask in zip(children, (w, x)):
            if child not in self.observed:
                yield from self._ask(mask, metadata)
        wy, xy = [self.observed[child] for child in children]
        ay, by = self.observed[a], self.observed[b]
        p1 = preference(ay, wy, self.reference[0])
        p2 = preference(xy, by, self.reference[0])
        productive = any(preference(incumbent, y, self.reference[0]) > 0 for y in (wy, xy))
        discount = .5 if p1 != 0 and p1 == p2 and not productive else 1.
        self.diagnostics.append({**metadata, 'preferences': [p1, p2], 'reversal': p1*p2 == -1,
                                 'productive': productive, 'descendant_discount': discount,
                                 'delta_first': (np.asarray(wy)-ay).tolist(),
                                 'delta_second': (np.asarray(by)-xy).tolist(),
                                 'interaction': (np.asarray(ay)+by-np.asarray(wy)-xy).tolist()})
        if mode == 'audit':
            self.audit_counts[self.masks[a] != self.masks[b]] += 1
        for parent, child in ((a, children[0]), (children[0], b), (a, children[1]), (children[1], b)):
            self._edge(parent, child, e['depth']+1, e['discount']*discount)
        for _ in range(2-(len(self.observed)-before)):
            yield from self._fallback({'reason': 'quartet_corner_already_revealed', 'mode': mode, 'parents': list(key)})

    def stream(self):
        for slot, mask in enumerate(self.public['banks']['A'][:CONFIG.initial]):
            yield from self._ask(mask, {'kind': 'seed_A', 'slot': slot}, connect=True)
        values = np.asarray(list(self.observed.values()))
        self.scale = np.maximum(CONFIG.scale_floor, np.percentile(values, 75, axis=0)-np.percentile(values, 25, axis=0))
        for round_index in range(CONFIG.rounds):
            yield from self._broad(round_index, False)
            yield from self._broad(round_index, True)
            yield from self._followup(round_index, 0, 'focus')
            yield from self._followup(round_index, 1, 'audit')
            yield from self._followup(round_index, 2, 'focus')
        if len(self.observed) != CONFIG.observations:
            raise RuntimeError('Controller did not finish 32 distinct observations')


def static_stream(public, arm):
    history = []
    for slot, mask in enumerate(public['banks'][arm]):
        key = mask_id(mask)
        value = yield {'mask_id': key, 'mask': mask, 'reason': {'kind': 'prepared_'+arm, 'slot': slot},
                       'history_sha256': digest(history)}
        history.append((key, tuple(map(float, value))))
