"""Per-arm revealed histories, shared physical cache, deterministic crash replay."""
from __future__ import annotations
from contextlib import contextmanager
from dataclasses import asdict
import fcntl
from pathlib import Path
import time
import numpy as np
from .adaptive_acquisition import CONFIG, Controller, digest, scores, static_stream, utility, preference
from .acquisition_io import read_record, sha, token_identity, tree_identity, write_new, validate_case


@contextmanager
def run_lock(output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    with (output/'.writer.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError('Another runner owns this output directory') from exc
        yield


def summarize(history, reference):
    result = {}
    for prefix in (8, 16, 32):
        rows = history[:prefix]
        if len(rows) != prefix:
            continue
        def winner(kind):
            def rank(row):
                g, s = row['G'], row['S']
                return {'G': (g, g-s), 'C': (g-s, g), 'gold_aware': utility((g, s), reference[0])}[kind]
            row = min(rows, key=lambda row: (tuple(-x for x in rank(row)), row['request']['mask_id']))
            return {'mask_id': row['request']['mask_id'], 'G': row['G'], 'S': row['S'], 'C': row['G']-row['S'],
                    'delta_G': row['G']-reference[0], 'delta_S': row['S']-reference[1],
                    'admissible': row['G'] >= reference[0]-CONFIG.epsilon}
        reversals = 0
        strict = 0
        for i, a in enumerate(rows):
            for b in rows[i+1:]:
                dg = b['G']-a['G']
                dc = (b['G']-b['S'])-(a['G']-a['S'])
                admitted = min(a['G'], b['G']) >= reference[0]-CONFIG.epsilon
                if admitted and abs(dg) > CONFIG.margin and abs(dc) > CONFIG.margin:
                    strict += 1
                    reversals += dg*dc < 0
        result[str(prefix)] = {'winners': {k: winner(k) for k in ('G', 'C', 'gold_aware')},
                 'admissible_count': sum(row['G'] >= reference[0]-CONFIG.epsilon for row in rows),
                 'strict_G_C_disagreements': int(reversals), 'strict_admissible_pairs': strict,
                 'all_pair_count': prefix*(prefix-1)//2,
                 'fallbacks': sum(row['request']['reason']['kind'] == 'fallback_A' for row in rows)}
    return result


def run_case(case, output, backend_factory, backend_identity, *, max_new_observations=None):
    """Caller holds run_lock. Factory builds one persistent question checkpoint lazily.

    max_new_observations provides a controlled interruption for tests/CPU previews.
    A score cached by another arm is invisible until this arm asks for that mask.
    """
    validate_case(case)
    output = Path(output)
    public = case['public']
    name = public['case']
    reference = scores(case['reader']['reference_likelihoods'])
    identity = {'schema': 'stage0-acquisition-run-v1', 'case_sha256': digest(case),
                'controller': asdict(CONFIG), 'code_sha256': tree_identity(), 'backend': backend_identity}
    write_new(output/name/'identity.json', identity)
    controller = Controller(public, reference)
    streams = {'R': static_stream(public, 'R'), 'A': static_stream(public, 'A'), 'adaptive': controller.stream()}
    pending = {arm: next(stream) for arm, stream in streams.items()}
    histories = {arm: [] for arm in streams}
    backend, added, physical = None, 0, 0
    runtime_identity = digest(identity)
    arm_order = ['R', 'A', 'adaptive'] if int(name[1:]) % 2 else ['A', 'R', 'adaptive']
    try:
        for slot in range(32):
            for arm in arm_order:
                request = pending[arm]
                base = output/name/arm
                event_path = base/f'{slot:02}.result.json'
                decision = {'identity_sha256': runtime_identity, 'arm': arm, 'slot': slot, 'request': request}
                write_new(base/f'{slot:02}.request.json', decision)
                token_hash, retained_ids = token_identity(public, request['mask'])
                if len(retained_ids) != public['budget']:
                    raise ValueError('Controller requested incorrect physical token count')
                key = digest([runtime_identity, token_hash])
                cache_path = output/name/'physical'/f'{key}.json'
                cache_hit = cache_path.exists()
                if event_path.exists():
                    event = read_record(event_path)
                    if event['decision_sha256'] != digest(decision) or event['physical_key'] != key:
                        raise ValueError('Resume proposal/history drift')
                    cached = read_record(cache_path)
                    if event['physical_record_sha256'] != sha(cache_path):
                        raise ValueError('Physical cache changed after observation')
                else:
                    if max_new_observations is not None and added >= max_new_observations:
                        return {'case': name, 'complete': False, 'new_observations': added, 'new_physical_scores': physical}
                    if cache_hit:
                        cached = read_record(cache_path)
                    else:
                        if backend is None:
                            backend = backend_factory(case)
                        started = time.perf_counter()
                        measured = backend.score(retained_ids)
                        g, s = scores(measured['likelihoods'])
                        if len(measured['likelihoods']) != len(case['reader']['targets']):
                            raise ValueError('Reader returned wrong target count')
                        cached = {'identity_sha256': runtime_identity, 'token_mask_sha256': token_hash,
                                  'retained_token_count': len(retained_ids), 'measurement': measured,
                                  'G': g, 'S': s, 'wall_seconds': time.perf_counter()-started}
                        write_new(cache_path, cached)
                        physical += 1
                    event = {'decision_sha256': digest(decision), 'physical_key': key,
                             'physical_record_sha256': sha(cache_path), 'physical_cache_hit_at_reveal': cache_hit,
                             'G': cached['G'], 'S': cached['S']}
                    write_new(event_path, event)
                    added += 1
                if cached['identity_sha256'] != runtime_identity or cached['token_mask_sha256'] != token_hash:
                    raise ValueError('Physical cache identity mismatch')
                g, s = scores(cached['measurement']['likelihoods'])
                if (g, s) != (cached['G'], cached['S']) or (g, s) != (event['G'], event['S']):
                    raise ValueError('Corrupt score aggregation')
                histories[arm].append({'request': request, 'G': g, 'S': s})
                try:
                    pending[arm] = streams[arm].send((g, s))
                except StopIteration:
                    if slot != 31:
                        raise RuntimeError('Arm stopped early')
                    pending[arm] = None
        result = {'case': name, 'complete': True, 'identity_sha256': runtime_identity,
                  'synthetic': backend_identity.get('synthetic', False),
                  'arms': {arm: summarize(rows, reference) for arm, rows in histories.items()},
                  'adaptive_quartets': controller.diagnostics, 'adaptive_scale': controller.scale.tolist(),
                  'logical_observations': {arm: len(rows) for arm, rows in histories.items()},
                  'unique_physical_masks': len(list((output/name/'physical').glob('*.json'))),
                  'group_count': len(controller.groups), 'action_count': len(public['costs'])}
        write_new(output/name/'summary.json', result)
        return result
    finally:
        if backend is not None and hasattr(backend, 'close'):
            backend.close()


class SyntheticReader:
    """Deterministic test function, never a replay or estimate of pilot outcomes."""
    identity = {'synthetic': True, 'name': 'additive-plus-interaction-fixture-v1'}
    def __init__(self, case):
        from .adaptive_acquisition import rng_for
        self.case = case
        self.calls = 0
        n = len(case['public']['costs'])
        rng = rng_for(case['public']['case'], 'synthetic-reader')
        self.weights = rng.normal(0, .2, (2, n))
        self.reference = np.asarray(scores(case['reader']['reference_likelihoods']))

    def score(self, ids):
        self.calls += 1
        owners = np.asarray(self.case['public']['owners'])
        keep = np.zeros(len(self.weights[0]), dtype=float)
        keep[np.unique(owners[ids])] = 1
        y = self.reference + self.weights @ (keep-1)
        y += np.array([1., -.7]) * (keep[0]-keep[1]) * (keep[2]-.5)
        # Construct a max-reference contract with exactly the frozen target count.
        references = len(self.case['reader']['targets'])-1
        return {'likelihoods': [float(y[0]-i*.1) for i in range(references)]+[float(y[1])], 'synthetic': True}
