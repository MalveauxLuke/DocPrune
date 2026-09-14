"""Cache Col-patch overlap with actual reader-token ownership, without model calls."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(Path(path).read_text())


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ranks(values):
    _, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts + 1) / 2)[inverse]


def priorities(profiles, question_positions):
    x = np.asarray(profiles, dtype=float)[:, question_positions]
    lexical = x.mean(axis=1)
    sd = x.std(axis=0)
    sd[sd == 0] = 1
    peak = ((x - x.mean(axis=0)) / sd).max(axis=1)
    mix = (ranks(lexical) + ranks(peak)) / (2 * len(x))
    return {'question_only': lexical, 'query_peak': peak, 'automatic_mix': mix}


def owned_patch_fractions(patch_boxes, geometry, owner_indices, source_count):
    """Rows are Col patches; columns are unchanged original action IDs.

    Coordinates are normalized page footprints, not semantic receptive fields.
    geometry entries are [page_rank, row, column, reader_height, reader_width].
    """
    boxes = np.asarray(patch_boxes, dtype=np.float64)
    g = np.asarray(geometry, dtype=int)
    owners = np.asarray(owner_indices, dtype=int)
    if g.ndim != 2 or g.shape[1] != 5 or len(g) != len(owners):
        raise ValueError('Invalid reader geometry/ownership dimensions')
    if len(set(g[:, 0])) != 1 or len(set(map(tuple, g[:, 3:]))) != 1:
        raise ValueError('Expected a single page with one reader grid')
    h, w = g[0, 3:]
    if h <= 0 or w <= 0 or len(g) != h*w:
        raise ValueError('Reader grid is not complete')
    if np.any(g[:, 1] < 0) or np.any(g[:, 1] >= h) or np.any(g[:, 2] < 0) or np.any(g[:, 2] >= w):
        raise ValueError('Reader position outside its declared grid')
    if not np.array_equal(np.sort(g[:, 1]*w + g[:, 2]), np.arange(h*w)):
        raise ValueError('Duplicate or missing reader grid position')
    if np.any(owners < 0) or np.any(owners >= source_count):
        raise ValueError('Unknown owner')
    if not np.isfinite(boxes).all() or np.any(boxes < 0) or np.any(boxes > 1):
        raise ValueError('Invalid normalized Col patch footprint')
    area = (boxes[:, 2]-boxes[:, 0])*(boxes[:, 3]-boxes[:, 1])
    if np.any(area <= 0):
        raise ValueError('Empty Col patch')
    left, right = g[:, 2]/w, (g[:, 2]+1)/w
    top, bottom = g[:, 1]/h, (g[:, 1]+1)/h
    weights = np.zeros((len(boxes), source_count), dtype=np.float64)
    for start in range(0, len(boxes), 128):
        b = boxes[start:start+128]
        dx = np.maximum(0, np.minimum(b[:, 2, None], right) - np.maximum(b[:, 0, None], left))
        dy = np.maximum(0, np.minimum(b[:, 3, None], bottom) - np.maximum(b[:, 1, None], top))
        intersection = dx * dy
        for source in np.unique(owners):
            weights[start:start+len(b), source] = intersection[:, owners == source].sum(axis=1) / area[start:start+len(b)]
    np.testing.assert_allclose(weights.sum(axis=1), 1, atol=1e-12)
    expected_area = np.bincount(owners, minlength=source_count)/(h*w)
    np.testing.assert_allclose(area @ weights, expected_area, atol=2e-8)
    return weights


def main(root, audit, out):
    out.mkdir(parents=True, exist_ok=False)
    cases = read(audit / 'cases.json')
    records, coverage_rows, summary = [], [], []
    for case in cases:
        q = case['case']
        source = root / 'input/sources' / q
        feat = root / 'result/features' / q
        mapping, geom = read(source/'mapping.json'), read(source/'geometry.json')['geometry']
        comparison = read(source/'input/comparison-rp105.json')
        ids = comparison['design']['source_ids']
        index = {sid: i for i, sid in enumerate(ids)}
        owners = np.array([index[sid] for sid in mapping['token_to_source']])
        g = np.array(geom, dtype=int)
        assert len(owners) == len(g) == 10032
        profiles = np.zeros((len(ids), len(case['query_tokens'])))
        original_profiles = np.zeros_like(profiles)
        mass, support_count = np.zeros(len(ids)), np.zeros(len(ids), dtype=int)
        provenance = {}
        qout = out/q
        qout.mkdir()
        for page in range(4):
            npz_path = feat / f'page-{page:02d}.npz'
            meta = read(feat/f'page-{page:02d}.json')
            with np.load(npz_path, allow_pickle=False) as data:
                subset = g[:, 0] == page
                weights = owned_patch_fractions(data['patch_boxes_normalized'], g[subset], owners[subset], len(ids))
                sim = data['query_token_by_page_token'][:, data['image_positions']]
                for j, region in enumerate(meta['regions']):
                    i = index[region['source_id']]
                    support = weights[:, i] > 1e-12
                    if not support.any():
                        raise ValueError(f'No positive-area patch for existing action: {q}/{i}')
                    # Minimal mapping-only candidate: keep ordinary MaxSim,
                    # but restrict it to patches touching the owned footprint.
                    profiles[i] = sim[:, support].max(axis=1)
                    original_profiles[i] = data['region_query_maxsim'][j]
                    support_count[i] = int(support.sum())
                    mass[i] = weights[:, i].sum()
                np.savez_compressed(qout/f'page-{page:02d}.npz', owned_patch_fractions=weights,
                                    patch_boxes_normalized=data['patch_boxes_normalized'])
            provenance[f'page-{page:02d}.npz'] = sha(npz_path)
        lexical = case['question_token_indices']
        old, new = priorities(original_profiles, lexical), priorities(profiles, lexical)
        np.savez_compressed(qout/'profiles.npz', original_box_profiles=original_profiles,
                            owned_support_profiles=profiles, **new)
        region_rows = sorted([r for p in case['pages'] for r in p['regions']], key=lambda r:r['region_index'])
        assert [r['source_id'] for r in region_rows] == ids
        kind = np.array([r['kind'] for r in region_rows])
        for i, r in enumerate(region_rows):
            records.append({'case':q, 'region_index':i, 'source_id':ids[i], 'kind':r['kind'],
                            'page_rank':r['page_rank'], 'token_cost':r['token_cost'],
                            'owned_support_patches':int(support_count[i]), 'owned_patch_mass':float(mass[i]),
                            **{f'box_{key}':float(value[i]) for key,value in old.items()},
                            **{f'owned_{key}':float(value[i]) for key,value in new.items()}})
        for scope, subset in [('all', np.ones(len(ids),bool)), ('semantic',kind=='mineru-region'), ('residual',kind!='mineru-region')]:
            pool = np.flatnonzero(subset)
            k = math.ceil(.2*len(pool))
            for method, pp in [('box',old), ('owned_support',new)]:
                for key, values in pp.items():
                    selected = pool[np.argsort(-values[pool], kind='stable')[:k]]
                    for channel, sign in [('G',-1),('G',1),('S',1),('C',-1)]:
                        importance = np.maximum(sign*np.array([r['input_'+channel] for r in region_rows]),0)
                        total = importance[pool].sum()
                        coverage_rows.append({'case':q,'scope':scope,'mapping':method,'priority':key,
                            'target':('positive_' if sign>0 else 'negative_')+channel,
                            'coverage':float(importance[selected].sum()/total) if total>0 else None,
                            'selected_regions':len(selected),
                            'selected_tokens':sum(region_rows[i]['token_cost'] for i in selected)})
        metadata = {'case':q,'source_ids':ids,'source_mapping_sha256':sha(source/'mapping.json'),
                    'geometry_file_sha256':sha(source/'geometry.json'),'feature_files':provenance,
                    'query_file_sha256':sha(feat/'query.json'),
                    'contract':'Positive-area intersection of normalized Col patches with the union of unchanged owned reader-token cells; no semantic unmixing.',
                    'profile_aggregation':'Unweighted MaxSim over positive-area owned support; overlap fractions retained separately.',
                    'empty_actions':int((support_count==0).sum())}
        (qout/'manifest.json').write_text(json.dumps(metadata,indent=2)+'\n')
        summary.append({'case':q,'regions':len(ids),'empty_actions':metadata['empty_actions']})
        print(q, 'owned-overlap checked', len(ids), flush=True)
    for name, rows in [('regions',records),('coverage',coverage_rows)]:
        with (out/f'{name}.csv').open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    (out/'summary.json').write_text(json.dumps({'cases':summary,'script_sha256':sha(__file__)},indent=2)+'\n')


if __name__ == '__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--audit',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    main(args.root,args.audit,args.output)
