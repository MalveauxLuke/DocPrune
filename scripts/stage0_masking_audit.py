"""Retrospective Stage 0 audit. Reads immutable exported arrays; no model calls."""
import argparse
import csv
import hashlib
import html
import json
import math
import platform
import re
import textwrap
from pathlib import Path

import numpy as np
import PIL
from PIL import Image, ImageDraw, ImageFont


FOCUS = {
    'Q01': ['shaka', 'gesture', '2011'], 'Q02': ['distant', 'mountains'],
    'Q03': ['2007', 'award', 'nomination', '22'], 'Q04': ['screenplay'],
    'Q05': ['digest', 'rookie'], 'Q06': ['recent', 'year', 'female', 'category'],
    'Q07': ['species', 'snakes'], 'Q08': ['lower', 'points', 'norra', '1989'],
    'Q09': ['positions', 'division', '2000', '01'], 'Q10': ['singles', 'album'],
    'Q11': ['championship', 'champion', 'alfa', 'alfetta'], 'Q12': ['chasing'],
    'Q13': ['theme', 'paramore'], 'Q14': ['lower', 'area', 'km'],
    'Q15': ['highball', 'both'], 'Q16': ['worse', 'finishing', 'position'],
    'Q17': ['example', 'free'],
}


def read(path):
    return json.loads(Path(path).read_text())


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def csvout(path, rows):
    with Path(path).open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rank(x):
    _, inv, counts = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(counts) - (counts + 1) / 2)[inv]


def rho(x, y):
    x, y = np.asarray(x), np.asarray(y)
    keep = np.isfinite(x) & np.isfinite(y)
    if keep.sum() < 3 or np.ptp(x[keep]) == 0 or np.ptp(y[keep]) == 0:
        return None
    return float(np.corrcoef(rank(x[keep]), rank(y[keep]))[0, 1])


def text_content(value):
    if isinstance(value, dict):
        return ' '.join(str(v) if k in ('content', 'html') and isinstance(v, str)
                        else text_content(v) for k, v in value.items()
                        if k not in ('bbox', 'index', 'angle'))
    if isinstance(value, list):
        return ' '.join(text_content(v) for v in value)
    return ''


def block_texts(path):
    d = read(path)
    blocks = d['pdf_info'][0].get('preproc_blocks', [])
    return [(np.array(b['bbox']), re.sub(r'\s+', ' ', html.unescape(
        re.sub('<[^>]+>', ' ', text_content(b)))).strip()) for b in blocks]


def matched_text(region, blocks):
    a = np.array(region['bbox'])
    out = []
    for b, text in blocks:
        overlap = np.maximum(0, np.minimum(a[2:], b[2:]) - np.maximum(a[:2], b[:2])).prod()
        union = np.maximum(a[2:] - a[:2], 0).prod() + np.maximum(b[2:] - b[:2], 0).prod() - overlap
        if union and overlap / union > .85:
            out.append(text)
    return ' '.join(out)


def budget_order(priority, costs, budget):
    selected, used = [], 0
    for i in np.argsort(-np.asarray(priority), kind='stable'):
        if used + costs[i] <= budget:
            selected.append(int(i))
            used += int(costs[i])
    return selected, used


def coverage(mass, selected):
    return float(mass[selected].sum() / mass.sum()) if mass.sum() > 1e-12 else None


def ridge_cv(x, y):
    """Fixed-alpha diagnostic, not a refit of original Lasso; five disjoint folds."""
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    pred = np.zeros_like(y)
    fits = []
    for k in range(5):
        test = np.arange(len(x)) % 5 == k
        train = ~test
        mean, sd = x[train].mean(0), x[train].std(0)
        sd[sd == 0] = 1
        z = (x[train] - mean) / sd
        ym = y[train].mean(0)
        coef = np.linalg.solve(z.T @ z + np.eye(x.shape[1]), z.T @ (y[train] - ym))
        pred[test] = ((x[test] - mean) / sd) @ coef + ym
        fits.append(coef / sd[:, None])
    return pred, np.stack(fits)


def main(root, out):
    out.mkdir(parents=True, exist_ok=False)
    (out / 'contact-sheets').mkdir()
    regions_out, pages_out, query_out, coverage_out, teacher_out, cv_out, mask_out = [], [], [], [], [], [], []
    cases_out = []
    font = ImageFont.load_default(size=22)
    for n in range(1, 18):
        label = f'Q{n:02d}'
        source = root / 'input/sources' / label
        feat = root / 'result/features' / label
        case, mapping = read(source / 'case.json'), read(source / 'mapping.json')
        comp = read(source / 'input/comparison-rp105.json')
        ids = comp['design']['source_ids']
        idx = {s: i for i, s in enumerate(ids)}
        reg = {r['source_id']: r for r in mapping['sources']}
        costs = np.array([reg[s]['token_cost'] for s in ids])
        assert costs.sum() == len(mapping['token_to_source']) == 10032
        tokens = read(feat / 'query.json')['tokens']
        lex = np.array([i for i, t in enumerate(tokens) if i >= 2 and t != '<|endoftext|>'])
        aug = np.array([i for i, t in enumerate(tokens) if t == '<|endoftext|>'])
        # Term selection is an explicitly hand-authored question-only diagnostic.
        # It is not re-encoding a new query and is not causal token ablation.
        decoded = [t.replace('Ġ', ' ').replace('Ċ', '\n').lower() for t in tokens]
        focus = np.array([i for i in lex if any(w in decoded[i] for w in FOCUS[label])], dtype=int)
        if not len(focus):
            raise ValueError(f'No focus token match: {label}')
        for i, t in enumerate(tokens):
            query_out.append({'case': label, 'index': i, 'token': t,
                              'lexical': i in lex, 'augmentation': i in aug, 'focus': i in focus})
        profiles = np.zeros((len(ids), len(tokens)))
        priorities = {k: np.zeros(len(ids)) for k in ['all_maxsim', 'lex_maxsim', 'augmentation',
            'focus', 'top3_patch', 'mean_patch', 'per_token_cost', 'unique_coverage', 'exclusive_centroid',
            'query_peak', 'automatic_mix']}
        page_data = []
        sheet = Image.new('RGB', (1224, 1785), '#ffffff')
        draw = ImageDraw.Draw(sheet)
        draw.multiline_text((15, 10), textwrap.fill(label + '  ' + case['question'], 100),
                            font=font, fill='#182b40', spacing=6)
        for p in range(4):
            meta = read(feat / f'page-{p:02d}.json')
            image_path = root / 'input' / meta['page']['image']
            im = Image.open(image_path).convert('RGB')
            preview = im.resize((612, 792))
            # Readable 2x2 sheets keep the original page images linked separately.
            x, y = (p % 2) * 612, 170 + (p // 2) * 807
            sheet.paste(preview, (x, y))
            draw.text((x + 10, y - 27), f'Input rank {p}; source page {meta["page"]["page_index"]}', font=font, fill='#182b40')
            data = np.load(feat / f'page-{p:02d}.npz', allow_pickle=False)
            sim = data['query_token_by_page_token'][:, data['image_positions']]
            overlap = data['region_patch_overlap']
            blocks = block_texts(source / f'mineru/raw/page-{p:02d}_middle.json')
            local_rows = []
            # Alternative geometry diagnostic: original exclusive reader-token owner
            # sampled at each Col patch center, rather than full residual-grid boxes.
            geom = read(source / 'geometry.json')['geometry']
            owners = np.array(mapping['token_to_source'], dtype=object)
            grid = np.array([g for g in geom if g[0] == p])
            gh, gw = int(grid[0][-2]), int(grid[0][-1])
            page_owners = np.empty((gh, gw), dtype=object)
            for t, g in enumerate(geom):
                if g[0] == p:
                    page_owners[int(g[1]), int(g[2])] = owners[t]
            h, w = meta['grid_after_merge']
            patch_owners = np.array([page_owners[min(gh - 1, int((yy + .5) * gh / h)),
                min(gw - 1, int((xx + .5) * gw / w))] for yy in range(h) for xx in range(w)])
            for j, r in enumerate(meta['regions']):
                k = idx[r['source_id']]
                profiles[k] = data['region_query_maxsim'][j]
                member = overlap[j] > 0
                own = patch_owners == r['source_id']
                ps = sim[:, member]
                priorities['all_maxsim'][k] = profiles[k].sum()
                priorities['lex_maxsim'][k] = profiles[k, lex].mean()
                priorities['augmentation'][k] = profiles[k, aug].mean()
                priorities['focus'][k] = profiles[k, focus].mean()
                priorities['top3_patch'][k] = np.sort(ps[lex], axis=1)[:, -min(3, ps.shape[1]):].mean()
                priorities['mean_patch'][k] = ps[lex].mean()
                priorities['per_token_cost'][k] = profiles[k, lex].mean() / costs[k]
                priorities['exclusive_centroid'][k] = sim[np.ix_(lex, own)].max(axis=1).mean() if own.any() else -1
                owned_bbox_fraction = float(own[member].mean())
                rr = {'case': label, 'region_index': k, 'page_rank': p, 'source_id': r['source_id'],
                      'kind': r['source_kind'], 'type': r['region_type'], 'token_cost': int(costs[k]),
                      'bbox': r['bbox'], 'page_size': r['page_size'], 'patches': int(member.sum()),
                      'centroid_owned_patches': int(own.sum()), 'bbox_owner_fraction': owned_bbox_fraction,
                      'text': matched_text(r, blocks) if r['source_kind'] == 'mineru-region' else '',
                      'query_maxsim': profiles[k].tolist(),
                      **{key: float(v[k]) for key, v in priorities.items()}}
                local_rows.append(rr)
                regions_out.append(rr)
            pr = {'case': label, 'page_rank': p, 'doc_id': meta['page']['doc_id'],
                'source_page': meta['page']['page_index'], 'old_colpali': case['pages'][p]['score'],
                'full_maxsim': meta['full_page_maxsim'], 'image_maxsim': meta['image_only_maxsim'],
                'lex_image': float(sim[lex].max(axis=1).mean()),
                'focus_image': float(sim[focus].max(axis=1).mean()),
                'augmentation_image': float(sim[aug].max(axis=1).mean()), 'regions': len(local_rows)}
            pages_out.append(pr)
            page_data.append({'meta': meta['page'], 'image': str(image_path.resolve()), 'regions': local_rows, 'scores': pr})
        # Distinct region deletion-loss from the fixed retrieval feature bank.
        best = profiles[:, lex].max(axis=0)
        # Original question only: nominate a region unusually high for ANY query
        # token without hand-picking words. This is salience, not signed utility.
        question_profiles = profiles[:, lex]
        sd = question_profiles.std(axis=0)
        sd[sd == 0] = 1
        priorities['query_peak'] = ((question_profiles - question_profiles.mean(axis=0)) / sd).max(axis=1)
        priorities['automatic_mix'] = (rank(priorities['lex_maxsim']) + rank(priorities['query_peak'])) / (2 * len(ids))
        for k in range(len(ids)):
            priorities['unique_coverage'][k] = (best - np.delete(profiles[:, lex], k, axis=0).max(axis=0)).sum()
        for rr in regions_out[-len(ids):]:
            for key in ['unique_coverage', 'query_peak', 'automatic_mix']:
                rr[key] = float(priorities[key][rr['region_index']])
        sheet.save(out / 'contact-sheets' / f'{label}.jpg', quality=94)
        budget = int(comp['selections']['budgets'][0]['achieved_token_count'])
        case_out = {'case': label, 'question': case['question'], 'gold': case['complete_gold_sequences'],
                    'query_tokens': tokens, 'question_token_indices': lex.tolist(),
                    'focus_terms': FOCUS[label], 'focus_tokens': [tokens[i] for i in focus],
                    'pages': page_data, 'historical_budget': budget}
        for depth in ['input', 'intermediate']:
            d = read(source / depth / 'comparison-rp105.json')
            assert d['design']['source_ids'] == ids
            x = np.array([m['vector'] for m in d['design']['fit_masks']], dtype=float)
            gs = np.array(d['raw_likelihoods'])
            assert gs.shape == (256, 2) and x.shape == (256, len(ids))
            raw = read(source / depth / 'mask-scores.json')['raw']
            np.testing.assert_equal(gs, np.array(raw)[1:])
            full = np.array(d['full_context_likelihoods'])
            np.testing.assert_equal(full, np.array(raw)[0])
            y = np.column_stack((gs, gs[:, 0] - gs[:, 1]))
            base = np.r_[full, full[0] - full[1]]
            token_cost = x @ costs
            pred, folds = ridge_cv(x, y)
            coef = np.column_stack([np.array([d['surrogates'][kind]['fit']['coefficients'][sid]
                for sid in ids]) for kind in ['gold_support', 'fixed_self_support', 'gold_margin']])
            for t, name in enumerate(['G', 'S', 'C']):
                err = float(np.mean((pred[:, t] - y[:, t]) ** 2))
                r2 = float(1 - err / np.var(y[:, t])) if np.var(y[:, t]) else None
                stability = float(np.mean(np.mean(np.sign(folds[:, :, t]) == np.sign(coef[:, t]), axis=0)))
                cv_out.append({'case': label, 'depth': depth, 'actual_boundary': d['boundary'],
                    'target': name, 'ridge_cv_r2': r2, 'ridge_cv_spearman': rho(pred[:, t], y[:, t]),
                    'fold_sign_agreement_with_full_lasso': stability, 'cost_spearman': rho(token_cost, y[:, t])})
            for rr in regions_out[-len(ids):]:
                k = rr['region_index']
                rr.update({f'{depth}_{name}': float(coef[k, t]) for t, name in enumerate(['G', 'S', 'C'])})
                rr[f'{depth}_C_fold_sign_fraction'] = float(np.mean(np.sign(folds[:, k, 2]) == np.sign(coef[k, 2])))
            targets = {'abs_G': np.abs(coef[:, 0]), 'negative_G': np.maximum(-coef[:, 0], 0),
                'positive_G': np.maximum(coef[:, 0], 0), 'positive_S': np.maximum(coef[:, 1], 0),
                'negative_S': np.maximum(-coef[:, 1], 0), 'abs_C': np.abs(coef[:, 2]),
                'negative_C': np.maximum(-coef[:, 2], 0)}
            rng = np.random.default_rng(20260911 + n)
            random_orders = [rng.random(len(ids)) for _ in range(200)]
            for scope in ['regions20', 'tokens20']:
                selections = {}
                for key, priority in priorities.items():
                    selections[key] = np.argsort(-priority, kind='stable')[:math.ceil(.2 * len(ids))].tolist() if scope == 'regions20' else budget_order(priority, costs, int(.2 * costs.sum()))[0]
                random_sets = [np.argsort(-v)[:math.ceil(.2 * len(ids))].tolist() if scope == 'regions20'
                              else budget_order(v, costs, int(.2 * costs.sum()))[0] for v in random_orders]
                for target, mass in targets.items():
                    rc = [coverage(mass, s) for s in random_sets]
                    rb = float(np.mean(rc)) if rc[0] is not None else None
                    for key, s in selections.items():
                        coverage_out.append({'case': label, 'depth': depth, 'scope': scope, 'priority': key,
                            'target': target, 'coverage': coverage(mass, s), 'random_mean': rb,
                            'selected_token_fraction': float(costs[s].sum() / costs.sum()),
                            'selected_regions': len(s), 'signed_G_spearman': rho(priorities[key], coef[:, 0])})
            near = np.abs(token_cost - budget) <= .05 * budget
            for eps in [0, .1, .25, .5]:
                for poolname, pool in [('all_256_variable_cost', np.ones(256, dtype=bool)), ('within_5pct_historical_budget', near)]:
                    inds = np.flatnonzero(pool)
                    if not len(inds):
                        teacher_out.append({'case': label, 'depth': depth, 'pool': poolname, 'epsilon': eps,
                            'n': 0, 'admissible_fraction': None, 'contrast_gain_gold_drop_fraction': None,
                            'gold_best_seed': None, 'contrast_best_seed': None, 'aware_best_seed': None,
                            'aware_G_delta': None, 'contrast_best_G_delta': None})
                        continue
                    admissible = gs[inds, 0] >= full[0] - eps
                    gbest = int(inds[np.argmax(y[inds, 0])]); cbest = int(inds[np.argmax(y[inds, 2])])
                    abest = int(max(inds, key=lambda z: (y[z, 0] >= full[0] - eps,
                                 y[z, 2] - base[2] if y[z, 0] >= full[0] - eps else y[z, 0] - base[0])))
                    teacher_out.append({'case': label, 'depth': depth, 'pool': poolname, 'epsilon': eps,
                        'n': len(inds), 'admissible_fraction': float(admissible.mean()),
                        'contrast_gain_gold_drop_fraction': float(np.mean((y[inds, 2] > base[2]) & (y[inds, 0] < base[0]))),
                        'gold_best_seed': gbest, 'contrast_best_seed': cbest, 'aware_best_seed': abest,
                        'aware_G_delta': float(y[abest, 0] - base[0]), 'contrast_best_G_delta': float(y[cbest, 0] - base[0])})
            for j in range(256):
                # Fixed-feature set coverage is a matching diagnostic, never a reader prediction.
                retained = x[j].astype(bool)
                val = profiles[retained][:, lex].max(0).mean() if retained.any() else -1
                mask_out.append({'case': label, 'depth': depth, 'seed': j, 'cost': int(token_cost[j]),
                    'budget': budget, 'G': float(y[j, 0]), 'S': float(y[j, 1]), 'C': float(y[j, 2]),
                    'delta_G': float(y[j, 0] - base[0]), 'delta_S': float(y[j, 1] - base[1]),
                    'delta_C': float(y[j, 2] - base[2]), 'fixed_bank_col_coverage': float(val)})
            case_out[depth + '_boundary'] = d['boundary']
            case_out[depth + '_full_GS'] = full.tolist()
        cases_out.append(case_out)
        print(label, 'audited', len(ids), 'regions', flush=True)
    for name, rows in [('regions', regions_out), ('pages', pages_out), ('query-tokens', query_out),
                       ('priority-coverage', coverage_out), ('teacher-combinations', teacher_out),
                       ('surrogate-cv', cv_out), ('mask-comparison', mask_out)]:
        csvout(out / f'{name}.csv', rows)
    dump(out / 'cases.json', cases_out)
    summary = {'cases': len(cases_out), 'pages': len(pages_out), 'regions': len(regions_out),
        'measured_nonanchor_masks': len(mask_out), 'geometry_empty_centroid_regions':
        sum(r['centroid_owned_patches'] == 0 for r in regions_out),
        'scope': 'retrospective diagnostic; no new query encoding, reader pass, or generated answer',
        'python': platform.python_version(), 'numpy': np.__version__, 'pillow': PIL.__version__,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'random_priority_seeds': '20260911 + one-based case number',
        'ridge': {'alpha': 1, 'fold_assignment': 'row_index % 5'},
        'input_root': str(root.resolve())}
    dump(out / 'summary.json', summary)
    print(summary)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    a = ap.parse_args()
    main(a.root, a.output)
