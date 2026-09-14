"""Count distinct gold-aware supervision in measured G/S rows; never infer new outcomes."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def preference_yield(gs,reference,epsilon,tie_margin=0.,pairs=None):
    gs=np.asarray(gs,dtype=float)
    reference=np.asarray(reference,dtype=float)
    if gs.ndim!=2 or gs.shape[1]!=2 or reference.shape!=(2,) or not np.isfinite(gs).all() or not np.isfinite(reference).all():
        raise ValueError('Expected finite, measured G/S pairs and a common all-keep reference')
    if epsilon<0 or tie_margin<0:
        raise ValueError('Negative tolerance')
    n=len(gs)
    g=gs[:,0]-reference[0]
    c=gs[:,0]-gs[:,1]-(reference[0]-reference[1])
    admissible=g>=-epsilon
    if pairs is None:
        i,j=np.triu_indices(n,1)
    else:
        pp=np.asarray(pairs,dtype=int)
        if pp.size==0:pp=np.empty((0,2),dtype=int)
        if pp.ndim!=2 or pp.shape[1]!=2 or np.any(pp<0) or np.any(pp>=n) or np.any(pp[:,0]==pp[:,1]):
            raise ValueError('Invalid comparison pairs')
        pp=np.unique(np.sort(pp,axis=1),axis=0)
        i,j=pp.T
    dg,dc=g[i]-g[j],c[i]-c[j]
    both=admissible[i]&admissible[j]
    strict=(np.abs(dg)>tie_margin)&(np.abs(dc)>tie_margin)
    disagreement=both&strict&(dg*dc<0)
    a=int(admissible.sum())
    upper=a*(a-1)//2
    best_g=int(np.argmax(g)) if n else None
    best_c=int(np.argmax(c)) if n else None
    best_a=max(range(n),key=lambda k:(bool(admissible[k]),c[k] if admissible[k] else g[k])) if n else None
    return {'masks':n,'admissible_masks':a,'has_two_admissible':a>=2,
        'all_pairs_S_disagreement_upper_bound':upper,'comparisons':len(i),
        'both_admissible_comparisons':int(both.sum()),
        'non_tied_both_admissible_comparisons':int((both&strict).sum()),
        'strict_G_goldaware_disagreements':int(disagreement.sum()),
        'strict_disagreement_fraction':float(disagreement.mean()) if len(i) else None,
        'G_ties_with_C_resolution':int((both&(np.abs(dg)<=tie_margin)&(np.abs(dc)>tie_margin)).sum()),
        'best_G_index':best_g,'best_C_index':best_c,'best_goldaware_index':best_a,
        'goldaware_best_admissible':bool(admissible[best_a]) if n else None,
        'goldaware_best_g':float(g[best_a]) if n else None,
        'goldaware_best_c':float(c[best_a]) if n else None}


def main(root,audit,out):
    out.mkdir(parents=True,exist_ok=False)
    cases=json.loads((audit/'cases.json').read_text())
    records=[]
    for case in cases:
        q=case['case']
        regions=sorted([r for p in case['pages'] for r in p['regions']],key=lambda r:r['region_index'])
        costs=np.array([r['token_cost'] for r in regions])
        for depth in ['input','intermediate']:
            data=json.loads((root/'input/sources'/q/depth/'comparison-rp105.json').read_text())
            gs=np.array(data['raw_likelihoods'])
            masks=np.array([m['vector'] for m in data['design']['fit_masks']])
            b=case['historical_budget']
            for pool,selected in [('all_256',np.ones(len(masks),bool)),('within_5pct_historical_budget',np.abs(masks@costs-b)<=.05*b)]:
                for eps in [0.,.1,.25,.5]:
                    for margin in [0.,.01,.05]:
                        records.append({'case':q,'depth':depth,'pool':pool,'epsilon':eps,'tie_margin':margin,
                            **preference_yield(gs[selected],data['full_context_likelihoods'],eps,margin)})
    with (out/'legacy-yield.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(records[0]));w.writeheader();w.writerows(records)
    (out/'scope.json').write_text(json.dumps({'scope':'Retrospective existing measured masks only; no prospective G/S rows exist.',
        'note':'Best indices refer to the selected pool order. No actual training comparison set was used by this historical audit.'},indent=2)+'\n')
    for pool in ['all_256','within_5pct_historical_budget']:
        rr=[r for r in records if r['depth']=='input' and r['epsilon']==.1 and r['tie_margin']==.01 and r['pool']==pool]
        print(pool,'questions with >=2 admissible',sum(r['has_two_admissible'] for r in rr),
              'strict disagreements',sum(r['strict_G_goldaware_disagreements'] for r in rr),
              'of',sum(r['comparisons'] for r in rr),'pairs')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True);ap.add_argument('--audit',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();main(args.root,args.audit,args.output)
