"""Freeze a stratified, outcome-blind pool; no model calls or evidence filtering."""
import argparse, collections, hashlib, json, math
from pathlib import Path
from urllib.parse import unquote, urlsplit
from inventory import sha

SEED='docprune-m3doc-singlehop-600-v1-20260915'
def rank(qid):return hashlib.sha256((SEED+'\0'+qid).encode()).hexdigest()
def family(url):
    u=urlsplit(url)
    return (u.netloc.lower()+unquote(u.path).replace('_',' ')).casefold()
def quotas(counts,n):
    total=sum(counts.values())
    if total<n:raise ValueError(f'Only {total} eligible questions for target {n}')
    raw={k:n*v/total for k,v in counts.items()};out={k:math.floor(v) for k,v in raw.items()}
    for k in sorted(raw,key=lambda k:(-(raw[k]-out[k]),k))[:n-sum(out.values())]:out[k]+=1
    return out

def run(inventory,out):
    d=json.loads(Path(inventory).read_text());urls={r['id']:r['url'] for r in d['id_url_mapping']}
    byid={q['qid']:q for q in d['questions']}
    def families(q):
        return {family(urls[s['doc_id']]) for s in q['supporting_context']}
    excluded=set(d['exposure']['older600'])
    eligible=[byid[k] for k in d['eligible_qids'] if k not in excluded]
    counts=collections.Counter(q['metadata']['type'] for q in eligible);allocation=quotas(counts,600)
    chosen=[];seen=set()
    for t in sorted(allocation):
        candidates=sorted((q for q in eligible if q['metadata']['type']==t),key=lambda q:rank(q['qid']))
        selected=[]
        for q in candidates:
            fs=families(q)
            if not fs&seen and len(selected)<allocation[t]:selected.append(q);seen.update(fs)
        ids={q['qid'] for q in selected}
        for q in candidates:
            if len(selected)==allocation[t]:break
            if q['qid'] not in ids:selected.append(q);ids.add(q['qid']);seen.update(families(q))
        chosen.extend(selected)
    assert len(chosen)==600 and len({q['qid'] for q in chosen})==600
    prior=set().union(*(set(v) for v in d['exposure'].values()))
    prior_families=set().union(*(families(byid[k]) for k in prior))
    records=[]
    for q in sorted(chosen,key=lambda q:rank(q['qid'])):
        k=q['qid'];pages=d['retrieved_pages'][k];sf=families(q)
        records.append({'qid':k,'question':q['question'],'answers':q['answers'],'metadata':q['metadata'],
            'supporting_context':q['supporting_context'],'support_families':sorted(sf),
            'original_top4':pages,'original_top4_families':sorted({family(urls[p['doc_id']]) for p in pages}),
            'prior_selected_cohorts':[label for label,ids in d['exposure'].items() if k in ids],
            'historical_baseline_scored':k in d['historical_baseline_scored_qids'],
            'prior_support_family_overlap':sorted(sf&prior_families)})
    fc=collections.Counter(f for r in records for f in r['support_families'])
    bc=collections.Counter(f for r in records for f in r['original_top4_families'])
    result={'schema':'m3docvqa-singlehop-600-pool-v1','seed':SEED,'inventory_sha256':sha(inventory),
        'selection':'proportional question-type quotas; prefer unused support families within each type; SHA256 seeded order; no correctness or evidence-coverage filtering',
        'excluded_prior_qids':sorted(prior),'eligible_count':len(eligible),'eligible_types':dict(counts),
        'type_quotas':allocation,'questions':records,'summary':{'questions':len(records),
        'support_families':len(fc),'repeated_support_families':sum(v>1 for v in fc.values()),
        'retrieved_background_families':len(bc),'repeated_background_families':sum(v>1 for v in bc.values()),
        'questions_with_prior_support_family_overlap':sum(bool(r['prior_support_family_overlap']) for r in records),
        'prior_qid_overlap':sum(bool(r['prior_selected_cohorts']) for r in records),
        'gold_pages_verified':0,'status':'qid_pool_frozen_evidence_localization_pending'},
        'support_family_members':{f:[r['qid'] for r in records if f in r['support_families']] for f in fc},
        'background_family_members':{f:[r['qid'] for r in records if f in r['original_top4_families']] for f in bc}}
    p=Path(out);p.parent.mkdir(parents=True,exist_ok=True);payload=json.dumps(result,indent=2)+'\n'
    if p.exists():assert p.read_text()==payload
    else:
        with p.open('x') as f:f.write(payload)
    print(json.dumps({'manifest_sha256':sha(p),'eligible_types':dict(counts),'type_quotas':allocation,**result['summary']},indent=2))
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--inventory',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.inventory,a.out)
