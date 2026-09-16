"""CPU-only inventory; deliberately excludes historical predictions/scores."""
import argparse, collections, hashlib, json, os
from pathlib import Path

TYPES = {'TextQ', 'TableQ', 'ImageQ', 'ImageListQ'}
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(1048576), b''): h.update(b)
    return h.hexdigest()
def read(p): return json.loads(Path(p).read_text())
def qids(v, known):
    if isinstance(v, str): return {v} & known
    if isinstance(v, list): return set().union(*(qids(x,known) for x in v)) if v else set()
    if isinstance(v, dict): return (set(v)&known) | set().union(*(qids(x,known) for x in v.values())) if v else set()
    return set()
def run(base,out):
    assert os.environ.get('SLURM_JOB_ID'), 'compute allocation required'
    base=Path(base);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    raw=base/'datasets/m3docvqa/multimodalqa';qp=raw/'MMQA_dev.jsonl'
    questions=[json.loads(x) for x in qp.read_text().splitlines() if x.strip()]
    known={q['qid'] for q in questions};assert len(known)==len(questions)==2441
    rp=base/'benchmark-4e2473b/attempt-2/eval-quality/docprune/top4/run/results.jsonl'
    retrieved={}
    for line in rp.open():
        r=json.loads(line); k=r['question_id'];assert k not in retrieved
        retrieved[k]=r['retrieved_pages'];assert len(retrieved[k])==4
    hp=base/'task6-holdout-primary-1213-v1/sealed-holdout/manifest.json';hold=read(hp)
    files={'historical_development_registry':hold['development_registry_path'],
           'pilot48':str(base/'task9-preliminary-random48-v1/cohort.json'),
           'confirmation100':str(base/'task9-baseline-wrong100-inputs-c0c9bee-v1/reference.json'),
           'older600':str(base/'task9-shared-probe-random600-v1/cohort.json')}
    exposure={};provenance={str(p):sha(p) for p in [qp,rp,hp]}
    for label,path in files.items():
        p=Path(path)
        if not p.exists(): raise FileNotFoundError(p)
        exposure[label]=sorted(qids(read(p),known));provenance[str(p)]=sha(p)
    excluded=set(exposure['historical_development_registry'])|set(exposure['pilot48'])|set(exposure['confirmation100'])
    eligible=[q for q in questions if q['metadata']['type'] in TYPES and q['qid'] in retrieved and q['qid'] not in excluded]
    support_ids={x['doc_id'] for q in eligible for x in q['supporting_context']}
    sources={}
    for name in ['MMQA_texts.jsonl','MMQA_tables.jsonl','MMQA_images.jsonl']:
        p=raw/name;provenance[str(p)]=sha(p)
        sources[name]={}
        for line in p.open():
            r=json.loads(line)
            if r['id'] in support_ids:sources[name][r['id']]=r
    result={'schema':'m3doc600-inventory-v1','source_hashes':provenance,'questions':questions,
            'retrieved_pages':retrieved,'exposure':exposure,'eligible_qids':[q['qid'] for q in eligible],
            'sources':sources,'historical_baseline_scored_qids':sorted(retrieved),
            'summary':{'total_questions':len(questions),'cached_questions':len(retrieved),
            'single_hop_types':dict(collections.Counter(q['metadata']['type'] for q in questions if q['metadata']['type'] in TYPES)),
            'eligible_types':dict(collections.Counter(q['metadata']['type'] for q in eligible)),
            'eligible_count':len(eligible),'exposure_counts':{k:len(v) for k,v in exposure.items()}}}
    p=out/'inventory.json'
    payload=json.dumps(result,indent=2)+'\n'
    if p.exists():assert p.read_text()==payload
    else:p.write_text(payload)
    (out/'inventory-summary.json').write_text(json.dumps(dict(result['summary'],inventory_sha256=sha(p)),indent=2)+'\n')
    print(json.dumps(result['summary'],indent=2),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.base,a.out)
