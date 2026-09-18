"""Freeze the owner-approved quality-first pilot from existing evidence reviews."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import sys
REPO=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(REPO/'src'))
from docprune.answer_adjudication import seal,read


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def norm(s):return ' '.join(re.findall(r'\w+',s.casefold()))
def tie(qid):return hashlib.sha256(('quality-first-v1|1729|'+qid).encode()).hexdigest()


def build(output):
    base=REPO/'outputs/m3doc600-review-v2';cohort=base/'training463-evidence-filtered-v1'
    labels_path=REPO/'outputs/training-pilot-v1/labels-deepseek-v4.1-flash.json'
    pilot_path=REPO/'outputs/colfeatures17-received-2026-09-11/input/data/case-index.json'
    ten_path=REPO/'outputs/omp10-20260917-v1/received/omp10-20260917-v1/review/frozen.json'
    pool=load(cohort/'pool.json')['questions'];by={q['qid']:q for q in pool}
    labels={r['qid']:r for r in read(labels_path)['rows']};baseline={r['qid']:r for r in load(cohort/'baseline-rows.json')['rows']}
    pilots={r['case_id'].split('::')[0]:r for r in load(pilot_path)}
    ten={r['qid']:r for r in load(ten_path)['selected']}
    smoke=set(load(REPO/'outputs/training-pilot-v1/smoke.json')['qids'])
    hold='4630c01f960500291aa851e5bf5313a7'
    records={};sources={str(p.relative_to(REPO)):sha(p) for p in [cohort/'pool.json',cohort/'selection.json',labels_path,pilot_path,ten_path,Path(__file__)]}
    for q in pool:
        qid=q['qid'];label=labels[qid];b=baseline[qid]
        assert label['question']==q['question'] and label['gold']==[str(a['answer']) for a in q['answers']]
        assert label['model_answer']==b['answer'] and label['source_answer_file_sha256']==b['answer_sha256']
        rp=q['review_provenance']['effective_review'];path=base/rp['path'];review=load(path)
        assert sha(path)==rp['sha256']
        evidence={(p['doc_id'],p['page_index']) for p in q['evidence_pages']}
        admitted={(p['doc_id'],p['page_index']) for p in q['admitted_pages']}
        assert evidence and evidence<=admitted
        uncertainty=review.get('uncertainty');u=uncertainty.get('level','') if isinstance(uncertainty,dict) else str(uncertainty)
        rationale=' '.join(str(review.get(k,'')) for k in ['rationale','reasoning','verified_answer','evidence_items','manual_relation_check'])
        low=u.lower()=='low';concrete=len(rationale)>180
        status='eligible' if label['verdict'] in ('correct','incorrect') else 'quarantined_uncertain_label'
        if qid==hold:status='held_missing_table_header'
        exposure=[name for name,ids in [('pilot17',pilots),('audit10',ten),('smoke',smoke)] if qid in ids]
        records[qid]=dict(qid=qid,question=q['question'],gold=label['gold'],baseline_answer=label['model_answer'],verdict=label['verdict'],label_source=label['label_source'],modality=q['metadata']['type'],context_stratum=b['context_stratum'],status=status,prior_analysis=exposure,review_path=str(path.relative_to(REPO)),review_sha256=sha(path),quality=dict(explicit_low_uncertainty=low,evidence_notes_present=concrete,evidence_page_count=len(evidence),basis='existing reviewed evidence and resolved answer labels; no mask-outcome filtering'),evidence_pages=q['evidence_pages'],admitted_pages=q['admitted_pages'],presentation_order=q['presentation_order'],support_families=q['support_families'])
    forced={q for q,r in records.items() if r['prior_analysis'] and r['status']=='eligible'}
    def quality(q):
        r=records[q];return (not r['quality']['explicit_low_uncertainty'],not r['quality']['evidence_notes_present'],tie(q))
    # Balance modality and correctness; context coverage breaks quality ties.
    def select(candidates,targets,initial=()):
        chosen=list(sorted(initial,key=tie));used={norm(records[q]['question']) for q in chosen}
        for modality,n in targets.items():
            for verdict in ('incorrect','correct'):
                target=n//2+(n%2 if verdict=='correct' else 0)
                bucket=[q for q in candidates if records[q]['modality']==modality and records[q]['verdict']==verdict and q not in chosen]
                while sum(records[q]['modality']==modality and records[q]['verdict']==verdict for q in chosen)<target:
                    eligible=[q for q in bucket if norm(records[q]['question']) not in used]
                    if not eligible:raise ValueError('Insufficient eligible distinct questions')
                    counts=Counter(records[q]['context_stratum'] for q in chosen if records[q]['modality']==modality and records[q]['verdict']==verdict)
                    q=min(eligible,key=lambda q:(*quality(q)[:2],counts[records[q]['context_stratum']],tie(q)))
                    chosen.append(q);used.add(norm(records[q]['question']));bucket.remove(q)
        assert len(chosen)==sum(targets.values())
        return sorted(chosen,key=tie)
    eligible={q for q,r in records.items() if r['status']=='eligible'}
    train64=select(eligible,{'TextQ':26,'TableQ':22,'ImageQ':16},forced)
    # Reserve every audited/smoked question in train; overlap of documents is unrestricted.
    forbidden={norm(records[q]['question']) for q in records if records[q]['prior_analysis'] or q in train64}
    dev_candidates={q for q in eligible if norm(records[q]['question']) not in forbidden}
    dev24=select(dev_candidates,{'TextQ':10,'TableQ':8,'ImageQ':6})
    dev72=select(dev_candidates,{'TextQ':30,'TableQ':26,'ImageQ':16},dev24)
    train391=sorted(set(records)-set(dev72),key=tie)
    assert len(train391)==391 and len(dev72)==72 and set(train64)<=set(train391) and set(dev24)<=set(dev72)
    assert not ({norm(records[q]['question']) for q in train391}&{norm(records[q]['question']) for q in dev72})
    assert forced<=set(train64) and all(records[q]['status']=='eligible' for q in train64+dev72)
    for q,r in records.items():r.update(split='dev' if q in dev72 else 'train',starter=q in train64,working_dev=q in dev24)
    exceptions=[]
    for q,r in pilots.items():
        if q not in records:exceptions.append(dict(qid=q,historical_id=r['id'],question=r['question'],requested_source='pilot17',status='training_candidate_not_in_current463',reason='No current frozen Qwen3 baseline/API-label/admitted-feature package; historical scores cannot substitute.'))
    exceptions.append(dict(qid=hold,requested_source='audit10',status='train_held_not_in_starter',reason='Visible continuation row lacks season-number header in admitted context; preserve analysis pending evidence repair.'))
    def summary(ids):
        return dict(count=len(ids),eligible=sum(records[q]['status']=='eligible' for q in ids),modality=dict(Counter(records[q]['modality'] for q in ids)),correctness=dict(Counter(records[q]['verdict'] for q in ids)),context=dict(Counter(records[q]['context_stratum'] for q in ids)))
    result=dict(schema='docprune-quality-first-split-v1',seed=1729,sources=sources,labels_sha256=sha(labels_path),cohort_selection_sha256=sha(cohort/'selection.json'),policy='Quality-first existing evidence review; prior analysis deliberately favored; document overlap and exposure not exclusion criteria; QIDs disjoint; no test set.',train=train391,dev=dev72,train64=train64,dev24=dev24,records=[records[q] for q in sorted(records)],exceptions=exceptions,summary={name:summary(ids) for name,ids in [('train',train391),('dev',dev72),('train64',train64),('dev24',dev24)]})
    out=Path(output);seal(out/'split.json',result)
    for name,ids,count in [('train64',train64,32),('dev24',dev24,8)]:
        seal(out/(name+'.json'),dict(schema='docprune-pilot-collection-v1',split_sha256=sha(out/'split.json'),subset=name,qids=ids,mask_count=count,labels_sha256=sha(labels_path),cohort_selection_sha256=sha(cohort/'selection.json'),acquisition='adaptive' if name=='train64' else 'independent_dev',records=[records[q] for q in ids]))
    lines=['# Frozen quality-first pilot split','', 'Source: unchanged training463; frozen API labels. Prior exposure/document overlap allowed. No test set.','',json.dumps(result['summary'],indent=2),'','## Historical inclusion',f"{len(set(pilots)&set(train64))}/17 pilot cases and {len(set(ten)&set(train64))}/10 audited cases are in train64. Exceptions are explicit below.",'']
    lines += [f"- {e['qid']}: {e['reason']}" for e in exceptions]
    for name,ids in [('Training starter (64)',train64),('Working development (24)',dev24)]:
        lines += ['', '## '+name,'','| QID | Type | Baseline | Question |','|---|---|---|---|']
        lines += [f"| {q} | {records[q]['modality']} | {records[q]['verdict']} | {records[q]['question'].replace('|','/')} |" for q in ids]
    (out/'README.md').write_text('\n'.join(lines)+'\n')
    seal(out/'verification.json',dict(qids_unique=True,all463_accounted=True,train_dev_qid_disjoint=True,normalized_question_disjoint=True,evidence_pages_admitted=True,review_hashes_verified=True,labels_and_baseline_matched=True,prior_analysis_not_excluded=True,mask_outcomes_unused=True,files={p.name:sha(p) for p in sorted(out.glob('*')) if p.name!='verification.json'}))
    print(json.dumps(result['summary'],indent=2));print('forced',len(forced),'exceptions',len(exceptions))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);build(p.parse_args().output)
