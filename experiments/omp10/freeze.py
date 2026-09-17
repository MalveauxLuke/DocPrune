"""Bind reviewed decisions to immutable original baseline files on SOL."""
import argparse
from collections import Counter
import os
from pathlib import Path
import socket
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'m3doc600'))
from common import read,publish,sha,fingerprint


def main(root,decisions):
    assert os.environ.get('SLURM_JOB_ID') and 'login' not in socket.gethostname()
    root=Path(root); rows=read(decisions)
    assert len(rows)==10 and len({r['qid'] for r in rows})==10
    assert Counter(r['modality'] for r in rows)=={'TextQ':3,'TableQ':3,'ImageQ':4}
    c=read(root/'catalog.json');pool={q['qid']:q for q in read(c['pool_path'])['questions']}
    for r in rows:
        q=pool[r['qid']]; p=root/'baseline-qwen3-8b-admitted-v2/answers'/f"{fingerprint(r['qid'])}.json";b=read(p)
        assert r['decision']=='confirmed_wrong' and r['reason'] and r['evidence_pages']
        assert r['gold_items']==[a['answer'] for a in q['answers']]
        admitted={(v['doc_id'],v['page_index'],v['pdf_sha256']) for v in q['admitted_pages']}
        assert all((v['doc_id'],v['page_index'],v['pdf_sha256']) in admitted for v in r['evidence_pages'])
        assert r['modality']==b['modality'] and r['context_stratum']==b['context_stratum']
        r.update(baseline_file_sha256=sha(p),baseline_answer=b['answer'],visual_tokens=b['visual_tokens'])
    # Largest context plus largest context from another modality bounds memory and varies content.
    ordered=sorted(rows,key=lambda r:(r['visual_tokens'],r['qid']),reverse=True)
    smoke=[ordered[0]['qid'],next(r['qid'] for r in ordered if r['modality']!=ordered[0]['modality'])]
    out=root/'omp10-20260917-v1/review/frozen.json'
    publish(out,dict(status='frozen_before_mask_scoring',selected=rows,smoke_qids=smoke,
                    decisions_sha256=sha(decisions),selection='Semantic evidence review before masks; development audit, not prevalence estimate'))
    print('FROZEN',sha(out),'SMOKE',smoke,flush=True)
    print([(r['qid'],r['modality'],r['visual_tokens']) for r in rows],flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--decisions',required=True)
    a=p.parse_args();main(a.root,a.decisions)
