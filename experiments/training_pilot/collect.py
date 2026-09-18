"""One reader load for frozen train64 + dev24; existing smoke banks remain immutable."""
import argparse
from pathlib import Path
import sys
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'src'),str(REPO)]
from experiments.training_pilot.run import teacher,read,sha,publish,load_catalog,load_labels,VERSION,RETRIEVAL_SCHEMA,REVISION,SELECTOR_REV
from docprune.answer_adjudication import read as sealed


def collection_context(a):
    root=Path(a.root);out=Path(a.output);directory=root/'training-pilot-v1/split-quality-v1'
    split=sealed(directory/'split.json');train=sealed(directory/'train64.json');dev=sealed(directory/'dev24.json')
    assert train['split_sha256']==dev['split_sha256']==sha(directory/'split.json')
    assert train['qids']==split['train64'] and dev['qids']==split['dev24']
    qids=train['qids']+dev['qids'];assert len(qids)==len(set(qids))==88
    cohort=root/'training463-evidence-filtered-v1';selection=read(cohort/'selection.json')
    assert sha(cohort/'selection.json')==split['cohort_selection_sha256']
    assert sha(cohort/'pool.json')==selection['pool_sha256']
    pool={q['qid']:q for q in read(cohort/'pool.json')['questions']}
    labelpath=root/'training-pilot-v1/labels-deepseek-v4.1-flash.json'
    assert sha(labelpath)==split['labels_sha256'];labels=load_labels(labelpath)
    assert set(qids)<=set(selection['qids'])
    rows={r['qid']:r for r in split['records']}
    assert all(rows[q]['status']=='eligible' and rows[q]['verdict']==labels[q]['verdict'] for q in qids)
    catalog,digest=load_catalog(root)
    reuse={};old=root/'training-pilot-smoke-v1'
    old_contract=read(old/'contract.json')
    assert old_contract['catalog']==digest and old_contract['acquisition']==VERSION
    assert old_contract['smoke']['labels_sha256']==split['labels_sha256']
    assert old_contract['reader']==REVISION and old_contract['retrieval_schema']==RETRIEVAL_SCHEMA
    assert read(old/'complete.json')['status']=='passed'
    from docprune.stage2.data import load_example
    for q in old_contract['smoke']['qids']:
        assert q in train['qids']
        ex,bank=load_example(old/q/'example')
        assert ex.identity.question_id==q and bank.adjudication_identity==labels[q]['manifest_identity']
        assert bank.retention_mode=='variable_pilot_v1' and len(bank.masks)==32 and bank.baseline_correct
        reuse[q]=dict(directory=str(old/q),example_manifest_sha256=sha(old/q/'example/manifest.json'),anchor_sha256=sha(old/q/'anchor.json'))
        del ex,bank
    manifest=dict(schema='quality-first-collection-v1',split_sha256=sha(directory/'split.json'),catalog=digest,labels_sha256=split['labels_sha256'],reader=REVISION,selector=SELECTOR_REV,acquisition=VERSION,retrieval_schema=RETRIEVAL_SCHEMA,train_qids=train['qids'],dev_qids=dev['qids'],reuse=reuse,sources={str(p.relative_to(REPO)):sha(p) for p in [Path(__file__),REPO/'experiments/training_pilot/run.py',REPO/'src/docprune/stage2/pilot.py']})
    publish(out/'contract.json',manifest)
    jobs=dict(qids=[q for q in qids if q not in reuse],mask_count=32,mask_counts={q:8 for q in dev['qids']},dev_qids=dev['qids'],test_empty_interface=False,require_pairs=False)
    return root,out,catalog,pool,labels,jobs,manifest


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for name in ('root','output','reader'):p.add_argument('--'+name,required=True)
    a=p.parse_args()
    import torch,transformers
    assert transformers.__version__=='4.57.3' and torch.cuda.is_available()
    torch.set_num_threads(2)
    teacher(a,context_fn=collection_context)
    root,out,catalog,pool,labels,jobs,contract=collection_context(a)
    publish(out/'collection-complete.json',dict(status='passed',train_questions=64,dev_questions=24,new_question_banks=len(jobs['qids']),reused_question_banks=len(contract['reuse']),teacher_receipt_sha256=sha(out/'teacher-complete.json'),split_sha256=contract['split_sha256']))
