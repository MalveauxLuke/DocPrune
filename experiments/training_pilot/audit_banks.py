"""Fail-closed, CPU-only audit of the frozen 64/24 pilot collection."""
import argparse
import json
from pathlib import Path
import re
import sys
sys.path[:0]=[str(Path(__file__).resolve().parents[2]/'src'),str(Path(__file__).resolve().parents[2])]
from experiments.training_pilot.run import read,sha
from docprune.answer_adjudication import read as sealed
from docprune.stage2.api_version import load_labels
from docprune.stage2.data import load_example
from docprune.stage2.storage import publish

EPSILONS=(.05,.1,.2);MARGINS=(.02,.05,.1)

def require(ok,message):
    if not ok:raise ValueError(message)

def normalized(text):return ' '.join(re.findall(r'\w+',text.casefold()))

def pair_summary(bank,epsilon,margin):
    pairs=bank.pairs('gold_aware',epsilon,margin)
    distances=[int((bank.masks[i]!=bank.masks[j]).sum()) for i,j in pairs]
    return dict(pairs=len(pairs),families=dict(single_flip=sum(d==1 for d in distances),two_flip=sum(d==2 for d in distances),broad=sum(d>2 for d in distances)))

def inspect_bank(ex,bank,label,count,dev):
    require(label.get('verdict') in ('correct','incorrect'),'Unresolved frozen verdict')
    require(bank.baseline_correct==(label['verdict']=='correct'),'Bank stratum differs from frozen verdict')
    require(bank.adjudication_identity==label.get('manifest_identity'),'Frozen adjudication identity mismatch')
    require(bank.correct_preservation=='s','Pilot bank must use S for correct-case preservation')
    require(bank.proposal_uses_s is (not dev),'Acquisition routing differs from frozen train/dev contract')
    require(bank.retention_mode=='variable_pilot_v1','Unexpected retention contract')
    require(bank.pair_weighting=='hamming_families_v1','Unexpected pair-weighting contract')
    bank.validate(ex)
    expected=min(count,2**len(ex.layout.region_ids))
    require(len(bank.masks)==expected,f'Expected {expected} complete masks, found {len(bank.masks)}')
    require(all((o.g is None)==bank.baseline_correct for o in (*bank.outcomes,bank.reference)),'G/S channel routing mismatch')
    primary=pair_summary(bank,.1,.05);costs=(bank.masks*ex.layout.costs).sum(1)
    return dict(qid=ex.identity.question_id,correct=bank.baseline_correct,masks=len(bank.masks),**primary,
        retained_min=int(costs.min()),retained_max=int(costs.max()),full_tokens=ex.budget,pages=len(ex.identity.ordered_pages),
        sensitivity={f'{e:g}/{m:g}':pair_summary(bank,e,m)['pairs'] for e in EPSILONS for m in MARGINS})

def verify_raw_artifacts(folder,ex,bank):
    """Prove the sealed example bank still equals its raw collection records."""
    folder=Path(folder);design=read(folder/'design.json')
    require(design.get('region_ids')==list(ex.layout.region_ids),'Design region identities differ from sealed example')
    require(design.get('costs')==ex.layout.costs.tolist(),'Design token costs differ from sealed example')
    reference=read(folder/'reference.json')
    require(reference==dict(g=bank.reference.g,s=bank.reference.s),'Raw reference differs from sealed bank')
    for i,(mask,outcome) in enumerate(zip(bank.masks,bank.outcomes)):
        measured=read(folder/'measurements'/f'{i:02d}.json')
        expected_mask=mask.tolist();expected_tokens=int((mask*ex.layout.costs).sum())
        require(measured.get('mask')==expected_mask,f'Raw measurement {i} mask differs from sealed bank')
        require(measured.get('g')==outcome.g and measured.get('s')==outcome.s,f'Raw measurement {i} G/S differs from sealed bank')
        require(measured.get('retained_tokens')==expected_tokens,f'Raw measurement {i} token cost differs from sealed bank')
    return sha(folder/'design.json')

def summarize(rows,split):
    selected=[r for r in rows if r['split']==split]
    sensitivity={k:dict(pairs=sum(r['sensitivity'][k] for r in selected),active_questions=sum(r['sensitivity'][k]>0 for r in selected)) for k in (selected[0]['sensitivity'] if selected else ())}
    return dict(questions=len(selected),correct=sum(r['correct'] for r in selected),incorrect=sum(not r['correct'] for r in selected),
        masks=sum(r['masks'] for r in selected),pairs=sum(r['pairs'] for r in selected),active_questions=sum(r['pairs']>0 for r in selected),
        zero_pair_qids=[r['qid'] for r in selected if not r['pairs']],families={n:sum(r['families'][n] for r in selected) for n in ('single_flip','two_flip','broad')},sensitivity=sensitivity)

def audit(root,collection,output):
    root,collection=Path(root),Path(collection);failures=[];rows=[]
    splitpath=root/'training-pilot-v1/split-quality-v1/split.json';contractpath=collection/'contract.json'
    split_sha=contract_sha=None;train=[];dev=[];labels={};contract={};teacher_questions={}
    try:
        split=sealed(splitpath);contract=read(contractpath);split_sha=sha(splitpath);contract_sha=sha(contractpath)
        require(contract.get('schema')=='quality-first-collection-v1','Unexpected collection contract schema')
        require(contract.get('split_sha256')==split_sha,'Collection/split hash mismatch')
        train,dev=split['train64'],split['dev24']
        require(contract.get('train_qids')==train and contract.get('dev_qids')==dev,'Collection QID order differs from frozen split')
        require(len(train)==len(set(train))==64,'Frozen training subset must contain 64 unique QIDs')
        require(len(dev)==len(set(dev))==24,'Frozen development subset must contain 24 unique QIDs')
        require(not set(train)&set(dev),'Training and development QIDs overlap')
        records={r['qid']:r for r in split['records']}
        require(all(records[q]['status']=='eligible' for q in train+dev),'Selected QID is not eligible')
        require(all(records[q]['split']==('train' if q in train else 'dev') for q in train+dev),'Selected QID has inconsistent parent split')
        require(not {normalized(records[q]['question']) for q in train}&{normalized(records[q]['question']) for q in dev},'Normalized question text crosses train/dev')
        labelpath=root/'training-pilot-v1/labels-deepseek-v4.1-flash.json';labelsha=sha(labelpath)
        require(labelsha==split['labels_sha256']==contract.get('labels_sha256'),'Frozen-label hash mismatch')
        labels=load_labels(labelpath);require(set(train+dev)<=set(labels),'Frozen label missing for selected QID')
        reuse=contract.get('reuse');require(isinstance(reuse,dict) and set(reuse)<=set(train),'Reuse must be an explicit training-only map')
        require(all(isinstance(v,dict) for v in reuse.values()),'Invalid reuse provenance record')
    except (AssertionError,ValueError,OSError,KeyError,TypeError) as error:
        failures.append(dict(scope='collection',qid=None,error=f'{type(error).__name__}: {error}'))
    if not failures:
        try:
            complete=read(collection/'collection-complete.json');teacher=read(collection/'teacher-complete.json')
            require(complete.get('status')=='passed','Collection completion status is not passed')
            require(complete.get('split_sha256')==split_sha,'Completion/split hash mismatch')
            require(complete.get('teacher_receipt_sha256')==sha(collection/'teacher-complete.json'),'Teacher receipt hash mismatch')
            require(complete.get('train_questions')==64 and complete.get('dev_questions')==24,'Completion subset counts mismatch')
            require(complete.get('reused_question_banks')==len(contract['reuse']),'Completion reuse count mismatch')
            require(complete.get('new_question_banks')==88-len(contract['reuse']),'Completion new-bank count mismatch')
            require(teacher.get('status')=='passed','Teacher completion status is not passed')
            receipts=teacher.get('questions');require(isinstance(receipts,list),'Teacher receipt lacks question records')
            ids=[r.get('qid') for r in receipts];expected=[q for q in train+dev if q not in contract['reuse']]
            require(ids==expected and len(ids)==len(set(ids)),'Teacher receipt QIDs/order differ from new-bank contract')
            teacher_questions=dict(zip(ids,receipts))
        except (AssertionError,ValueError,OSError,KeyError,TypeError) as error:
            failures.append(dict(scope='completion',qid=None,error=f'{type(error).__name__}: {error}'))
    if train and dev:
        for qid in train+dev:
            reused=qid in contract.get('reuse',{});folder=Path(contract['reuse'][qid]['directory']) if reused else collection/qid
            try:
                receipt=read(folder/'question-complete.json');require(receipt.get('qid')==qid,'Question receipt QID mismatch')
                if reused:
                    require(sha(folder/'example/manifest.json')==contract['reuse'][qid].get('example_manifest_sha256'),'Reused example provenance mismatch')
                    require(sha(folder/'anchor.json')==contract['reuse'][qid].get('anchor_sha256'),'Reused anchor provenance mismatch')
                else:require(teacher_questions.get(qid)==receipt,'Question receipt differs from teacher completion receipt')
                anchor=read(folder/'anchor.json')
                require(anchor.get('adjudication')==labels[qid]['manifest_identity'],'Anchor adjudication identity mismatch')
                require(anchor.get('label_source')==labels[qid]['label_source'],'Anchor label source mismatch')
                require(anchor.get('answer')==labels[qid]['model_answer'],'Anchor answer differs from frozen label')
                require(isinstance(anchor.get('continuation_ids'),list) and anchor['continuation_ids'],'Anchor continuation is empty')
                ex,bank=load_example(folder/'example');require(ex.identity.question_id==qid,'Example QID mismatch')
                row=inspect_bank(ex,bank,labels[qid],8 if qid in dev else 32,qid in dev)
                design_sha=verify_raw_artifacts(folder,ex,bank)
                require(receipt.get('masks')==row['masks'],'Question receipt mask count mismatch')
                require(receipt.get('pairs')==row['pairs'],'Question receipt pair count mismatch')
                require(receipt.get('pair_families')==list(row['families'].values()),'Question receipt pair families mismatch')
                require(receipt.get('retained_min')==row['retained_min'] and receipt.get('retained_max')==row['retained_max'],'Question receipt retained-token range mismatch')
                require(receipt.get('full_tokens')==row['full_tokens'],'Question receipt full-token count mismatch')
                rows.append(dict(**row,split='dev' if qid in dev else 'train',reused=reused,directory=str(folder),manifest_sha256=sha(folder/'example/manifest.json'),design_sha256=design_sha,question_receipt_sha256=sha(folder/'question-complete.json')))
                del ex,bank
            except (AssertionError,ValueError,OSError,KeyError,TypeError) as error:
                failures.append(dict(scope='question',qid=qid,error=f'{type(error).__name__}: {error}'))
    summaries={name:summarize(rows,name) for name in ('train','dev')}
    if len(rows)==88:
        for name in ('train','dev'):
            if not summaries[name]['active_questions']:failures.append(dict(scope='yield',qid=None,error=f'No {name} question has a strict pair at epsilon=0.1, margin=0.05'))
    record=dict(schema='pilot-bank-audit-v2',status='passed' if not failures and len(rows)==88 else 'failed',collection_contract_sha256=contract_sha,split_sha256=split_sha,
        primary_threshold=dict(epsilon=.1,margin=.05),expected=dict(train_questions=64,dev_questions=24,train_masks=2048,dev_masks=192),
        observed=dict(rows=len(rows),reused=sum(r['reused'] for r in rows)),summaries=summaries,rows=rows,failures=failures,
        train_active=summaries['train']['active_questions'],dev_active=summaries['dev']['active_questions'])
    publish(output,record);return record

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','collection','output'):p.add_argument('--'+n,required=True)
    a=p.parse_args();r=audit(a.root,a.collection,a.output)
    compact=dict(status=r['status'],observed=r['observed'],train={k:r['summaries']['train'][k] for k in ('questions','pairs','active_questions','zero_pair_qids')},dev={k:r['summaries']['dev'][k] for k in ('questions','pairs','active_questions','zero_pair_qids')},failures=r['failures'])
    print(json.dumps(compact,separators=(',',':')))
    if r['status']!='passed':raise SystemExit(1)
