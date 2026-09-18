from dataclasses import replace
from pathlib import Path
import pytest
import torch
from experiments.training_pilot.audit_banks import inspect_bank,legacy_reuse_receipt,summarize,verify_raw_artifacts
from test_stage2_contracts import example
from docprune.stage2.supervision import Outcome,TeacherBank

def fixture():
    x=example();x.budget=8
    masks=torch.tensor([[0,0,0,0],[1,0,0,0],[1,1,0,0],[1,1,1,1]],dtype=torch.bool)
    bank=TeacherBank(x.identity.key,masks,tuple(Outcome(None,s) for s in (-4.,-2.,-1.,0.)),Outcome(None,0.),True,True,'doc','s','sealed','variable_pilot_v1','hamming_families_v1')
    return x,bank

def test_inspect_bank_reports_complete_pair_diagnostics():
    x,b=fixture();row=inspect_bank(x,b,{'verdict':'correct','manifest_identity':'sealed'},4,False)
    assert row['masks']==4 and row['pairs']==sum(row['families'].values())
    assert len(row['sensitivity'])==9

@pytest.mark.parametrize('changed,message',[
    ({'baseline_correct':False},'stratum'),({'adjudication_identity':'other'},'adjudication'),
    ({'correct_preservation':'g'},'preservation'),({'proposal_uses_s':False},'routing'),
    ({'retention_mode':'exact'},'retention'),({'pair_weighting':'all_pairs'},'pair-weighting')])
def test_inspect_bank_rejects_provenance_channel_and_routing_drift(changed,message):
    x,b=fixture();label={'verdict':'correct','manifest_identity':'sealed'}
    with pytest.raises(ValueError,match=message):inspect_bank(x,replace(b,**changed),label,4,False)

def test_inspect_bank_rejects_wrong_mask_count_and_dev_routing():
    x,b=fixture();label={'verdict':'correct','manifest_identity':'sealed'}
    with pytest.raises(ValueError,match='complete masks'):inspect_bank(x,b,label,8,False)
    with pytest.raises(ValueError,match='routing'):inspect_bank(x,b,label,4,True)

def test_summary_exposes_zero_pair_questions_and_sensitivity():
    base=dict(split='train',correct=True,masks=4,pairs=0,families=dict(single_flip=0,two_flip=0,broad=0),sensitivity={'0.1/0.05':2})
    rows=[dict(base,qid='zero'),dict(base,qid='active',correct=False,pairs=3,families=dict(single_flip=1,two_flip=1,broad=1),sensitivity={'0.1/0.05':4})]
    result=summarize(rows,'train')
    assert result['correct']==result['incorrect']==1 and result['pairs']==3
    assert result['active_questions']==1 and result['zero_pair_qids']==['zero']
    assert result['sensitivity']['0.1/0.05']==dict(pairs=6,active_questions=2)

def test_raw_artifacts_match_sealed_bank_and_design(tmp_path):
    x,b=fixture();folder=tmp_path/'q';folder.mkdir()
    (folder/'design.json').write_text(__import__('json').dumps(dict(region_ids=list(x.layout.region_ids),costs=x.layout.costs.tolist())))
    (folder/'reference.json').write_text(__import__('json').dumps(dict(g=None,s=0.)))
    (folder/'measurements').mkdir()
    for i,(mask,outcome) in enumerate(zip(b.masks,b.outcomes)):
        row=dict(mask=mask.tolist(),g=outcome.g,s=outcome.s,retained_tokens=int((mask*x.layout.costs).sum()))
        (folder/'measurements'/f'{i:02d}.json').write_text(__import__('json').dumps(row))
    assert len(verify_raw_artifacts(folder,x,b))==64
    bad=folder/'measurements/01.json';row=__import__('json').loads(bad.read_text());row['retained_tokens']+=1;bad.write_text(__import__('json').dumps(row))
    with pytest.raises(ValueError,match='token cost'):verify_raw_artifacts(folder,x,b)

def test_legacy_reuse_requires_matching_parent_receipts_and_identities(tmp_path):
    parent=tmp_path/'smoke';folder=parent/'reuse-q';folder.mkdir(parents=True)
    current=dict(labels_sha256='labels',catalog='catalog',reader='reader',selector='selector',acquisition='acquisition',retrieval_schema='retrieval')
    old=dict(schema='correct-preservation-pilot-smoke-v1',smoke=dict(qids=['reuse-q'],labels_sha256='labels'),**{k:current[k] for k in ('catalog','reader','selector','acquisition','retrieval_schema')})
    teacher=dict(status='passed',questions=[dict(qid='reuse-q',masks=32,pairs=1)])
    for name,value in [('contract.json',old),('teacher-complete.json',teacher),('complete.json',dict(status='passed'))]:
        (parent/name).write_text(__import__('json').dumps(value))
    receipt,provenance=legacy_reuse_receipt(folder,'reuse-q',current)
    assert receipt['qid']=='reuse-q' and provenance['receipt_source']=='legacy_parent_teacher_complete'
    old['reader']='changed';(parent/'contract.json').write_text(__import__('json').dumps(old))
    with pytest.raises(ValueError,match='reader identity'):legacy_reuse_receipt(folder,'reuse-q',current)

def test_full_audit_completion_route_and_missing_receipt_fail_closed(tmp_path,monkeypatch):
    import experiments.training_pilot.audit_banks as module
    train=[f't{i}' for i in range(64)];dev=[f'd{i}' for i in range(24)];qids=train+dev
    records=[dict(qid=q,status='eligible',split='train' if q in train else 'dev',question='question '+q) for q in qids]
    split=dict(train64=train,dev24=dev,records=records,labels_sha256='labels')
    labels={q:dict(verdict='correct',manifest_identity='sealed',label_source='source',model_answer='answer') for q in qids}
    receipts=[dict(qid=q,masks=32 if q in train else 8,pairs=1,pair_families=[1,0,0],retained_min=0,retained_max=8,full_tokens=8) for q in qids]
    contract=dict(schema='quality-first-collection-v1',split_sha256='split',labels_sha256='labels',train_qids=train,dev_qids=dev,reuse={})
    complete=dict(status='passed',split_sha256='split',teacher_receipt_sha256='teacher',train_questions=64,dev_questions=24,reused_question_banks=0,new_question_banks=88)
    teacher=dict(status='passed',questions=receipts)
    monkeypatch.setattr(module,'sealed',lambda path:split)
    monkeypatch.setattr(module,'load_labels',lambda path:labels)
    monkeypatch.setattr(module,'sha',lambda path:{'split.json':'split','labels-deepseek-v4.1-flash.json':'labels','teacher-complete.json':'teacher'}.get(Path(path).name,'hash'))
    def fake_read(path):
        path=Path(path)
        if path.name=='contract.json':return contract
        if path.name=='collection-complete.json':return complete
        if path.name=='teacher-complete.json':return teacher
        if path.name=='question-complete.json':return receipts[qids.index(path.parent.name)]
        if path.name=='anchor.json':return dict(adjudication='sealed',label_source='source',answer='answer',continuation_ids=[1])
        raise AssertionError(path)
    monkeypatch.setattr(module,'read',fake_read)
    class Identity:
        question_id=None
    class Example:
        identity=Identity()
    monkeypatch.setattr(module,'load_example',lambda path:(setattr(Example.identity,'question_id',Path(path).parent.name) or Example(),object()))
    def fake_inspect(ex,bank,label,count,dev_flag):
        return dict(qid=ex.identity.question_id,correct=True,masks=count,pairs=1,families=dict(single_flip=1,two_flip=0,broad=0),retained_min=0,retained_max=8,full_tokens=8,pages=1,sensitivity={'0.1/0.05':1})
    monkeypatch.setattr(module,'inspect_bank',fake_inspect)
    monkeypatch.setattr(module,'verify_raw_artifacts',lambda folder,ex,bank:'design')
    written=[];monkeypatch.setattr(module,'publish',lambda path,value:written.append(value))
    result=module.audit(tmp_path,tmp_path/'collection',tmp_path/'audit.json')
    assert result['status']=='passed' and len(result['rows'])==88 and result['rows'][0]['design_sha256']=='design'
    complete['teacher_receipt_sha256']='wrong'
    failed=module.audit(tmp_path,tmp_path/'collection',tmp_path/'failed.json')
    assert failed['status']=='failed' and failed['failures'][0]['scope']=='completion'
