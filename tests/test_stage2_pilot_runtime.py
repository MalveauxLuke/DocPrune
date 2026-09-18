import copy
from dataclasses import replace
import pytest
import torch
from test_stage2_pilot import setup
from docprune.stage2.pilot import phase_optimizer,train_epoch,branch_scheduler
from docprune.stage2.pilot_runtime import TensorCache,checkpoint,restore_parameters,adaptable_state
from experiments.training_pilot.audit_banks import inspect_bank
from experiments.training_pilot.train64 import ranking_metrics,evaluate


def test_disk_cache_restart_parity_and_lora_rejection(tmp_path):
    x,b,cfg,m,batch=setup();m.disk_cache=TensorCache(tmp_path,{'revision':'test'})
    phase_optimizer(m,'warmup');m.eval()
    with torch.no_grad():direct=m(x,batch['proxy_prompts'][0],bypass_cache=True)
    m.hidden_cache.clear();m.vision_cache.clear()
    with torch.no_grad():cached=m(x,batch['proxy_prompts'][0])
    torch.testing.assert_close(direct,cached,rtol=0,atol=0)
    assert m.calls['language']==1 and m.calls['vision']==1
    # Subsequent epochs need no pixel load/transfer once native vision is sealed.
    batch['proxy_prompts'][0].pixel_values=None
    opt=phase_optimizer(m,'lora');train_epoch(m,[batch],cfg,opt)
    assert m.calls['language']==2 and m.calls['vision']==1
    with pytest.raises(ValueError,match='identity LoRA'):phase_optimizer(m,'frozen')
    path=next(tmp_path.rglob('data.safetensors'));path.write_bytes(b'corrupt')
    seal=path.parent
    with pytest.raises(ValueError,match='Changed cached'):m.disk_cache.get(seal.parent.name,seal.name)


def test_epoch_checkpoint_exact_resume_and_frozen_branch_identity(tmp_path):
    x,b,cfg,m,batch=setup();opt=phase_optimizer(m,'warmup')
    train_epoch(m,[batch],cfg,opt)
    checkpoint(tmp_path/'warm.pt',m,'contract','warmup',1,opt,None,[],None,0)
    original=adaptable_state(m)
    opt=phase_optimizer(m,'lora');sched=branch_scheduler(opt,4)
    train_epoch(m,[batch]*4,cfg,opt,scheduler=sched)
    checkpoint(tmp_path/'latest.pt',m,'contract','lora',1,opt,sched,[],1.,0)
    train_epoch(m,[batch]*4,cfg,opt,scheduler=sched);expected=adaptable_state(m)
    rec=restore_parameters(tmp_path/'latest.pt',m,'contract')
    opt=phase_optimizer(m,'lora');sched=branch_scheduler(opt,4)
    opt.load_state_dict(rec['optimizer']);sched.load_state_dict(rec['scheduler'])
    train_epoch(m,[batch]*4,cfg,opt,scheduler=sched)
    for k,v in adaptable_state(m).items():torch.testing.assert_close(v,expected[k],rtol=0,atol=0)
    restore_parameters(tmp_path/'warm.pt',m,'contract');phase_optimizer(m,'frozen')
    for k,v in adaptable_state(m).items():torch.testing.assert_close(v,original[k],rtol=0,atol=0)
    with pytest.raises(ValueError,match='contract'):restore_parameters(tmp_path/'warm.pt',m,'different')


def test_audit_channel_and_acquisition_routing():
    x,b,cfg,m,batch=setup();label={'verdict':'correct','manifest_identity':'sealed'}
    report=inspect_bank(x,b,label,4,False);assert report['pairs']>0
    with pytest.raises(ValueError):inspect_bank(x,b,dict(label,verdict='incorrect'),4,False)
    with pytest.raises(ValueError):inspect_bank(x,b,label,4,True)
    with pytest.raises(ValueError):inspect_bank(x,b,label,8,False)


def test_dev_evaluation_does_not_update_and_emits_two_capacities():
    x,b,cfg,m,batch=setup();phase_optimizer(m,'warmup');before=adaptable_state(m)
    class Stream:
        def batches(self,qids):
            assert qids==['dev-only']
            yield batch
    result,masks=evaluate(m,Stream(),['dev-only'])
    assert result['head1_loss']>0 and set(masks['dev-only'])=={'0.75','0.5'}
    assert masks['dev-only']['0.75']['retained_tokens']<=6
    assert masks['dev-only']['0.5']['retained_tokens']<=4
    for k,v in adaptable_state(m).items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
    for p in m.parameters():assert p.grad is None
    perfect=torch.tensor([-4.,-2.,-1.,0.])
    assert ranking_metrics(perfect,b)['accuracy']==1.


def test_streaming_three_phase_runner_and_completed_resume(tmp_path,monkeypatch):
    from types import SimpleNamespace
    from docprune.stage2.storage import publish,read_record
    from experiments.training_pilot import train64 as runner
    x,b,cfg,initial,batch=setup()
    train_ids=['train-a','train-b'];dev_ids=[f'dev-{i}' for i in range(24)]
    rows=[dict(qid=q,split='train' if q in train_ids else 'dev',pairs=6) for q in train_ids+dev_ids]
    audit=tmp_path/'audit.json';publish(audit,dict(status='passed',rows=rows,train_active=2,dev_active=24))
    seen=[]
    class Stream:
        def __init__(self,*args):pass
        def batch(self,qid):return batch
        def batches(self,qids):
            seen.append(list(qids))
            for q in qids:yield batch
    monkeypatch.setattr(runner,'Stream',Stream)
    monkeypatch.setattr(runner,'load_model',lambda _: (object(), __import__('test_stage2_models').tiny_model()))
    monkeypatch.setattr(runner,'ExperimentConfig',lambda **kw:cfg)
    monkeypatch.setattr(runner.torch.cuda,'reset_peak_memory_stats',lambda:None)
    monkeypatch.setattr(runner,'stats',lambda *args:{})
    a=SimpleNamespace(output=str(tmp_path/'out'),root=str(tmp_path),audit=str(audit),selector=runner.SELECTOR_REV,seed=0)
    runner.run(a)
    assert read_record(tmp_path/'out/training-complete.json')['status']=='passed'
    for group in seen:assert set(group)<=set(train_ids) or group==dev_ids
    assert not read_record(tmp_path/'out/frozen-complete.json')['lora_nonzero']
    assert read_record(tmp_path/'out/lora-complete.json')['lora_nonzero']
    count=len(seen);runner.run(a);assert len(seen)==count
