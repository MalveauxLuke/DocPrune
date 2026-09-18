from dataclasses import replace
import numpy as np
import pytest
import torch
from test_stage2_contracts import example
from test_stage2_models import tiny_model,prompt
from docprune.stage2.contracts import RetrievalFeatures
from docprune.stage2.experiment import ExperimentConfig,Stage1Contract
from docprune.stage2.training import build_selector
from docprune.stage2.supervision import TeacherBank,Outcome,question_loss
from docprune.stage2.pilot import next_probe,fit_omp,CachedPilotSelector,RETRIEVAL_SCHEMA,phase_optimizer,train_epoch


def test_deterministic_variable_mask_acquisition_and_frozen_predictions():
    rows=[];costs=[1,2,3,5,7,11,13,17]
    for i in range(32):
        p=next_probe(rows,costs,'q',True,dict(g=None,s=-1.))
        assert p==next_probe(rows,costs,'q',True,dict(g=None,s=-1.))
        s=-2.+float(np.asarray(p['mask'])@np.array([1,-.5,.3,0,0,.1,0,0]))
        rows.append(dict(mask=p['mask'],g=None,s=s))
        if i in (22,23):assert p['kind']=='fresh_check' and p['omp']['channels']==1
    assert len({tuple(r['mask']) for r in rows})==32
    assert len({np.array(r['mask'])@costs for r in rows})>1
    assert next_probe(rows,costs,'q',True,dict(g=None,s=-1.)) is None


def test_small_mask_space_exhaustion_and_omp_channels():
    rows=[]
    while (p:=next_probe(rows,[1,10],'small',True,dict(g=None,s=0.))) is not None:
        rows.append(dict(mask=p['mask'],g=None,s=float(sum(p['mask']))))
    assert len(rows)==4 and {tuple(r['mask']) for r in rows}=={(0,0),(0,1),(1,0),(1,1)}
    fit=fit_omp([r['mask'] for r in rows],[r['s'] for r in rows])
    np.testing.assert_allclose(fit['coefficients'],[[1],[1]],atol=1e-6)


def setup():
    torch.manual_seed(0);torch.set_num_threads(1)
    x=example();x.budget=8;x.retrieval=RetrievalFeatures(torch.randn(4,3,129),RETRIEVAL_SCHEMA)
    masks=torch.tensor([[0,0,0,0],[1,0,0,0],[1,1,0,0],[1,1,1,1]],dtype=torch.bool)
    bank=TeacherBank(x.identity.key,masks,tuple(Outcome(None,s) for s in (-4.,-2.,-1.,0.)),Outcome(None,0.),True,True,'doc','s','sealed','variable_pilot_v1','hamming_families_v1')
    cfg=ExperimentConfig(vision_mode='native',head2=True,retrieval_dim=129,retrieval_schema=RETRIEVAL_SCHEMA,width=16,heads=2,slots=2,stage1=Stage1Contract(epsilon=.1,margin=.05))
    backbone=tiny_model();model=CachedPilotSelector(build_selector(cfg,answerer_config=backbone.config,selector_model=backbone))
    batch=dict(examples=[x],banks=[bank],proxy_prompts=[prompt()],native_layouts=[x.layout])
    return x,bank,cfg,model,batch


def test_variable_validation_family_means_and_old_contract():
    x,b,_,_,_=setup();b.validate(x)
    with pytest.raises(ValueError):replace(b,retention_mode='exact').validate(x)
    with pytest.raises(ValueError):b.validate(replace(x,budget=4))
    scores=torch.tensor([.2,.3,.4,.5],requires_grad=True)
    pairs=b.pairs('gold_aware',.1,.05);vals=b.masks.float()@scores
    groups={1:[],2:[],3:[]}
    for i,j in pairs:
        groups[min(3,int((b.masks[i]!=b.masks[j]).sum()))].append(torch.nn.functional.softplus(-(vals[i]-vals[j])))
    expected=torch.stack([torch.stack(g).mean() for g in groups.values() if g]).mean()
    actual=question_loss(scores,b,mode='gold_aware',epsilon=.1,margin=.05)
    torch.testing.assert_close(actual,expected)


def test_frozen_cache_parity_then_lora_invalidation_and_gradients():
    x,b,cfg,m,batch=setup();opt=phase_optimizer(m,'warmup');m.eval()
    p=batch['proxy_prompts'][0]
    first=m(x,p).detach();again=m(x,p).detach()
    torch.testing.assert_close(first,again,atol=0,rtol=0)
    assert m.calls['language']==1 and m.calls['hidden_hits']==1
    initial={n:v.detach().clone() for n,v in m.named_parameters() if 'lora_' in n}
    train_epoch(m,[batch,batch],cfg,opt)
    assert all(torch.equal(initial[n],v) for n,v in m.named_parameters() if n in initial)
    assert m.calls['language']==1
    opt=phase_optimizer(m,'lora');assert not m.hidden_cache
    train_epoch(m,[batch,batch],cfg,opt)
    assert any(not torch.equal(initial[n],v) for n,v in m.named_parameters() if n in initial)
    assert m.calls['language']==3 and m.calls['vision']==1


def test_wrong_case_acquisition_uses_two_channels_and_dev_ignores_outcomes():
    rows=[];costs=list(range(1,10))
    for i in range(32):
        p=next_probe(rows,costs,'wrong',False,dict(g=-1.,s=-.1))
        assert p is not None
        z=np.asarray(p['mask'])
        rows.append(dict(mask=p['mask'],g=float(-3+z[0]+z[1]),s=float(-2+z[2]-.7*z[0])))
        if i==22:assert p['omp']['channels']==2
    assert len({tuple(r['mask']) for r in rows})==32
    a=next_probe(rows[:5],costs,'dev',False,dict(g=-1.,s=0.),count=8,random_only=True)
    changed=[dict(r,g=100.,s=-100.) for r in rows[:5]]
    assert a==next_probe(changed,costs,'dev',False,dict(g=999.,s=999.),count=8,random_only=True)
