import torch
from test_stage2_pilot import setup
from experiments.training_pilot.evaluate64 import candidates
from docprune.stage2.policy import allocate


def test_retrieval_control_uses_maxsim_not_query_vector_and_masks_are_frozen():
    ex,bank,_,_,_=setup()
    ex.retrieval.values.zero_()
    ex.retrieval.values[:,:,0]=torch.tensor([1.,4.,3.,2.])[:,None]
    ex.retrieval.values[:,:,-1]=torch.tensor([100.,-100.,-100.,100.])[:,None]
    branches={phase:{'dev':{'0.75':{'mask':[True,True,True,False]},'0.5':{'mask':[True,True,False,False]}}} for phase in ('untrained','frozen','lora')}
    first=candidates('dev',ex,branches);second=candidates('dev',ex,branches)
    assert first==second and len(first)==11
    expected,_=allocate(ex.retrieval.values[:,:,0].sum(1),ex.layout.costs,4)
    assert first['colqwen/0.5']==expected.tolist()
    randoms={k:v for k,v in first.items() if k.startswith('random/')}
    changed=candidates('other-dev',ex,{p:{'other-dev':v['dev']} for p,v in branches.items()})
    assert randoms!={k:v for k,v in changed.items() if k.startswith('random/')}
    assert all(len(z)==4 and all(isinstance(b,bool) for b in z) for z in first.values())
