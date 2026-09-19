from dataclasses import replace
from types import SimpleNamespace
import torch
from test_stage2_contracts import example
from docprune.stage2.readouts import RegionReader
from docprune.stage2.policy import SetCorrection
from experiments.training_pilot.scorer_inputs import configure_scorer_inputs


def test_hidden_only_ignores_engineered_inputs_preserving_parameters_and_costs():
    torch.set_num_threads(1);torch.manual_seed(0)
    reader=RegionReader(32,32,8,width=16,heads=2,slots=2,retrieval_dim=3)
    head2=SetCorrection(16)
    before={k:v.clone() for k,v in reader.state_dict().items()}
    rng=torch.get_rng_state().clone()
    model=SimpleNamespace(base=SimpleNamespace(reader=reader,correction=head2))
    configure_scorer_inputs(model,True)
    assert torch.equal(rng,torch.get_rng_state())
    for k,v in reader.state_dict().items():torch.testing.assert_close(v,before[k],rtol=0,atol=0)
    layout=example().layout; costs=layout.costs.clone()
    memory=torch.randn(8,32,requires_grad=True); question=torch.randn(3,32)
    changed=replace(layout,metadata=layout.metadata+100,coordinates=layout.coordinates+20)
    a=reader(memory,question,layout,1.,object())
    b=reader(memory,question,changed,1.,object())
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    assert not torch.equal(a,reader(memory+torch.randn_like(memory),question,layout,1.,object()))
    assert not torch.equal(a,reader(memory,question,layout,.5,object()))
    a.sum().backward()
    assert memory.grad is not None
    torch.testing.assert_close(layout.costs,costs)
    pooled=torch.randn(2,33); altered=pooled.clone();altered[:,-1]+=50
    torch.testing.assert_close(head2.readout(pooled),head2.readout(altered),rtol=0,atol=0)
