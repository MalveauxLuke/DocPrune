import sys
from pathlib import Path
import numpy as np
import torch
from test_omp10 import tiny_model,prompt
from docprune.stage2.answerer import FrozenAnswerer
from docprune.stage2.qwen import assemble_prefill
sys.path.insert(0,str(Path(__file__).parents[1]/'experiments/omp10'))
from profile_padding import padded_score

def test_diagnostic_padding_preserves_scores():
    torch.set_num_threads(1);torch.manual_seed(19)
    model=tiny_model();p=prompt();reader=FrozenAnswerer(model);mem=reader.vision(p,'diagnostic')
    a=torch.arange(8);b=torch.arange(2);g=[7,8,9];s=[11,12]
    refs=[reader.teacher_scores(p,mem,r,[g],s,reuse_prefill=False) for r in [a,b]]
    width=assemble_prefill(model,p,mem,a)[0]['inputs_embeds'].shape[1]
    for retained,expected,pad in [([a,a],[refs[0],refs[0]],0),([a,b],refs,0),([b],[refs[1]],width)]:
        actual=padded_score(reader,p,mem,retained,g,s,pad_to=pad)
        for x,y in zip(actual,expected):np.testing.assert_allclose([x['G'],x['S']],[y['G'],y['S']],atol=1e-6)


def test_varlen_boundaries_and_visual_positions_cpu():
    from varlen_diagnostic import unpadded_attention
    from torch.nn import functional as F
    def cpu_kernel(q,k,v,cu,maxlen,scale):
        rows=[]
        for lo,hi in zip(cu[:-1].tolist(),cu[1:].tolist()):
            args=[x[lo:hi].transpose(0,1)[None] for x in [q,k,v]]
            rows.append(F.scaled_dot_product_attention(*args,is_causal=True,scale=scale,enable_gqa=True)[0].transpose(0,1))
        return torch.cat(rows)
    torch.set_num_threads(1);torch.manual_seed(20)
    model=tiny_model();p=prompt();reader=FrozenAnswerer(model);mem=reader.vision(p,'varlen')
    kept=[torch.arange(8),torch.arange(2)];g=[7,8,9];s=[11,12]
    refs=[reader.teacher_scores(p,mem,r,[g],s,reuse_prefill=False) for r in kept]
    old=model.model.language_model.config._attn_implementation
    with unpadded_attention(model,cpu_kernel):
        actual=padded_score(reader,p,mem,kept,g,s)
    assert model.model.language_model.config._attn_implementation==old
    for a,b in zip(actual,refs):np.testing.assert_allclose([a['G'],a['S']],[b['G'],b['S']],atol=1e-6)
