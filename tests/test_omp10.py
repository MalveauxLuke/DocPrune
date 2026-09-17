import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from docprune.stage2.qwen import PackedPrompt

def tiny_model(width=32):
    c = Qwen3VLConfig(
        text_config=dict(
            vocab_size=80,
            hidden_size=width,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=width // 8,
            num_key_value_heads=1,
            head_dim=8,
            rope_scaling={
                "rope_type": "default",
                "mrope_section": [1, 1, 2],
                "mrope_interleaved": True,
            },
        ),
        vision_config=dict(
            depth=2,
            hidden_size=32,
            intermediate_size=64,
            num_heads=4,
            out_hidden_size=width,
            patch_size=2,
            temporal_patch_size=1,
            spatial_merge_size=2,
            num_position_embeddings=16,
            deepstack_visual_indexes=[0, 1],
        ),
        image_token_id=70,
        video_token_id=71,
        vision_start_token_id=72,
        vision_end_token_id=73,
    )
    return Qwen3VLForConditionalGeneration(c).cpu().eval()

def prompt():
    return PackedPrompt(
        torch.tensor([[1, 2, 72, 70, 70, 70, 70, 73, 72, 70, 70, 70, 70, 73, 3]]),
        torch.tensor([0, 1]),
        torch.tensor([[1, 4, 4], [1, 4, 4]]),
        torch.randn(32, 12),
    )

spec=importlib.util.spec_from_file_location('discovery',Path(__file__).parents[1]/'experiments/omp10/discovery.py')
m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)


def test_independent_masks_are_reproducible_and_not_exact_budget():
    ids=[str(i) for i in range(15)]
    x=m.masks_for('q',ids)
    np.testing.assert_array_equal(x,m.masks_for('q',ids))
    assert x.shape == (22,15)
    assert np.unique(x@np.arange(1,16)).size > 1
    assert not np.array_equal(x,m.masks_for('other',ids))


def test_joint_fit_recovers_known_signed_sparse_signal():
    x=m.masks_for('known',[str(i) for i in range(8)])
    b=np.zeros((8,2)); b[2]=[3,-2]; b[5]=[-1,4]
    y=x@b+[-5,-7]
    f=m.fit_omp(x,y)
    assert set(f['support']) == {2,5}
    np.testing.assert_allclose(f['coefficients'],b,atol=1e-10)
    np.testing.assert_allclose(f['intercept'],[-5,-7],atol=1e-10)


def test_no_signal_does_not_invent_supported_region():
    x=m.masks_for('quiet',[str(i) for i in range(8)])
    x[:,0]=1
    f=m.fit_omp(x,np.tile([-3.,-4.],(22,1)))
    assert f['support'] == [] and 0 in f['constant_columns']
    assert 0 not in f['shortlist']


def test_invalid_and_incomplete_bank_rejected():
    with pytest.raises(ValueError):m.fit_omp(np.ones((21,3)),np.ones((21,2)))
    with pytest.raises(ValueError):m.fit_omp(np.ones((22,3)),np.full((22,2),np.nan))


def test_shared_prefix_scores_match_independent_and_do_not_leak_targets():
    import torch
    from docprune.stage2.answerer import FrozenAnswerer
    torch.set_num_threads(1)
    torch.manual_seed(17)
    model=tiny_model(); p=prompt(); reader=FrozenAnswerer(model)
    memory=reader.vision(p,'tiny-smoke-contract')
    retained=torch.tensor([0,2,5,7]); targets=[[7,8,9],[11,12],[7,8,9]]
    actual=reader.likelihoods_shared_prefill(p,memory,retained,targets)
    expected=[reader.likelihood(p,memory,retained,t) for t in targets]
    np.testing.assert_allclose([x['mean'] for x in actual],[x['mean'] for x in expected],atol=1e-6)
    assert actual[0]==actual[2]


@pytest.mark.parametrize("lengths", [[0,2,8],[1,7],[8,8],[2,2,2,2]])
def test_padded_mask_batch_matches_independent_full_prefix(lengths):
    import sys
    sys.path.insert(0,str(Path(__file__).parents[1]/'experiments/omp10'))
    from batching import score_batch
    from docprune.stage2.answerer import FrozenAnswerer
    torch.set_num_threads(1);torch.manual_seed(51)
    model=tiny_model();p=prompt();reader=FrozenAnswerer(model)
    memory=reader.vision(p,'batch-test')
    kept=[torch.arange(n) for n in lengths]
    actual=score_batch(reader,p,memory,kept,[7,8,9,10],[11,12])
    reference=[reader.teacher_scores(p,memory,r,[[7,8,9,10]],[11,12],reuse_prefill=False) for r in kept]
    for a,b in zip(actual,reference):
        np.testing.assert_allclose([a['G'],a['S']],[b['G'],b['S']],atol=1e-6)
