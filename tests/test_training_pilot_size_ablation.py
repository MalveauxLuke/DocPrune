from dataclasses import replace

import pytest
import torch
from test_stage2_contracts import example
from experiments.training_pilot.train64 import regional_metadata_ablation


def test_only_count_feature_changes_and_source_costs_remain_intact():
    layout = example().layout
    metadata = torch.arange(len(layout.region_ids)*8, dtype=torch.float32).reshape(-1,8)
    metadata[:, -1] = layout.costs
    layout = replace(layout, metadata=metadata)
    original = metadata.clone()
    result = regional_metadata_ablation(layout, True)
    torch.testing.assert_close(result.metadata[:, :-1], original[:, :-1])
    assert torch.count_nonzero(result.metadata[:, -1]) == 0
    torch.testing.assert_close(layout.metadata, original)
    torch.testing.assert_close(result.costs, layout.costs)
    assert result.owner is layout.owner
    assert regional_metadata_ablation(layout, False) is layout
    with pytest.raises(ValueError, match='schema mismatch'):
        regional_metadata_ablation(replace(layout,metadata=metadata+1), True)


def test_shared_cache_reads_control_and_refuses_missing_hidden_states(tmp_path):
    from docprune.stage2.pilot_runtime import TensorCache
    from docprune.stage2.storage import publish
    from experiments.training_pilot.shared_cache import attach_control_cache
    source = dict(audit_sha256='audit',config={'stage1':{'selector_revision':'rev'}},
                  prompt={'condition':'evidence-v1'},runtime={'precision':'bf16'},
                  sources={'src/docprune/stage2/qwen.py':'same','experiments/training_pilot/train64.py':'old'})
    publish(tmp_path/'control/contract.json',source)
    identity = dict(selector='rev',sources=source['sources'],runtime=source['runtime'],audit='audit',processor={})
    donor = TensorCache(tmp_path/'control/cache',identity)
    donor.put('identity-language','question',{'visual':torch.ones(2,3)})
    current = dict(source,sources=dict(source['sources'],**{'experiments/training_pilot/train64.py':'new'}))
    cache = TensorCache(tmp_path/'new/cache',{'run':'new'})
    attach_control_cache(cache,tmp_path/'control',current,{})
    assert cache.contains('identity-language','question')
    torch.testing.assert_close(cache.get('identity-language','question')['visual'],torch.ones(2,3))
    assert cache.control_hits == 1 and cache.written_bytes == 0
    with pytest.raises(ValueError,match='refusing recomputation'):
        cache.get('identity-language','absent')
    bad = dict(current,prompt={'condition':'different'})
    with pytest.raises(ValueError,match='prompt'):
        attach_control_cache(cache,tmp_path/'control',bad,{})
