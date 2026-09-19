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
