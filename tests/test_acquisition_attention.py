import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn.functional as F
from docprune.acquisition_attention import segmented_sdpa, install_segmented_vision_sdpa


class AttentionTests(unittest.TestCase):
    def test_matches_dense_boundaries_and_uses_bounded_rank_four_calls(self):
        torch.manual_seed(17)
        q, k, v = [torch.randn(13, 2, 8) for _ in range(3)]
        cu = torch.tensor([0, 3, 8, 11, 13], dtype=torch.int32)
        mask = torch.zeros(13, 13, dtype=torch.bool)
        for a, b in zip(cu[:-1], cu[1:]):
            mask[a:b, a:b] = True
        expected = F.scaled_dot_product_attention(q.transpose(0, 1), k.transpose(0, 1),
                                                  v.transpose(0, 1), mask).transpose(0, 1)
        original = F.scaled_dot_product_attention
        shapes = []
        def observed(q, k, v, **kwargs):
            shapes.append(tuple(q.shape))
            self.assertNotIn('attn_mask', kwargs)
            return original(q, k, v, **kwargs)
        with patch('docprune.acquisition_attention.F.scaled_dot_product_attention', side_effect=observed):
            actual = segmented_sdpa(q, k, v, cu)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        self.assertEqual(shapes, [(2, 2, 3, 8), (1, 2, 5, 8), (1, 2, 2, 8)])
        changed = v.clone(); changed[3:8] += 100
        result = segmented_sdpa(q, k, changed, cu)
        torch.testing.assert_close(result[:3], actual[:3])
        torch.testing.assert_close(result[8:], actual[8:])
        for bad in ([0, 0, 13], [1, 13], [0, 12]):
            with self.assertRaises(ValueError):
                segmented_sdpa(q, k, v, torch.tensor(bad))

    def test_pinned_module_parity_and_instance_isolation(self):
        from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionSdpaAttention
        torch.manual_seed(23)
        original = Qwen2_5_VLVisionSdpaAttention(32, num_heads=4).eval()
        import copy
        modified = copy.deepcopy(original)
        visual = SimpleNamespace(blocks=[SimpleNamespace(attn=modified)])
        install_segmented_vision_sdpa(visual)
        x = torch.randn(13, 32)
        cu = torch.tensor([0, 3, 8, 13], dtype=torch.int32)
        angles = torch.randn(13, 8)
        with torch.inference_mode():
            expected = original(x, cu, position_embeddings=(angles.cos(), angles.sin()))
            actual = modified(x, cu, position_embeddings=(angles.cos(), angles.sin()))
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        self.assertNotIn('forward', original.__dict__)


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
