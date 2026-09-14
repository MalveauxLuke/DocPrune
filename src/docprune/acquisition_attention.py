"""Bounded Qwen2.5 vision SDPA for the acquisition reader only.

Retain existing cu_seqlens boundaries; batch equal-sized independent windows.
The installed Transformers package and decoder attention are left untouched.
"""
from contextlib import nullcontext
from types import MethodType

import torch
import torch.nn.functional as F


def segmented_sdpa(q, k, v, cu_seqlens):
    """[tokens, heads, width] -> same shape, without a global square mask."""
    boundaries = cu_seqlens.detach().cpu().tolist()
    if (len(boundaries) < 2 or boundaries[0] != 0 or boundaries[-1] != q.shape[0]
            or any(b <= a for a, b in zip(boundaries, boundaries[1:]))):
        raise ValueError('Invalid vision attention boundaries')
    if q.shape != k.shape or q.shape != v.shape or q.ndim != 3:
        raise ValueError('Vision q/k/v must have matching rank-three shapes')
    groups = {}
    for index, (start, end) in enumerate(zip(boundaries, boundaries[1:])):
        groups.setdefault(end-start, []).append((index, start, end))
    outputs = [None] * (len(boundaries)-1)
    context = nullcontext()
    if q.is_cuda:
        from torch.nn.attention import SDPBackend, sdpa_kernel
        # Fail explicitly if neither efficient kernel supports this GPU/input.
        context = sdpa_kernel([SDPBackend.FLASH_ATTENTION, SDPBackend.EFFICIENT_ATTENTION])
    with context:
        for entries in groups.values():
            packed = [torch.stack([t[a:b].transpose(0, 1) for _, a, b in entries])
                      for t in (q, k, v)]
            result = F.scaled_dot_product_attention(*packed, dropout_p=0.0, is_causal=False)
            for row, (index, _, _) in enumerate(entries):
                outputs[index] = result[row].transpose(0, 1)
    return torch.cat(outputs, dim=0)


def _vision_forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None,
                    position_embeddings=None):
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import apply_rotary_pos_emb_vision
    length = hidden_states.shape[0]
    q, k, v = self.qkv(hidden_states).reshape(length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    if position_embeddings is None:
        if rotary_pos_emb is None:
            raise ValueError('Missing vision rotary positions')
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos().float(), emb.sin().float())
    q, k = apply_rotary_pos_emb_vision(q, k, *position_embeddings)
    return self.proj(segmented_sdpa(q, k, v, cu_seqlens).reshape(length, -1))


def install_segmented_vision_sdpa(visual):
    """Patch only this reader instance, including its stock generation path."""
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLVisionSdpaAttention
    attentions = [block.attn for block in visual.blocks]
    if not attentions or any(type(attn) is not Qwen2_5_VLVisionSdpaAttention for attn in attentions):
        raise TypeError('Expected pinned Qwen2.5 vision SDPA modules')
    for attn in attentions:
        attn.forward = MethodType(_vision_forward, attn)
    return {'implementation': 'segmented_sdpa_v1', 'layers': len(attentions),
            'cuda_backends': ['flash_attention', 'efficient_attention'],
            'dense_math_fallback': False}
