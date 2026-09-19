"""Input-only ablation; parameter shapes and seeded initialization stay unchanged."""
from dataclasses import replace
import torch


def hidden_reader_inputs(module, args):
    memory, question, layout, budget, *_ = args
    sanitized = replace(layout, coordinates=torch.zeros_like(layout.coordinates),
                        metadata=torch.zeros_like(layout.metadata))
    return memory, question, sanitized, budget, None


def hidden_head2_inputs(module, args):
    (pooled,) = args
    return (torch.cat((pooled[..., :-1], torch.zeros_like(pooled[..., -1:])), dim=-1),)


def configure_scorer_inputs(model, hidden_state_only=False):
    if not hidden_state_only:
        return
    if getattr(model, '_hidden_input_hooks', None):
        raise ValueError('Hidden-state-only hooks already installed')
    if model.base.correction is None:
        raise ValueError('Matched hidden-state-only experiment requires Head 2')
    model._hidden_input_hooks = (
        model.base.reader.register_forward_pre_hook(hidden_reader_inputs),
        model.base.correction.readout.register_forward_pre_hook(hidden_head2_inputs),
    )
