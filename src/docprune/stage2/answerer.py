"""Frozen Qwen3 decoder-input deletion, complete-answer scoring and greedy decoding."""

import torch
from torch import nn
from torch.nn import functional as F

from .qwen import assemble_prefill, extract_vision


class FrozenAnswerer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(False)
        return self

    def vision(self, prompt, provenance):
        return extract_vision(self.model, prompt, provenance=provenance)

    @torch.no_grad()
    def prefill(self, prompt, memory, retained, *, cache=False):
        self.model.eval()
        packed, indices, next_position = assemble_prefill(
            self.model, prompt, memory, retained
        )
        out = self.model.model.language_model(
            **packed, use_cache=cache, return_dict=True
        )
        return (
            self.model.lm_head(out.last_hidden_state[:, -1]),
            out.past_key_values,
            packed,
            indices,
            next_position,
        )

    @torch.no_grad()
    def likelihood(self, prompt, memory, retained, target):
        """Mean continuation log likelihood; EOS appears only if explicitly in target."""
        self.model.eval()
        target = torch.as_tensor(
            target, dtype=torch.long, device=prompt.input_ids.device
        )
        if target.ndim != 1 or not target.numel():
            raise ValueError("A nonempty, already frozen target sequence is required")
        packed, _, start = assemble_prefill(self.model, prompt, memory, retained)
        prefix = packed["inputs_embeds"].shape[1]
        tail = target[:-1]
        embeddings = self.model.model.get_input_embeddings()(tail[None])
        packed["inputs_embeds"] = torch.cat(
            (packed["inputs_embeds"], embeddings), dim=1
        )
        positions = torch.arange(start, start + tail.numel(), device=target.device)[
            None, None
        ].expand(3, 1, -1)
        packed["position_ids"] = torch.cat((packed["position_ids"], positions), dim=-1)
        packed["visual_pos_masks"] = torch.cat(
            (
                packed["visual_pos_masks"],
                torch.zeros_like(tail[None], dtype=torch.bool),
            ),
            dim=1,
        )
        length = prefix + tail.numel()
        packed["attention_mask"] = torch.ones(
            (1, length), dtype=torch.long, device=target.device
        )
        packed["cache_position"] = torch.arange(length, device=target.device)
        output = self.model.model.language_model(
            **packed, use_cache=False, return_dict=True
        )
        logits = self.model.lm_head(output.last_hidden_state[:, prefix - 1 :]).float()
        ll = F.log_softmax(logits, dim=-1)[0].gather(1, target[:, None]).squeeze(1)
        return {"mean": float(ll.mean()), "token_loglikelihoods": ll.cpu().tolist()}

    @torch.no_grad()
    def likelihoods_shared_prefill(self, prompt, memory, retained, targets):
        """One masked context prefill, independently scored target continuations.

        The dynamic cache is cropped back to the unchanged prefix after each
        branch. Target continuations never see tokens from another answer.
        The original full-prefix likelihood() remains the parity reference.
        """
        if not targets:
            raise ValueError("At least one target is required")
        tensors = [torch.as_tensor(t, dtype=torch.long, device=prompt.input_ids.device) for t in targets]
        if any(t.ndim != 1 or not t.numel() for t in tensors):
            raise ValueError("Every target must be a nonempty frozen token sequence")
        logits, cache, _, indices, start = self.prefill(prompt, memory, retained, cache=True)
        prefix = indices.numel()
        first = F.log_softmax(logits[0].float(), dim=-1)
        results = []
        for target in tensors:
            tail = target[:-1]
            token_ll = first[target[:1]]
            try:
                if tail.numel():
                    out = self.model.model.language_model(
                        input_ids=tail[None], past_key_values=cache, use_cache=True,
                        attention_mask=torch.ones((1, prefix + tail.numel()), dtype=torch.long, device=target.device),
                        position_ids=torch.arange(start, start + tail.numel(), device=target.device)[None,None].expand(3,1,-1),
                        cache_position=torch.arange(prefix, prefix + tail.numel(), device=target.device),
                        return_dict=True,
                    )
                    values = F.log_softmax(self.model.lm_head(out.last_hidden_state).float(), dim=-1)[0]
                    token_ll = torch.cat((token_ll, values.gather(1, target[1:,None]).squeeze(1)))
                results.append({"mean": float(token_ll.mean()), "token_loglikelihoods": token_ll.cpu().tolist()})
            finally:
                cache.crop(prefix)
        return results

    @torch.no_grad()
    def teacher_scores(
        self, prompt, memory, retained, accepted_targets, fixed_self_target, *, reuse_prefill=True
    ):
        if not accepted_targets:
            raise ValueError(
                "Accepted targets must contain complete answer alternatives"
            )
        targets = list(accepted_targets) + [fixed_self_target]
        results = (self.likelihoods_shared_prefill(prompt, memory, retained, targets)
                   if reuse_prefill else [self.likelihood(prompt, memory, retained, t) for t in targets])
        gold, own = results[:-1], results[-1]
        g, s = max(x["mean"] for x in gold), own["mean"]
        return {"G": g, "S": s, "C": g - s, "accepted": gold, "fixed_self": own}

    @torch.no_grad()
    def generate(
        self,
        prompt,
        memory,
        retained,
        *,
        max_new_tokens,
        eos_ids,
        repetition_penalty=1.0,
    ):
        if max_new_tokens < 1 or repetition_penalty <= 0 or not eos_ids:
            raise ValueError("Explicit decoding contract is required")
        logits, cache, packed, indices, start = self.prefill(
            prompt, memory, retained, cache=True
        )
        tokens = []
        seen = prompt.input_ids[0, indices].tolist()
        prefill_length = indices.numel()
        stopped = False
        for step in range(max_new_tokens):
            scores = logits[0].float().clone()
            if repetition_penalty != 1.0:
                ix = torch.tensor(sorted(set(seen)), device=scores.device)
                values = scores[ix]
                scores[ix] = torch.where(
                    values < 0, values * repetition_penalty, values / repetition_penalty
                )
            token = int(scores.argmax())
            tokens.append(token)
            seen.append(token)
            if token in eos_ids:
                stopped = True
                break
            if step + 1 == max_new_tokens:
                break
            one = torch.tensor([[token]], device=prompt.input_ids.device)
            length = prefill_length + step + 1
            output = self.model.model.language_model(
                input_ids=one,
                past_key_values=cache,
                use_cache=True,
                attention_mask=torch.ones(
                    (1, length), dtype=torch.long, device=one.device
                ),
                position_ids=torch.full(
                    (3, 1, 1), start + step, dtype=torch.long, device=one.device
                ),
                cache_position=torch.tensor([length - 1], device=one.device),
                return_dict=True,
            )
            cache = output.past_key_values
            logits = self.model.lm_head(output.last_hidden_state[:, -1])
        return {
            "token_ids": tokens,
            "stopped_on_eos": stopped,
            "truncated": not stopped,
            "retained_original_tokens": retained.cpu().tolist(),
            "retained_sequence_length": prefill_length,
        }
