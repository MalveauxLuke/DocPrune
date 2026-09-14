"""Persistent frozen Qwen B_input teacher adapter. Torch imports are lazy."""
from __future__ import annotations
import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import time
import numpy as np
from .acquisition_io import sha, token_identity, tree_identity
from .adaptive_acquisition import digest, scores


def environment_report(runtime, snapshot):
    versions = {}
    for package in runtime['expected_versions']:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    snapshot = Path(snapshot)
    mismatches = {k: {'expected': v, 'actual': versions[k]} for k, v in runtime['expected_versions'].items()
                  if versions[k] is None or versions[k].split('+')[0] != v}
    files = ['config.json', 'preprocessor_config.json', 'tokenizer_config.json',
             'tokenizer.json', 'generation_config.json', 'model.safetensors.index.json']
    identities = {name: sha(snapshot/name) if (snapshot/name).is_file() else None for name in files}
    return {'python': platform.python_version(), 'versions': versions, 'version_mismatches': mismatches,
            'snapshot_name_matches_revision': snapshot.name == runtime['execution_qwen']['revision'],
            'model_metadata_sha256': identities, 'snapshot': str(snapshot),
            'code_sha256': tree_identity(), 'synthetic': False}


def validate_contract(reader, *, prompt_sha256, input_ids_sha256, grids, runtime):
    if prompt_sha256 != reader['prompt_sha256'] or input_ids_sha256 != reader['input_ids_sha256']:
        raise ValueError('Frozen assistant prompt or prefill token identity mismatch')
    if grids != [runtime['processor']['grid_thw']]*4:
        raise ValueError('Frozen page order/grid contract mismatch')


def validate_parity(actual, expected, *, tolerance=1e-4):
    a, b = np.asarray(actual, dtype=float), np.asarray(expected, dtype=float)
    if a.shape != b.shape or a.ndim != 1 or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Invalid parity likelihood vectors')
    error = float(np.max(np.abs(a-b)))
    if error > tolerance:
        raise ValueError(f'Teacher likelihood parity failed: max absolute difference {error} > {tolerance}')
    return error


def validate_generation(ids, eos, reader):
    if list(ids) != reader['expected_generated_ids'] or eos != reader['expected_terminal_eos']:
        raise ValueError('All-keep generated answer or terminal EOS changed')


def validate_retained(ids, population, budget):
    if len(ids) != budget or ids != sorted(set(ids)) or any(type(i) is not int or not 0 <= i < population for i in ids):
        raise ValueError('Intervention must contain sorted unique original visual IDs at the exact budget')


def validate_trace(trace, ids, population):
    expected = {'boundary': 'B_input', 'mode': 'physical_delete', 'visual_population': population,
                'requested_budget': len(ids), 'achieved_budget': len(ids), 'retained_visual_ids': list(ids)}
    if any(trace.get(k) != v for k, v in expected.items()):
        raise ValueError('Reader intervention trace violates the original-token deletion contract')


class QwenTeacher:
    """One model instance shared across sequential questions, loaded offline only."""
    def __init__(self, runtime, snapshot):
        self.runtime = runtime
        self.identity = environment_report(runtime, snapshot)
        if self.identity['version_mismatches']:
            raise RuntimeError(f"Runtime differs from the frozen reader: {self.identity['version_mismatches']}")
        if not self.identity['snapshot_name_matches_revision'] or any(v is None for v in self.identity['model_metadata_sha256'].values()):
            raise RuntimeError('Missing or mismatched pinned local model snapshot')
        if platform.python_version_tuple()[:2] != ('3', '10'):
            raise RuntimeError('Frozen H200 runtime requires Python 3.10')
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise RuntimeError('Expose exactly one authorized GPU through CUDA_VISIBLE_DEVICES')
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            str(snapshot), torch_dtype=torch.bfloat16, low_cpu_mem_usage=True,
            attn_implementation='sdpa', local_files_only=True).to('cuda:0').eval()
        self.model.config.vision_config.torch_dtype = torch.bfloat16
        self.processor = AutoProcessor.from_pretrained(str(snapshot), local_files_only=True, use_fast=False)
        self.model.generation_config.repetition_penalty = 1.05
        eos = self.model.generation_config.eos_token_id
        if list(eos if isinstance(eos, (tuple, list)) else [eos]) != runtime['eos_ids']:
            raise ValueError('Frozen EOS contract changed')
        if self.model.config._attn_implementation != 'sdpa':
            raise ValueError('Reader must use the frozen SDPA attention implementation')
        self.identity['gpu_name'] = torch.cuda.get_device_name(0)
        self.identity['cuda_runtime'] = torch.version.cuda
        self.identity['physical_device'] = os.environ.get('CUDA_VISIBLE_DEVICES')

    def question(self, case, package):
        return QuestionCheckpoint(self, case, package)


class QuestionCheckpoint:
    """Compose the existing sparse-reader primitives, reusing full vision once.

    The original batched scorer is retained as an independent parity path. The
    existing resume helper clones the checkpoint cache for every intervention.
    """
    def __init__(self, teacher, case, package):
        import torch
        from PIL import Image
        from docprune.answerers import (_prepare_batch_with_prompt, _grid, _validate_prepared_batch,
                                       _move_batch, _input_ids_identity, _validate_placeholder_count)
        from docprune.qwen2vl.preprocessing import prepare_qwen_page, prepared_raster_image
        from docprune.qwen2vl.model import DocPruneQwen2VL, VisionPruningMasks
        from docprune.qwen2vl.sequence import compact_multimodal_sequence
        from docprune.qwen2vl.vision import compact_vision_batch
        from docprune.qwen2vl.decoder import capture_forced_boundary_checkpoint
        torch.cuda.reset_peak_memory_stats()
        self.teacher, self.case = teacher, case
        self.targets = tuple(tuple(t) for t in case['reader']['targets'])
        self.population = len(case['public']['owners'])
        self.budget = case['public']['budget']
        self.guarded = False
        self.model = teacher.model
        self.adapter = DocPruneQwen2VL(self.model)
        images = []
        for item in case['reader']['images']:
            path = Path(package)/item['path']
            if sha(path) != item['sha256']:
                raise ValueError('Page bytes changed before reader preparation')
            with Image.open(path) as image:
                images.append(image.convert('RGB'))
        started = time.perf_counter()
        prepared = [prepare_qwen_page(teacher.processor, image) for image in images]
        prompt, batch = _prepare_batch_with_prompt(teacher.processor, [prepared_raster_image(p) for p in prepared], case['reader']['question'])
        _validate_prepared_batch(prepared, batch, self.model, teacher.processor)
        grid = _grid(batch)
        shape, ids_hash = _input_ids_identity(batch['input_ids'])
        validate_contract(case['reader'], prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),
                          input_ids_sha256=ids_hash, grids=grid.tolist(), runtime=teacher.runtime)
        _validate_placeholder_count(self.model, teacher.processor, batch['input_ids'], grid)
        self.batch = _move_batch(batch, 'cuda:0')
        self.grid = grid.to('cuda:0')
        keep = torch.ones(self.population, dtype=torch.bool, device='cuda:0')
        self.pruning_masks = VisionPruningMasks(keep, keep)
        with torch.inference_mode():
            positions, _ = self.model.get_rope_index(self.batch['input_ids'], image_grid_thw=self.grid,
                                                    attention_mask=self.batch['attention_mask'])
            vision = compact_vision_batch(self.model.visual,
                       self.batch['pixel_values'].to(device=self.model.visual.get_device(), dtype=self.model.visual.get_dtype()),
                       self.grid, keep)
            compact = compact_multimodal_sequence(input_ids=self.batch['input_ids'],
                       attention_mask=self.batch['attention_mask'], position_ids=positions,
                       image_token_id=self.model.config.image_token_id, group_keep_mask=keep)
            if compact.visual_indices.numel() != self.population or vision.image_embeds.shape[0] != self.population:
                raise ValueError('Unexpected full-context visual population')
            embeddings = self.model.model.embed_tokens(compact.input_ids).clone()
            embeddings[0, compact.visual_indices] = vision.image_embeds.to(device=embeddings.device, dtype=embeddings.dtype)
            self.checkpoint = capture_forced_boundary_checkpoint(self.model.model, embeddings, compact.position_ids,
                                    visual_indices=compact.visual_indices, boundary='input')
        torch.cuda.synchronize()
        self.preparation_seconds = time.perf_counter()-started
        self._guard()

    def _measure(self, ids):
        import torch
        from docprune.qwen2vl.decoder import ForcedVisualIntervention, resume_forced_boundary_checkpoint
        from docprune.qwen2vl.model import teacher_forced_sequence_loglikelihoods
        started = time.perf_counter()
        intervention = ForcedVisualIntervention(boundary='input', mode='physical_delete', retained_visual_ids=tuple(ids))
        with torch.inference_mode():
            prefill = resume_forced_boundary_checkpoint(self.model.model, self.checkpoint, intervention)
            likelihoods = teacher_forced_sequence_loglikelihoods(self.model.model, self.model.lm_head, prefill, self.targets)
        torch.cuda.synchronize()
        if prefill.forced is None:
            raise RuntimeError('Reader lost its forced intervention trace')
        trace = prefill.forced.to_dict()
        validate_trace(trace, ids, self.population)
        return {'likelihoods': list(likelihoods), 'forced_intervention': trace,
                'decoder_seconds': time.perf_counter()-started,
                'question_peak_allocated_gpu_bytes': int(torch.cuda.max_memory_allocated()), 'synthetic': False}

    def _legacy_measure(self, ids):
        import torch
        from docprune.qwen2vl.decoder import ForcedVisualIntervention
        with torch.inference_mode():
            result = self.adapter.score_forced_intervention_likelihoods(
                input_ids=self.batch['input_ids'], attention_mask=self.batch['attention_mask'],
                pixel_values=self.batch['pixel_values'], image_grid_thw=self.grid, pruning_masks=self.pruning_masks,
                forced_interventions=(ForcedVisualIntervention(boundary='input', mode='physical_delete', retained_visual_ids=tuple(ids)),),
                teacher_forced_target_token_ids=self.targets)
        if result.original_visual_tokens != self.population or result.post_btp_visual_tokens != self.population or result.post_qtp_visual_tokens != self.population:
            raise ValueError('Parity path unexpectedly pruned before the intervention')
        return list(result.branches[0].teacher_forced_loglikelihoods)

    def _guard(self):
        import torch
        started = time.perf_counter()
        all_ids = list(range(self.population))
        actual = self._measure(all_ids)['likelihoods']
        historical_error = validate_parity(actual, self.case['reader']['reference_likelihoods'])
        # One masked comparison tests deletion and the persistent-checkpoint path.
        _, selected = token_identity(self.case['public'], self.case['public']['banks']['A'][0])
        selected_actual = self._measure(selected)['likelihoods']
        adapter_error = validate_parity(selected_actual, self._legacy_measure(selected))
        with torch.inference_mode():
            generated = self.model.generate(**self.batch, max_new_tokens=self.teacher.runtime['max_new_tokens'],
                 do_sample=False, num_beams=1, eos_token_id=self.teacher.runtime['eos_ids'], repetition_penalty=1.05)
        response = generated[0, self.batch['input_ids'].shape[1]:].tolist()
        eos = response.pop() if response and response[-1] in self.teacher.runtime['eos_ids'] else None
        validate_generation(response, eos, self.case['reader'])
        torch.cuda.synchronize()
        self.guard = {'historical_all_keep_error': historical_error, 'persistent_vs_legacy_mask_error': adapter_error,
                      'all_keep_generated_ids_match': True, 'reference_targets': len(self.targets),
                      'preparation_seconds': self.preparation_seconds, 'guard_seconds': time.perf_counter()-started,
                      'extra_likelihood_evaluations': 3, 'extra_generations': 1,
                      'boundary': 'input', 'mode': 'physical_delete'}
        self.guarded = True

    def score(self, retained_ids):
        if not self.guarded:
            raise RuntimeError('Reader has not passed its reference and adapter parity checks')
        validate_retained(retained_ids, self.population, self.budget)
        return {**self._measure(retained_ids), 'parity_guard': self.guard}

    def close(self):
        self.checkpoint = None
        self.batch = None
