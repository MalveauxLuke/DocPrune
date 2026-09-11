"""Extract ColQwen features on the sealed 17 cases; no retrieval or answerer calls."""
import argparse
import csv
import json
import platform
import shutil
import subprocess
import time
from pathlib import Path

from colfeatures_package import inventory, sha

ADAPTER = ('vidore/colqwen2.5-v0.2', 'dcbe8d9cede518bce830488364ba0e40c873645b')
BASE = ('vidore/colqwen2.5-base', '92908120384b7a2110c5beda3ab29cbdb2c08e49')


def load_verified_adapter(model, adapter_path):
    """Map historical Qwen keys and verify every loaded adapter tensor."""
    import re
    import torch
    from safetensors import safe_open

    # Transformers' automatic VLM detection does not recognize ColQwen2_5.
    mapping = dict(model._checkpoint_conversion_mapping)
    model.load_adapter(adapter_path, adapter_kwargs={'key_mapping': mapping})
    loaded = model.get_adapter_state_dict()
    matched = set()
    with safe_open(str(Path(adapter_path) / 'adapter_model.safetensors'),
                   framework='pt', device='cpu') as source:
        for key in source.keys():
            target = key.removeprefix('base_model.model.')
            for pattern, replacement in mapping.items():
                target, count = re.subn(pattern, replacement, target)
                if count:
                    break
            if target in matched or target not in loaded:
                raise ValueError(f'Unmatched or duplicate adapter key: {key} -> {target}')
            actual = loaded[target].detach().cpu()
            expected = source.get_tensor(key).to(dtype=actual.dtype)
            if actual.shape != expected.shape or not torch.equal(actual, expected):
                raise ValueError(f'Adapter tensor differs from pinned checkpoint: {target}')
            matched.add(target)
    if matched != set(loaded):
        raise ValueError('Loaded adapter contains unverified tensors')
    return {'status': 'passed', 'key_mapping': mapping, 'verified_tensors': len(matched),
            'comparison': 'exact equality after casting checkpoint to loaded dtype',
            'active_adapters': model.active_adapters()}


def dump(path, data):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def audit(packet):
    from PIL import Image
    cases = []
    for ordinal in range(1, 18):
        label = f'Q{ordinal:02d}'
        source = packet / 'sources' / label
        case = json.loads((source / 'case.json').read_text())
        mapping = json.loads((source / 'mapping.json').read_text())
        if len(case['pages']) != 4 or not case['question'].strip():
            raise ValueError(f'Invalid case: {label}')
        for required in ['baseline.json', 'geometry.json', 'input/mask-scores.json',
                         'intermediate/mask-scores.json', 'input/comparison-rp105.json',
                         'intermediate/comparison-rp105.json']:
            if not (source / required).is_file():
                raise ValueError(f'Missing source {label}/{required}')
        pages = []
        for rank, page in enumerate(case['pages']):
            image = packet / 'images' / label / f'page-{rank:02d}.png'
            with Image.open(image) as im:
                rgb = im.convert('RGB')
                size = list(rgb.size)
                import hashlib
                rgb_sha = hashlib.sha256(rgb.tobytes()).hexdigest()
            # Authenticate the image pixels against the saved reader geometry provenance.
            records = [r for a in mapping['artifacts'] for r in a.get('pages', [])
                       if r['input_page_index'] == rank]
            if not records or any(r['rendered_rgb_sha256'] != rgb_sha or
                                  r['document_id'] != page['doc_id'] or
                                  r['source_page_index'] != page['page_index'] for r in records):
                raise ValueError(f'Image identity mismatch: {label}/{rank}')
            pages.append({'rank': rank, 'doc_id': page['doc_id'], 'page_index': page['page_index'],
                          'image': str(image.relative_to(packet)), 'sha256': sha(image),
                          'rgb_sha256': rgb_sha, 'size': size})
        cases.append({'label': label, 'qid': case['qid'], 'question': case['question'],
                      'pages': pages, 'mapping_sha256': sha(source / 'mapping.json')})
    if len({c['qid'] for c in cases}) != 17:
        raise ValueError('Duplicate case identity')
    return cases


def region_membership(sources, rank, height, width):
    """Geometric overlap, intentionally separate from original reader token assignment."""
    import numpy as np
    boxes = np.array([[x / width, y / height, (x + 1) / width, (y + 1) / height]
                      for y in range(height) for x in range(width)], dtype=np.float32)
    regions = [r for r in sources if r['input_page_index'] == rank]
    overlap = np.zeros((len(regions), len(boxes)), dtype=np.float32)
    for i, region in enumerate(regions):
        w, h = region['page_size']
        if w <= 0 or h <= 0:
            raise ValueError('Invalid region page size')
        x0, y0, x1, y1 = region['bbox']
        rect = np.array([x0 / w, y0 / h, x1 / w, y1 / h])
        lo = np.maximum(boxes[:, :2], rect[:2])
        hi = np.minimum(boxes[:, 2:], rect[2:])
        overlap[i] = np.maximum(hi - lo, 0).prod(axis=1) * height * width
    return regions, boxes, overlap


def extract(packet, output, limit):
    import numpy as np
    import torch
    from PIL import Image
    from colpali_engine.models import ColQwen2_5, ColQwen2_5_Processor
    from huggingface_hub import snapshot_download
    cases = audit(packet)
    output.mkdir(parents=True, exist_ok=False)
    provenance = output / 'provenance'
    provenance.mkdir()
    dump(provenance / 'input-inventory.json', inventory(packet))
    dump(provenance / 'cases.json', cases)
    dump(provenance / 'run.json', {'status': 'started', 'adapter': ADAPTER, 'base': BASE,
         'python': platform.python_version(), 'torch': torch.__version__,
         'cuda': torch.version.cuda, 'git_commit': subprocess.check_output(
             ['git', 'rev-parse', 'HEAD'], text=True).strip(), 'case_limit': limit,
         'query_prefix': 'Query: ', 'suffix': '<|endoftext|>' * 10,
         'max_pixels': 602112, 'attn_implementation': 'sdpa', 'fresh_retrieval': False})
    subprocess.run(['python', '-m', 'pip', 'freeze'], stdout=(provenance / 'pip-freeze.txt').open('w'), check=True)
    subprocess.run(['nvidia-smi'], stdout=(provenance / 'nvidia-smi.txt').open('w'), check=True)
    adapter_path = snapshot_download(ADAPTER[0], revision=ADAPTER[1])
    base_path = snapshot_download(BASE[0], revision=BASE[1])
    # Loading base and adapter separately pins BOTH repositories (adapter config has revision=null).
    model = ColQwen2_5.from_pretrained(base_path, torch_dtype=torch.bfloat16,
                                      device_map='cuda:0', attn_implementation='sdpa')
    dump(provenance / 'adapter-verification.json', load_verified_adapter(model, adapter_path))
    model.eval()
    processor = ColQwen2_5_Processor.from_pretrained(adapter_path)
    processor.query_prefix = 'Query: '
    if processor.image_processor.max_pixels != 602112:
        raise ValueError('Processor resolution drift')
    processor.save_pretrained(provenance / 'processor')
    model.config.to_json_file(provenance / 'model-config.json')
    for name, path in [('adapter', adapter_path), ('base', base_path)]:
        # Hash actual downloaded weights; copy only small configs/cards, never weights.
        dump(provenance / f'{name}-snapshot-inventory.json', inventory(Path(path).resolve())
             if not any(p.is_symlink() for p in Path(path).rglob('*')) else
             [{'path': str(p.relative_to(path)), 'bytes': p.stat().st_size, 'sha256': sha(p)}
              for p in sorted(Path(path).rglob('*')) if p.is_file()])
        for filename in ['config.json', 'adapter_config.json', 'README.md']:
            if (Path(path) / filename).is_file():
                shutil.copyfile(Path(path) / filename, provenance / f'{name}-{filename}')
    captured = {}

    def capture_hidden(_module, args):
        captured['pre_projection_hidden'] = args[0].detach().float().cpu().numpy()

    def capture_visual(_module, _args, result):
        if not isinstance(result, torch.Tensor):
            raise TypeError('Unexpected vision output; audit required')
        captured['merged_vision_features'] = result.detach().float().cpu().numpy()

    handles = [model.custom_text_proj.register_forward_pre_hook(capture_hidden),
               model.visual.register_forward_hook(capture_visual)]
    summaries = []
    for case in cases[:limit]:
        start = time.time()
        target = output / 'features' / case['label']
        target.mkdir(parents=True)
        query_batch = processor.process_queries([case['question']])
        query_inputs = {k: v.numpy() for k, v in query_batch.items()}
        captured.clear()
        with torch.inference_mode():
            query = model(**query_batch.to(model.device))[0].float().cpu().numpy()
        query_extra = dict(captured)
        np.savez_compressed(target / 'query.npz', embeddings=query, **query_inputs, **query_extra)
        dump(target / 'query.json', {'question': case['question'], 'tokens':
             processor.tokenizer.convert_ids_to_tokens(query_inputs['input_ids'][0].tolist())})
        mapping = json.loads((packet / 'sources' / case['label'] / 'mapping.json').read_text())
        for page in case['pages']:
            rank = page['rank']
            with Image.open(packet / page['image']) as image:
                batch = processor.process_images([image.convert('RGB')])
            arrays = {k: v.numpy() for k, v in batch.items()}
            grid_t, grid_h, grid_w = arrays['image_grid_thw'][0].tolist()
            merge = model.spatial_merge_size
            h, w = grid_h // merge, grid_w // merge
            image_positions = np.flatnonzero(arrays['input_ids'][0] == model.config.image_token_id)
            if grid_t != 1 or len(image_positions) != h * w:
                raise ValueError('Image grid/token mismatch')
            captured.clear()
            with torch.inference_mode():
                embedding = model(**batch.to(model.device))[0].float().cpu().numpy()
            if not {'pre_projection_hidden', 'merged_vision_features'} <= captured.keys():
                raise ValueError('Missing feature hook output')
            if not np.isfinite(embedding).all() or not np.allclose(np.linalg.norm(embedding, axis=-1), 1, atol=.03):
                raise ValueError('Invalid normalized embeddings')
            similarity = query @ embedding.T
            patch_similarity = similarity[:, image_positions]
            regions, boxes, overlap = region_membership(mapping['sources'], rank, h, w)
            profiles = np.full((len(regions), len(query)), np.nan, dtype=np.float32)
            for i in range(len(regions)):
                member = overlap[i] > 0
                if member.any():
                    profiles[i] = patch_similarity[:, member].max(axis=1)
            direct_score = float(similarity.max(axis=1).sum())
            with torch.inference_mode():
                library_score = float(processor.score_multi_vector(
                    [torch.from_numpy(query)], [torch.from_numpy(embedding)], device='cpu')[0, 0])
            if not np.isclose(direct_score, library_score, atol=1e-4, rtol=1e-5):
                raise ValueError('MaxSim reconstruction failed')
            np.savez_compressed(target / f'page-{rank:02d}.npz', embeddings=embedding,
                image_positions=image_positions, patch_boxes_normalized=boxes,
                region_patch_overlap=overlap, query_token_by_page_token=similarity,
                region_query_maxsim=profiles, **arrays, **captured)
            dump(target / f'page-{rank:02d}.json', {'page': page,
                'grid_after_merge': [h, w], 'regions': [{k: v for k, v in r.items() if k != 'token_ids'} for r in regions],
                'token_strings': processor.tokenizer.convert_ids_to_tokens(arrays['input_ids'][0].tolist()),
                'full_page_maxsim': direct_score, 'image_only_maxsim': float(patch_similarity.max(axis=1).sum()),
                'uncovered_patches': int((overlap.max(axis=0) == 0).sum()) if len(regions) else h * w,
                'overlap_rule': 'positive area intersection; many-to-many; no nearest fallback',
                'empty_region_profile': 'NaN means no overlapping image token',
                'floating_storage': 'float32; lossless expansion of BF16 states; processor inputs retain native dtype'})
            summaries.append({'case': case['label'], 'rank': rank, 'tokens': len(embedding),
                              'visual_tokens': len(image_positions), 'regions': len(regions),
                              'full_page_maxsim': direct_score, 'image_only_maxsim': float(patch_similarity.max(axis=1).sum())})
        print(f"Completed {case['label']} in {time.time() - start:.1f}s", flush=True)
    for handle in handles:
        handle.remove()
    with (output / 'page-summary.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)
    dump(output / 'completion.json', {'status': 'complete' if limit == 17 else 'smoke_only',
         'cases': min(limit, 17), 'pages': len(summaries), 'fresh_retrieval': False,
         'answerer_calls': 0, 'training_steps': 0, 'files': inventory(output)})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('packet', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--audit-only', action='store_true')
    parser.add_argument('--limit', type=int, default=17, choices=range(1, 18))
    args = parser.parse_args()
    if args.audit_only:
        print(json.dumps(audit(args.packet), indent=2))
    else:
        if args.output is None:
            parser.error('--output is required for extraction')
        extract(args.packet, args.output, args.limit)
