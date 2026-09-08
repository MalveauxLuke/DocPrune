#!/usr/bin/env python3
"""Prepare sealed page regions for the correction depth comparison.

Reuses authenticated Task 8 rendering, MinerU, geometry and mapping APIs. The
MinerU template is an existing H200 run-manifest with current local model paths;
its outputs and the separate 600-question preparation are never modified.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

from docprune.correction_depth import read, resource, validate_case


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def canonical_write(path, value):
    with Path(path).open('x') as stream:
        stream.write(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True))


def require_gpu():
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if visible not in {'4', '5', '6', '7'}:
        raise ValueError('set CUDA_VISIBLE_DEVICES to one allocated physical H200 index 4..7')
    physical = subprocess.run(['nvidia-smi', '--id='+visible, '--query-gpu=uuid', '--format=csv,noheader'],
                              check=True, capture_output=True, text=True).stdout.strip()
    running = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid', '--format=csv,noheader'],
                             check=True, capture_output=True, text=True).stdout.splitlines()
    for row in running:
        uuid, pid = [x.strip() for x in row.split(',')]
        if uuid == physical and int(pid) != os.getpid():
            raise ValueError('allocated GPU is not idle: another process is present')
    import torch
    if not os.environ.get('CUDA_VISIBLE_DEVICES'):
        raise ValueError('set CUDA_VISIBLE_DEVICES to the explicitly allocated H200')
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError('preparation requires exactly one allocated CUDA-visible GPU')
    props = torch.cuda.get_device_properties(0)
    if 'H200' not in props.name:
        raise ValueError('this execution package is scoped to H200')
    drivers = subprocess.run(['nvidia-smi', '--query-gpu=driver_version', '--format=csv,noheader'],
                             check=True, capture_output=True, text=True).stdout.splitlines()
    drivers = {x.strip() for x in drivers if x.strip()}
    if len(drivers) != 1:
        raise ValueError('could not establish GPU driver identity')
    return {'name': props.name, 'memory_total_mib': props.total_memory//(1024*1024),
            'driver_version': next(iter(drivers))}


def seal(case, fixture, fixture_sha, case_root, commit):
    from docprune.task8_runtime import seal_task8_smoke_inputs, load_task8_smoke_inputs
    path = case_root/'inputs'/'smoke-input-manifest.json'
    if path.exists():
        value = load_task8_smoke_inputs(path)
        if value['qid'] != case['qid'] or value['fixed_page_fixture_sha256'] != fixture_sha:
            raise ValueError('sealed input fixture changed')
    else:
        seal_task8_smoke_inputs(fixture_path=fixture, fixture_sha256=fixture_sha,
            qid=case['qid'], output_root=path.parent, runtime_commit=commit)
    return path


def segment(case, smoke, case_root, commit, template_path, executable, gpu):
    from docprune.task8_runtime import (prepare_task8_mineru_smoke, finalize_task8_mineru_smoke,
        load_task8_mineru_completion, _load_signed_run_manifest)
    job = case_root/'mineru'
    job.mkdir(exist_ok=True)
    completion = job/'completion-manifest.json'
    if completion.exists():
        value = load_task8_mineru_completion(completion, expected_sha256=sha(completion))
        if _load_signed_run_manifest(Path(value['run_manifest_path']))['qid'] != case['qid']:
            raise ValueError('completed segmentation belongs to a different question')
        return completion
    template = _load_signed_run_manifest(template_path)
    run_path = job/'run-manifest.json'
    if not run_path.exists():
        kwargs = {}
        for key in ('configuration', 'tool_manifest', 'model_weights', 'model_inventory'):
            kwargs[key+'_path'] = Path(template[key+'_path'])
            kwargs[key+'_sha256'] = template[key+'_sha256']
        prepare_task8_mineru_smoke(smoke_input_manifest_path=smoke,
            smoke_input_manifest_sha256=sha(smoke), job_root=job, output_dir=job/'output',
            runtime_commit=commit, **kwargs)
    run = _load_signed_run_manifest(run_path)
    if run['smoke_input_manifest_sha256'] != sha(smoke) or run['runtime_commit'] != commit:
        raise ValueError('segmentation resume inputs changed')
    gpu_path = job/'gpu.json'
    if not gpu_path.exists():
        canonical_write(gpu_path, gpu)
    elif read(gpu_path) != gpu:
        raise ValueError('segmentation GPU identity changed')
    environment = dict(os.environ, MINERU_TOOLS_CONFIG_JSON=run['configuration_path'])
    for index, item in enumerate(run['inputs']):
        marker = job/f'page-{index:02d}.completed.json'
        command = [str(executable), '--path', item['path'], '--output', run['output_dir'],
                   '--backend', 'vlm-auto-engine', '--formula', 'true', '--table', 'true']
        identity = {'input': item, 'command': command, 'executable_sha256': sha(executable)}
        if marker.exists():
            saved = read(marker)
            if saved.get('identity') != identity:
                raise ValueError('segmentation page resume identity changed')
            for artifact in saved['outputs']:
                if not Path(artifact['path']).is_file() or sha(artifact['path']) != artifact['sha256']:
                    raise ValueError('completed MinerU page output missing or modified: '+artifact['path'])
            continue
        if sha(item['path']) != item['sha256']:
            raise ValueError('segmentation image hash changed')
        subprocess.run(command, env=environment, check=True)
        stem = Path(item['path']).stem
        outputs = [{'path': str(p), 'sha256': sha(p)} for p in sorted(Path(run['output_dir']).rglob('*'))
                   if p.is_file() and (stem in p.parts or p.name.startswith(stem))]
        if not any(Path(p['path']).name == stem+'_middle.json' for p in outputs):
            raise ValueError('MinerU returned success without expected page output')
        canonical_write(marker, {'identity': identity, 'outputs': outputs})
    finalize_task8_mineru_smoke(job_root=job, output_dir=job/'output',
        gpu_manifest_path=gpu_path, gpu_manifest_sha256=sha(gpu_path))
    return completion


def geometry(case, smoke, case_root, baseline_root, fixture, fixture_sha, commit, models):
    from docprune.task8_geometry import capture_task8_btp_qtp_geometry, load_task8_geometry_capture
    path = case_root/'geometry.json'
    if path.exists():
        value = load_task8_geometry_capture(path, expected_sha256=sha(path))
        if value['qid'] != case['qid'] or value['fixed_page_fixture_sha256'] != fixture_sha:
            raise ValueError('geometry resume fixture changed')
        return path
    reference = baseline_root/case['case_id']/'reference.jsonl'
    baseline = read(baseline_root/case['case_id']/'baseline.json')
    if not models:
        from docprune.m3docvqa_factory import load_pinned_colpali_query_encoder, load_pinned_qwen_processor
        models.extend([load_pinned_colpali_query_encoder(), load_pinned_qwen_processor()])
    capture_task8_btp_qtp_geometry(fixture_path=fixture, fixture_sha256=fixture_sha,
        smoke_input_manifest_path=smoke, smoke_input_manifest_sha256=sha(smoke),
        reference_results_path=reference, reference_results_sha256=sha(reference), qid=case['qid'],
        expected_geometry_count=baseline['unpruned']['trace']['post_qtp_visual_tokens'],
        expected_geometry_sha256=None, output_path=path, runtime_commit=commit,
        query_encoder=models[0], qwen_processor=models[1])
    return path


def mapping(case, case_root):
    from docprune.task8_mapping import publish_task8_region_mapping
    from docprune.segmentation import load_region_mapping
    geometry_path = case_root/'geometry.json'
    completion = case_root/'mineru'/'completion-manifest.json'
    path = case_root/'mapping.json'
    capture = read(geometry_path)
    if path.exists():
        value = load_region_mapping(path, validate_raw_artifacts=True)
        if value.geometry_sha256 != capture['geometry_sha256']:
            raise ValueError('mapping resume geometry changed')
    else:
        publish_task8_region_mapping(mineru_completion_path=completion,
            mineru_completion_sha256=sha(completion), geometry_capture_path=geometry_path,
            geometry_capture_sha256=sha(geometry_path), output_path=path,
            expected_qid=case['qid'], expected_geometry_count=capture['geometry_count'],
            expected_geometry_sha256=capture['geometry_sha256'], required_geometry_gpu_substring='H200')
    return {'path': str(path), 'sha256': sha(path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('phase', choices=['seal', 'segment', 'geometry', 'map'])
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--resources', type=Path, required=True)
    parser.add_argument('--baseline-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runtime-commit', required=True)
    parser.add_argument('--case-id', action='append')
    parser.add_argument('--mineru-template-run-manifest', type=Path)
    parser.add_argument('--mineru-executable', type=Path)
    parser.add_argument('--execute-gpu', action='store_true', help='operator explicitly authorizes this allocated-GPU stage')
    args = parser.parse_args()
    gpu_phase = args.phase in {'segment', 'geometry'}
    if gpu_phase and not args.execute_gpu:
        parser.error('GPU phases require --execute-gpu on the allocated H200')
    if args.phase in {'segment'} and (args.mineru_template_run_manifest is None or args.mineru_executable is None):
        parser.error('segment requires pinned existing template run manifest and MinerU executable')
    cases = [validate_case(c) for c in read(args.corpus)['cases']]
    if args.case_id:
        wanted = set(args.case_id)
        cases = [c for c in cases if c['case_id'] in wanted]
        if {c['case_id'] for c in cases} != wanted:
            parser.error('unknown case ID')
    resources = read(args.resources)
    root = args.resources.resolve().parent
    fixture = resource(resources['fixture'], root)
    args.output = args.output.resolve()
    args.output.mkdir(parents=True, exist_ok=True)
    models = []
    gpu = require_gpu() if gpu_phase else None
    for case in cases:
        case_root = args.output/case['case_id']
        case_root.mkdir(exist_ok=True)
        smoke = seal(case, fixture, resources['fixture']['sha256'], case_root, args.runtime_commit)
        if args.phase in {'segment'}:
            segment(case, smoke, case_root, args.runtime_commit,
                args.mineru_template_run_manifest.resolve(), args.mineru_executable.resolve(), gpu)
        if args.phase in {'geometry'}:
            geometry(case, smoke, case_root, args.baseline_root.resolve(), fixture,
                     resources['fixture']['sha256'], args.runtime_commit, models)
        if args.phase in {'map'}:
            resources.setdefault('mappings', {})[case['qid']] = mapping(case, case_root)
            # Derived resource registry only. Source inputs and previous baselines stay immutable.
            temporary = args.resources.with_suffix('.json.tmp')
            temporary.write_text(json.dumps(resources, indent=2, ensure_ascii=False)+'\n')
            temporary.replace(args.resources)
        print(json.dumps({'case_id': case['case_id'], 'phase': args.phase, 'status': 'completed'}), flush=True)

if __name__ == '__main__':
    main()
