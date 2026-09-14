#!/usr/bin/env python3
"""Prepare/validate/simulate locally; explicit GPU commands are later execution steps."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

# Support both repository and self-contained portable package execution.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from docprune.acquisition_io import prepare, bundle, validate_package, read_record, sha, write_new, tree_identity
from docprune.acquisition_runner import run_case, run_lock, SyntheticReader
from docprune.acquisition_reader import environment_report
from docprune.adaptive_acquisition import digest


def gpu_preflight(args, *, check_occupancy=True):
    if not args.execute_gpu:
        raise ValueError('GPU execution requires --execute-gpu; preparation never runs a model')
    if getattr(args, 'platform', 'h200') == 'sol':
        from docprune.acquisition_sol import sol_preflight
        return sol_preflight(args, check_occupancy=check_occupancy)
    if args.physical_gpu not in range(4, 8):
        raise ValueError('This experiment permits only CoRAL physical GPUs 4–7')
    if not args.coral_channel_or_relay:
        raise ValueError('The operator must confirm existing CoRAL channel membership or active relay')
    for path in (args.package, args.output):
        if not Path(path).resolve().is_relative_to('/mnt/data1/eunwooim/DocPrune'):
            raise ValueError('H200 inputs/outputs must be under the designated CoRAL project checkout')
    for name in ('HF_HOME', 'TRANSFORMERS_CACHE', 'PIP_CACHE_DIR', 'UV_CACHE_DIR', 'TORCH_HOME',
                 'XDG_CACHE_HOME', 'CONDA_PKGS_DIRS', 'TMPDIR'):
        if not os.environ.get(name) or not Path(os.environ[name]).resolve().is_relative_to('/mnt/data2/eunwooim'):
            raise ValueError(f'{name} must be pinned to CoRAL storage; source the task environment file')
    if not check_occupancy:
        return None
    device = str(args.physical_gpu)
    query = subprocess.check_output(['nvidia-smi', '-i', device, '--query-gpu=uuid,name,memory.used,utilization.gpu',
                                      '--format=csv,noheader,nounits'], text=True).strip()
    uuid, name, memory, utilization = [part.strip() for part in query.split(',')]
    processes = subprocess.check_output(['nvidia-smi', '-i', device, '--query-compute-apps=pid',
                                         '--format=csv,noheader,nounits'], text=True).strip()
    if processes or float(memory) > 512 or float(utilization) > 5:
        raise RuntimeError('Selected GPU is occupied; no workload was started')
    if 'H200' not in name:
        raise ValueError('Selected device is not an H200')
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    return {'physical_gpu': args.physical_gpu, 'uuid': uuid, 'name': name, 'memory_used_mb': memory,
            'utilization_percent': utilization, 'compute_processes': processes}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    prep = sub.add_parser('prepare')
    for name in ('receipt', 'banks', 'owned', 'output'):
        prep.add_argument('--'+name, required=True, type=Path)
    packed = sub.add_parser('bundle')
    packed.add_argument('--package', required=True, type=Path)
    packed.add_argument('--output', required=True, type=Path)
    for command in ('validate', 'simulate', 'doctor', 'smoke', 'score'):
        p = sub.add_parser(command)
        p.add_argument('--package', required=True, type=Path)
        if command != 'validate':
            p.add_argument('--cases', nargs='+', default=None)
        if command in ('simulate', 'smoke', 'score'):
            p.add_argument('--output', required=True, type=Path)
        if command in ('doctor', 'smoke', 'score'):
            p.add_argument('--snapshot', type=Path)
        if command in ('smoke', 'score'):
            p.add_argument('--execute-gpu', action='store_true')
            p.add_argument('--physical-gpu', type=int)
            p.add_argument('--platform', choices=('h200', 'sol'), default='h200')
            p.add_argument('--coral-channel-or-relay', action='store_true')
        if command == 'score':
            p.add_argument('--smoke-receipt', required=True, type=Path)
    args = parser.parse_args()
    if args.command == 'prepare':
        print(json.dumps(prepare(args.receipt, args.banks, args.owned, args.output), indent=2))
        return
    if args.command == 'bundle':
        print(json.dumps(bundle(args.package, args.output, Path(__file__).resolve().parents[1]), indent=2))
        return
    validated = validate_package(args.package)
    if args.command == 'validate':
        print(json.dumps(validated, indent=2))
        return
    runtime = read_record(args.package/'runtime.json')
    manifest = read_record(args.package/'manifest.json')
    selected = args.cases or (['Q12'] if args.command == 'smoke' else manifest['cases'])
    if len(selected) != len(set(selected)) or not set(selected) <= set(manifest['cases']):
        raise ValueError('Unknown or duplicate question IDs')
    if args.command == 'smoke' and len(selected) != 1:
        raise ValueError('Smoke is exactly one question')
    cases = [read_record(args.package/f'cases/{q}.json') for q in selected]
    snapshot = getattr(args, 'snapshot', None) or Path(runtime['execution_qwen']['snapshot_path'])
    if args.command == 'doctor':
        print(json.dumps(environment_report(runtime, snapshot), indent=2))
        return
    if args.command == 'simulate':
        with run_lock(args.output):
            results = [run_case(c, args.output, SyntheticReader, SyntheticReader.identity) for c in cases]
        print(json.dumps({'synthetic': True, 'completed_cases': len(results),
                          'logical_observations': sum(sum(r['logical_observations'].values()) for r in results),
                          'physical_synthetic_masks': sum(r['unique_physical_masks'] for r in results)}, indent=2))
        return
    gpu_preflight(args, check_occupancy=False)
    with run_lock(args.output):
        # No Torch/model imports occur before the explicit execution and occupancy gate.
        preflight = gpu_preflight(args)
        if args.command == 'score':
            smoke = read_record(args.smoke_receipt)
            expected = {'package_sha256': validated['manifest_sha256'], 'code_sha256': tree_identity()}
            if any(smoke.get(k) != v for k, v in expected.items()) or smoke.get('passed') is not True:
                raise ValueError('An intact matching successful smoke receipt is required')
        from docprune.acquisition_reader import QwenTeacher
        teacher = QwenTeacher(runtime, snapshot)
        if args.command == 'smoke':
            checkpoint = teacher.question(cases[0], args.package)
            from docprune.acquisition_io import token_identity
            _, ids = token_identity(cases[0]['public'], cases[0]['public']['banks']['A'][0])
            first = checkpoint.score(ids)
            second = checkpoint.score(ids)
            from docprune.acquisition_reader import validate_parity
            replay_error = validate_parity(first['likelihoods'], second['likelihoods'])
            receipt = {'passed': True, 'case': selected[0], 'package_sha256': validated['manifest_sha256'],
                       'code_sha256': tree_identity(), 'environment': teacher.identity, 'preflight': preflight,
                       'guard': checkpoint.guard, 'repeated_mask_error': replay_error,
                       'first_measurement': first, 'scope': 'one-question integration smoke, not experiment results'}
            write_new(args.output/'smoke.json', receipt)
            checkpoint.close()
            print(json.dumps({'passed': True, 'receipt': str(args.output/'smoke.json')}, indent=2))
        else:
            if smoke['environment'] != teacher.identity:
                raise ValueError('Smoke and scoring runtime identities differ')
            results = [run_case(c, args.output, lambda c: teacher.question(c, args.package), teacher.identity) for c in cases]
            print(json.dumps({'completed_cases': len(results), 'masked_scoring_complete': True,
                              'decoded_candidate_review_complete': False}, indent=2))


if __name__ == '__main__':
    main()
