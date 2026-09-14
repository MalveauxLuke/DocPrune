"""Portable sealed inputs and durable JSON records for the acquisition experiment."""
from __future__ import annotations
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import numpy as np
from .adaptive_acquisition import CONFIG, digest, make_groups, mask_id, scores


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write_new(path, value):
    """Atomic no-replace publication, including fsync before link publication."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {**value, 'artifact_sha256': digest(value)}
    if path.exists():
        if read_record(path) != value:
            raise ValueError(f'Conflicting immutable record: {path}')
        return
    fd, temporary = tempfile.mkstemp(prefix='.'+path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(payload, f, sort_keys=True, indent=2, allow_nan=False)
            f.write('\n')
            f.flush()
            os.fsync(f.fileno())
        os.link(temporary, path)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        os.unlink(temporary)


def read_record(path):
    value = read(path)
    expected = value.pop('artifact_sha256', None)
    if digest(value) != expected:
        raise ValueError(f'Corrupt record: {path}')
    return value


def tree_identity():
    root = Path(__file__).parent
    entrypoint = root.parent.parent/'scripts/stage0_acquisition.py'
    files = [('src/docprune/'+str(p.relative_to(root)), sha(p)) for p in sorted(root.rglob('*.py'))]
    files.append(('scripts/stage0_acquisition.py', sha(entrypoint)))
    return digest(files)


def token_identity(public, mask):
    a = np.asarray(mask)
    if a.shape != (len(public['costs']),) or not np.isin(a, [0, 1]).all():
        raise ValueError('Invalid original-action mask')
    token_mask = a.astype(bool)[public['owners']]
    return hashlib.sha256(np.packbits(token_mask).tobytes()).hexdigest(), np.flatnonzero(token_mask).tolist()


def validate_case(case):
    public, reader = case['public'], case['reader']
    costs = public['costs']
    n = len(costs)
    if any(type(c) is not int or c <= 0 for c in costs) or sum(costs) != 10032 or public['budget'] != 5016:
        raise ValueError('Pilot cost/population contract drift')
    if len(public['source_ids']) != n or len(set(public['source_ids'])) != n:
        raise ValueError('Non-bijective source identities')
    if len(public['owners']) != 10032 or any(type(i) is not int or not 0 <= i < n for i in public['owners']):
        raise ValueError('Invalid token ownership')
    if np.bincount(public['owners'], minlength=n).tolist() != costs:
        raise ValueError('Token ownership costs disagree')
    if len(public['priority']) != n or not np.isfinite(public['priority']).all() or len(public['regions']) != n:
        raise ValueError('Invalid proposal features')
    for arm in ('R', 'A'):
        masks = public['banks'][arm]
        if len(masks) != 32 or len({mask_id(mask) for mask in masks}) != 32:
            raise ValueError('Prepared bank must contain 32 distinct masks')
        for mask in masks:
            _, ids = token_identity(public, mask)
            if len(ids) != public['budget']:
                raise ValueError('Prepared bank token cost mismatch')
    targets = reader['targets']
    if len(targets) < 2 or any(not t or any(type(i) is not int or i < 0 for i in t) for t in targets):
        raise ValueError('Invalid frozen answer token sequences')
    if targets[-1] != reader['expected_generated_ids']:
        raise ValueError('S must be the actual original generated answer')
    if len(reader['reference_likelihoods']) != len(targets):
        raise ValueError('Reference likelihood/target mismatch')
    scores(reader['reference_likelihoods'])
    if len(reader['images']) != 4 or reader['repetition_penalty'] != 1.05:
        raise ValueError('Frozen page/decoding contract mismatch')
    groups = make_groups(public)
    if sorted(i for g in groups for i in g) != list(range(n)):
        raise ValueError('Invalid region grouping')
    return {'case': public['case'], 'actions': n, 'groups': len(groups), 'budget': public['budget']}


def prepare(receipt, banks, owned, output):
    """Copy selected images and sanitized metadata; never copy historical mask scores."""
    receipt, banks, owned, output = map(Path, (receipt, banks, owned, output))
    if output.exists():
        raise FileExistsError('Preparation requires a new output directory')
    root = receipt / 'input'
    source_manifest = dict((line.split(None, 1)[1].lstrip('*'), line.split(None, 1)[0])
                           for line in (root/'MANIFEST.sha256').read_text().splitlines() if line.strip())
    provenance = {}
    def verified(path):
        name = str(path.relative_to(root))
        actual = sha(path)
        if source_manifest.get(name) != actual:
            raise ValueError(f'Source receipt checksum mismatch: {name}')
        provenance[name] = actual
        return read(path) if path.suffix == '.json' else path
    run = verified(root/'sources/runtime/run-config.h200.json')
    processor = verified(root/'sources/runtime/processor-contract.qwen25.json')
    execution = run['execution_qwen']
    if sha(root/'sources/runtime/processor-contract.qwen25.json') != execution['processor_contract_sha256']:
        raise ValueError('Processor contract identity mismatch')
    cases = []
    # Validate every source before creating the destination.
    for name in [f'Q{i:02}' for i in range(1, 18)]:
        bank = read(banks/(name+'.json'))
        bank_signed = dict(bank)
        if bank_signed.pop('bank_sha256') != digest(bank_signed):
            raise ValueError('Prepared bank checksum mismatch')
        mapping = verified(root/f'sources/{name}/mapping.json')
        case = verified(root/f'sources/{name}/case.json')
        baseline = verified(root/f'sources/{name}/baseline.json')
        comparison = verified(root/f'sources/{name}/input/comparison-rp105.json')
        for field, path in [('source_mapping_sha256', root/f'sources/{name}/mapping.json'),
                            ('case_file_sha256', root/f'sources/{name}/case.json'),
                            ('baseline_file_sha256', root/f'sources/{name}/baseline.json'),
                            ('owned_profile_file_sha256', owned/name/'profiles.npz')]:
            if sha(path) != bank[field]:
                raise ValueError(f'Bank input drift: {name}/{field}')
        ids = bank['source_ids']
        lookup = {source['source_id']: source for source in mapping['sources']}
        if set(ids) != set(lookup):
            raise ValueError('Mapping and bank action identities differ')
        index = {sid: i for i, sid in enumerate(ids)}
        owners = [index[sid] for sid in mapping['token_to_source']]
        geometry = np.asarray(mapping['geometry'])
        regions = []
        for sid in ids:
            source = lookup[sid]
            token_ids = np.flatnonzero(np.asarray(owners) == index[sid])
            if sorted(source['token_ids']) != token_ids.tolist():
                raise ValueError('Source token IDs disagree with ownership')
            cells = geometry[token_ids]
            if len(set(cells[:, 0])) != 1 or len(set(map(tuple, cells[:, 3:]))) != 1:
                raise ValueError('Region spans incompatible pages or grids')
            regions.append({'kind': source['source_kind'], 'centroid': [int(cells[0, 0]),
                            float(cells[:, 1].mean()), float(cells[:, 2].mean()),
                            int(cells[0, 3]), int(cells[0, 4])]})
        public = {'case': name, 'source_ids': ids, 'costs': bank['costs'], 'budget': bank['token_budget'],
                  'priority': bank['priority'], 'regions': regions, 'owners': owners, 'banks': {}}
        for arm in ('R', 'A'):
            public['banks'][arm] = []
            for slot in bank['banks'][arm]['slots']:
                measurement = bank['measurements'][slot['mask_id']]
                mask = measurement['source_mask']
                token_hash, _ = token_identity(public, mask)
                if token_hash != measurement['token_mask_sha256']:
                    raise ValueError('Original prepared token hash drift')
                public['banks'][arm].append(mask)
        full = baseline['full_context_qwen']
        if (comparison['identity']['baseline_sha256'] != baseline['artifact_sha256']
                or comparison['identity']['mapping_sha256'] != bank['source_mapping_sha256']
                or comparison['boundary'] != 'FC_B_input' or comparison['identity']['boundary'] != 'input'):
            raise ValueError('All-keep reference belongs to a different teacher instance')
        reader = {'question': case['question'], 'qid': case['qid'], 'case_id': case['case_id'],
                  'targets': baseline['reference_token_ids'] + [baseline['fixed_self_token_ids']],
                  'reference_likelihoods': comparison['full_context_likelihoods'],
                  'expected_generated_ids': full['generated_response_token_ids'],
                  'expected_terminal_eos': full['terminal_eos_token_id'],
                  'prompt_sha256': full['assistant_prompt_sha256'],
                  'input_ids_sha256': full['prefill_input_ids_sha256'],
                  'repetition_penalty': comparison['generation_repetition_penalty'],
                  'images': [], 'pages': case['pages']}
        for page in range(4):
            image = root/f'images/{name}/page-{page:02}.png'
            verified(image)
            reader['images'].append({'path': f'images/{name}/page-{page:02}.png', 'sha256': sha(image)})
        clean = {'public': public, 'reader': reader, 'provenance': {'bank_sha256': bank['bank_sha256'],
                  'bank_file_sha256': sha(banks/(name+'.json')), 'source_mapping_sha256': bank['source_mapping_sha256'],
                  'baseline_file_sha256': bank['baseline_file_sha256'],
                  'all_keep_reference_source_sha256': sha(root/f'sources/{name}/input/comparison-rp105.json')}}
        validate_case(clean)
        cases.append(clean)
    output.mkdir(parents=True)
    runtime = {'execution_qwen': execution, 'processor': processor['qwen'],
               'max_new_tokens': run['max_new_tokens'], 'eos_ids': [151645, 151643],
               'dtype': 'bfloat16', 'boundary': 'input', 'mode': 'physical_delete',
               'expected_versions': {'torch': '2.4.1', 'transformers': '4.49.0', 'numpy': '1.26.4',
                                     'Pillow': '10.4.0', 'qwen-vl-utils': '0.0.8', 'accelerate': '1.1.0'}}
    write_new(output/'runtime.json', runtime)
    write_new(output/'controller.json', asdict(CONFIG))
    for clean in cases:
        name = clean['public']['case']
        write_new(output/f'cases/{name}.json', clean)
        for image in clean['reader']['images']:
            dest = output/image['path']
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root/image['path'], dest)
    files = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
    write_new(output/'manifest.json', {'schema': 'stage0-acquisition-input-v1', 'files': files,
              'source_receipt_manifest_sha256': sha(root/'MANIFEST.sha256'), 'verified_sources': provenance,
              'cases': [c['public']['case'] for c in cases], 'historical_mask_scores_included': False})
    return validate_package(output)


def validate_package(root):
    root = Path(root)
    manifest = read_record(root/'manifest.json')
    for name, expected in manifest['files'].items():
        path = root/name
        if not path.resolve().is_relative_to(root.resolve()) or sha(path) != expected:
            raise ValueError(f'Package file mismatch: {name}')
    if read_record(root/'controller.json') != asdict(CONFIG):
        raise ValueError('Frozen controller configuration mismatch')
    cases = [read_record(root/f'cases/{q}.json') for q in manifest['cases']]
    checks = [validate_case(c) for c in cases]
    for c in cases:
        for image in c['reader']['images']:
            if sha(root/image['path']) != image['sha256']:
                raise ValueError('Image identity mismatch')
    return {'manifest_sha256': sha(root/'manifest.json'), 'cases': checks,
            'image_count': sum(len(c['reader']['images']) for c in cases),
            'historical_mask_scores_included': False}


def bundle(package, output, repository):
    """Freeze code, tests, handoff and sealed inputs without weights or environments."""
    package, output, repository = map(Path, (package, output, repository))
    validation = validate_package(package)
    if output.exists():
        raise FileExistsError('Bundle requires a new output directory')
    output.mkdir(parents=True)
    shutil.copytree(package, output/'inputs')
    shutil.copytree(repository/'src', output/'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    selections = {
        'scripts/stage0_acquisition.py': 'scripts/stage0_acquisition.py',
        'docs/experiments/corrective-selection/STAGE0_ADAPTIVE_ACQUISITION.md': 'docs/STAGE0_ADAPTIVE_ACQUISITION.md',
        'docs/experiments/corrective-selection/STAGE0_MASKING_PREPARATION.md': 'docs/STAGE0_MASKING_PREPARATION.md',
        'AGENTS.md': 'AGENTS.md',
        'h200-operations/README.md': 'h200-operations/README.md',
        'h200-operations/CORAL_POLICY.md': 'h200-operations/CORAL_POLICY.md',
    }
    for name in ('test_adaptive_acquisition.py', 'test_acquisition_runner.py', 'test_acquisition_reader.py'):
        selections['tests/'+name] = 'tests/'+name
    for name in ('README.md', 'environment.sh', 'runtime-constraints.txt'):
        selections['h200/adaptive-acquisition/'+name] = 'h200/'+name
    for source, target in selections.items():
        destination = output/target
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repository/source, destination)
    (output/'README.md').write_text('Stage 0 adaptive acquisition: read h200/README.md and docs/STAGE0_ADAPTIVE_ACQUISITION.md.\n'
                                    'No GPU run is authorized by receiving this bundle.\n')
    files = {str(p.relative_to(output)): sha(p) for p in sorted(output.rglob('*')) if p.is_file()}
    (output/'BUNDLE.sha256').write_text(''.join(f'{value}  {name}\n' for name, value in files.items()))
    return {'input_manifest_sha256': validation['manifest_sha256'], 'files': len(files),
            'bytes': sum(p.stat().st_size for p in output.rglob('*') if p.is_file()),
            'bundle_manifest_sha256': sha(output/'BUNDLE.sha256')}
