"""Portable, fixed-page correction corpus assembly. No retrieval or model execution."""
from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def complete_gold(contract: dict) -> list[str]:
    """Score a complete set as one sequence, never its members as alternatives."""
    items = [item['canonical'] for item in contract['items']]
    if contract['kind'] in {'scalar', 'number'}:
        if len(items) != 1:
            raise ValueError('scalar contract cardinality')
        return items
    return [contract.get('separator', ', ').join(items)]


def collect_cases(source: Path) -> list[dict]:
    from docprune.correction_scoring import contract_digest, score_answer
    def read(name):
        return json.loads((source / name).read_text())
    originals = {r['qid']: r for r in read('candidates.json')}
    groups = [('verified-wrong-seed.json', read('verified-wrong-seed.json')),
              ('round3/evidence-ready-native-allkept-wrong.json', read('round3/evidence-ready-native-allkept-wrong.json')),
              ('minimal-repairs-v1.json', [r for r in read('minimal-repairs-v1.json')['cases'] if r.get('question')]),
              ('round2/minimal-revisions.json', read('round2/minimal-revisions.json')),
              ('round3/minimal-revisions.json', read('round3/minimal-revisions.json')),
              ('round4/selected22.json', read('round4/selected22.json'))]
    recovered = []
    for recovery in read('gold-page-examples/manifest.json'):
        row = copy.deepcopy(originals[recovery['qid']])
        row['original_pages'] = copy.deepcopy(row['pages'])
        # Rank 1 retains Banner Gwin's required 2007 table row; replace rank 3 only.
        row['pages'][3] = {k: recovery[k] for k in ('doc_id', 'page_index', 'source_pdf')}
        row['pages'][3]['score'] = row['original_pages'][3]['score']
        answer = '$94 million' if row['qid'].startswith('2b230') else 'Tom Berninger'
        aliases = ['94 million dollars', 'US$94 million', '$94,000,000'] if answer.startswith('$') else []
        row['answer_contract'] = {'kind': 'scalar', 'items': [{'canonical': answer, 'aliases': aliases}]}
        row['input_variant_id'] = row['qid'] + '::recovered-page-v1'
        row['recovery_evidence'] = recovery
        recovered.append(row)
    groups.append(('gold-page-examples/manifest.json', recovered))
    cases = []
    for source_name, rows in groups:
        for row in rows:
            row = copy.deepcopy(row)
            qid = row.get('parent_qid', row.get('qid'))
            original = originals[qid]
            contract = row.get('answer_contract') or row.get('draft_answer_contract') or row.get('auxiliary_audit', {}).get('answer_contract')
            if not contract:
                gold = row['gold']
                contract = {'kind': 'scalar' if len(gold) == 1 else 'set',
                            'items': [{'canonical': g, 'aliases': []} for g in gold]}
            contract = copy.deepcopy(contract)
            if qid == '0cd4923356d55af65924ab4dc6c8a99d':
                contract['items'][0]['aliases'] = ['Ennis House', 'Ennis House, Los Feliz, California', 'the Ennis House']
            if qid == 'ee67797fad4a7e1abc07e46bcd485887':
                contract['items'][0]['aliases'] = ['a church', 'Baptist church', 'a Baptist church']
            if qid == '265da4be224029df08ddc9dec1637a29':
                contract['items'][0]['aliases'] += ['gray', 'grey', 'silver', 'silver-gray', 'gray hair', 'grey hair']
            contract['frozen_before_oracle'] = True
            contract['freeze_stage'] = 'correction-depth-package-v1; agent-reviewed, not independent human adjudication'
            gold = complete_gold(contract)
            if contract['kind'] in {'set', 'ordered'}:
                contract.setdefault('accepted_complete', []).extend(g for g in gold if g not in contract.get('accepted_complete', []))
            if score_answer(gold[0], contract)['status'] != 'correct':
                raise ValueError('complete gold fails contract: ' + qid)
            pages = row['pages']
            changed_pages = [(p['doc_id'], p['page_index']) for p in pages] != [(p['doc_id'], p['page_index']) for p in original['pages']]
            changed_question = row['question'] != original['question']
            case_id = row.get('id') or (row.get('input_variant_id') if changed_pages else None) or qid
            cases.append(dict(qid=qid, fixture_qid=qid, case_id=case_id, parent_qid=qid,
                              parent_question_id=qid, question=row['question'],
                              gold_answers=gold, complete_gold_sequences=gold,
                              answer_contract=contract, contract_sha256=contract_digest(contract),
                              original_question=original['question'], original_answers=original['gold_original'],
                              original_pages=original['pages'], pages=pages,
                              question_variant=changed_question, input_variant=changed_pages,
                              question_type=row.get('question_type', original['question_type']),
                              modalities=row.get('modalities', original['modalities']),
                              document_component=original['document_component'], split='development',
                              protected_overlap_clear=True, source_manifest=source_name,
                              source_manifest_sha256=sha256(source / source_name),
                              evidence_review=row.get('review') or row.get('auxiliary_audit') or row.get('recovery_evidence') or {k:row[k] for k in ('evidence','support_ranks','reason','edit') if k in row},
                              prior_baseline=original.get('baseline') if not (changed_pages or changed_question) else None,
                              saved_native_answer=original.get('native_answer'),
                              saved_all_kept_reference=original.get('all_kept_reference'),
                              baseline_status='rerun matching baseline for this sealed case; retain correct cases as preservation controls'))
    expected = {r['qid'] for r in read('round4/combined40-index.json')['prior18']} | {r['qid'] for r in read('round4/selected22.json')}
    if len(cases) != 40 or {c['qid'] for c in cases} != expected or len({c['document_component'] for c in cases}) != 40:
        raise ValueError('combined40 membership/component mismatch')
    return cases


def build_package(source: Path, output: Path, ledger_path: Path, *, include_assets: bool = True) -> dict:
    if (output/'MANIFEST.sha256').exists():
        raise FileExistsError('sealed package already exists: ' + str(output))
    cases = collect_cases(source)
    ledger = {r['doc_id']: r for r in json.loads(ledger_path.read_text())}
    docs = {p['doc_id']: p['source_pdf'] for c in cases for p in c['pages']}
    assets = []
    output.mkdir(parents=True, exist_ok=True)
    selected_ledger = []
    missing = []
    for doc, pdf in sorted(docs.items()):
        entry = ledger[doc]
        selected_ledger.append(entry)
        for kind, filename, expected in [('pdf', pdf, None), ('features', entry['document_path'], entry['sha256'])]:
            path = Path(filename)
            relative = 'assets/' + kind + '/' + doc + path.suffix
            asset = dict(doc_id=doc, kind=kind, original_path=filename, path=relative)
            try:
                digest = sha256(path)
                if expected and digest != expected:
                    raise ValueError('feature SHA mismatch: ' + filename)
                asset.update(sha256=digest, bytes=path.stat().st_size, included=include_assets)
                if include_assets:
                    target = output / relative
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if not target.exists() or sha256(target) != digest:
                        shutil.copyfile(path, target)
            except OSError as error:
                asset.update(sha256=expected, bytes=None, included=False, missing_reason=str(error))
                missing.append(asset)
            assets.append(asset)
    by_doc = {r['doc_id']:r for r in assets if r['kind']=='pdf'}
    for case in cases:
        for page in case['pages']:
            page['source_pdf_sha256'] = by_doc[page['doc_id']]['sha256']
            page['source_pdf'] = by_doc[page['doc_id']]['path']
    package = dict(schema_version='correction-depth-corpus-v1', cases=cases, question_count=40,
                   page_count=160, global_index_included=False, new_retrieval=False,
                   corpus_role='exploratory development; no learned-selector train/test split',
                   source_ledger_sha256=sha256(ledger_path), assets=assets,
                   source_evidence_checks={'protected_exclusion':json.loads((source/'round4/protected-exclusion-check.json').read_text())})
    write_json(output/'corpus.json', package)
    write_json(output/'selected-source-ledger.json', selected_ledger)
    write_json(output/'build-summary.json', {'cases':40, 'documents':len(docs), 'question_variants':sum(c['question_variant'] for c in cases), 'input_variants':sum(c['input_variant'] for c in cases), 'asset_bytes_available':sum(a['bytes'] or 0 for a in assets), 'assets_included':sum(a['included'] for a in assets), 'assets_missing':len(missing), 'unsealed_pdf_identities':sum(a['sha256'] is None for a in assets)})
    write_json(output/'missing-assets.json', missing)
    manifest = [f'{sha256(p)}  {p.relative_to(output)}' for p in sorted(output.rglob('*')) if p.is_file()]
    (output/'MANIFEST.sha256').write_text('\n'.join(manifest)+'\n')
    return package


def resolve_asset(asset: dict, package: Path, roots: list[Path]) -> Path:
    """Search exact documented layouts, never scan or modify another cohort."""
    if not asset.get('sha256'):
        raise ValueError('Source PDF identity remains unsealed: ' + asset['original_path'])
    original = Path(asset['original_path'])
    candidates = [package / asset['path']]
    for root in roots:
        candidates += [root / original.relative_to('/'), root / 'raw' / original.relative_to('/'),
                       root / original.name, root / asset['path']]
    for candidate in candidates:
        try:
            if candidate.is_file() and not candidate.is_symlink() and sha256(candidate) == asset['sha256']:
                return candidate.resolve()
        except OSError:
            continue
    raise FileNotFoundError(f"Missing authenticated {asset['kind']} for {asset['doc_id']}: {asset['original_path']} sha256={asset['sha256']}")


def check_assets(package: Path, roots: list[Path]) -> dict:
    """Read-only CPU inventory; no images, models, retrieval or output assembly."""
    for line in (package/'MANIFEST.sha256').read_text().splitlines():
        digest, relative = line.split('  ', 1)
        if sha256(package/relative) != digest:
            raise ValueError('recipe checksum mismatch: ' + relative)
    corpus = json.loads((package/'corpus.json').read_text())
    found, missing = [], []
    for asset in corpus['assets']:
        try:
            path = resolve_asset(asset, package, roots)
            found.append({'doc_id':asset['doc_id'], 'kind':asset['kind'], 'path':str(path)})
        except (ValueError, OSError) as error:
            missing.append(dict(asset, reason=str(error)))
    blocked_docs = {a['doc_id'] for a in missing}
    available, blocked = [], []
    for case in corpus['cases']:
        absent = sorted({p['doc_id'] for p in case['pages']} & blocked_docs)
        if absent:
            blocked.append({'case_id':case['case_id'], 'missing_documents':absent})
        else:
            available.append(case['case_id'])
    return {'candidate_count':len(corpus['cases']), 'available_case_count':len(available),
            'available_case_ids':available, 'blocked_cases':blocked,
            'found_assets':found, 'missing_assets':missing,
            'note':'Missing means absent or unmatched in the supplied roots; no retrieval or rebuilding was attempted.'}


def rebase_index_manifest(source: Path, artifact_root: Path, ledger_path: Path):
    """Rebase selected-ledger paths while preserving the original index identity."""
    from dataclasses import fields, replace
    from docprune.artifacts import IndexManifest, canonical_json_sha256
    payload = json.loads(source.read_text())
    unsigned = {k:v for k,v in payload.items() if k != 'manifest_sha256'}
    if payload.get('schema_version') != 5 or payload.get('manifest_sha256') != canonical_json_sha256(unsigned):
        raise ValueError('invalid source IndexManifest SHA/schema')
    values = dict(payload)
    for label in ('qwen','colpali','colpali_backbone'):
        values[label+'_model'] = payload['resources'][label]['model']
        values[label+'_revision'] = payload['resources'][label]['revision']
    values['embedding_shape'] = tuple(payload['embeddings']['shape'])
    values['embedding_dtype'] = payload['embeddings']['dtype']
    original = IndexManifest(**{field.name:values[field.name] for field in fields(IndexManifest)})
    if original.to_dict() != payload:
        raise ValueError('source IndexManifest not canonical')
    replacements = {name:artifact_root/getattr(original,name).relative_to(original.artifact_root)
                    for name in ('embeddings_path','embedding_metadata_path','token2pageuid_path','index_path')}
    return replace(original, artifact_root=artifact_root, completion_ledger_path=ledger_path,
                   completion_ledger_sha256=sha256(ledger_path), **replacements)


def assemble(package: Path, output: Path, roots: list[Path], *, config: Path, run_config: Path, index_manifest: Path, case_ids: list[str] | None = None) -> dict:
    """Create a new H200 fixture using the established exact CPU PDF renderer."""
    from docprune.task6_runtime import (FixedPageFixture, FixedPageQuestion, FixedPageRecord,
                                       TASK6_RENDERER_CONTRACT, render_task6_pdf_page)
    package = package.resolve()
    for line in (package/'MANIFEST.sha256').read_text().splitlines():
        digest, relative = line.split('  ', 1)
        if sha256(package/relative) != digest:
            raise ValueError('package checksum mismatch: ' + relative)
    if output.exists():
        raise FileExistsError(output)
    corpus = json.loads((package/'corpus.json').read_text())
    if case_ids:
        wanted = set(case_ids)
        available = {c['case_id'] for c in corpus['cases']} | {c['qid'] for c in corpus['cases']}
        if wanted - available:
            raise ValueError('unknown case IDs: ' + str(sorted(wanted - available)))
        corpus['cases'] = [c for c in corpus['cases'] if c['case_id'] in wanted or c['qid'] in wanted]
    needed = {p['doc_id'] for c in corpus['cases'] for p in c['pages']}
    paths, failures = {}, []
    for asset in corpus['assets']:
        if asset['doc_id'] not in needed:
            continue
        try:
            paths[asset['doc_id'],asset['kind']] = resolve_asset(asset, package, roots)
        except (ValueError, FileNotFoundError, OSError) as error:
            failures.append(str(error))
    if failures:
        raise ValueError('Unresolved input assets (no fixture written):\n' + '\n'.join(failures))
    # Fail before creating an output directory if the exact renderer is absent.
    from pdf2image import convert_from_path  # noqa: F401
    from docprune.task6_runtime import TASK6_POPPLER_BIN, TASK6_PDFTOPPM_SHA256
    if sha256(TASK6_POPPLER_BIN/'pdftoppm') != TASK6_PDFTOPPM_SHA256:
        raise ValueError('assembly requires the pinned Task 6 pdftoppm binary')
    if not (TASK6_POPPLER_BIN/'pdfinfo').is_file():
        raise FileNotFoundError(TASK6_POPPLER_BIN/'pdfinfo')
    output = output.resolve()
    output.mkdir(parents=True)
    eligible = output/'eligible.jsonl'
    eligible.write_text(''.join(json.dumps({'qid':c['qid'], 'question':c['question'], 'answers':[{'answer':g} for g in c['gold_answers']], 'metadata':{'type':c['question_type'], 'modalities':c['modalities']}} ,ensure_ascii=False)+'\n' for c in corpus['cases']))
    reference = output/'reference.json'
    write_json(reference, {'selection_is_outcome_blind':False, 'fixed_page_selection_is_outcome_blind':False,
                          'selection_policy':'evidence-reviewed original or explicitly recovered pages; not fresh retrieval',
                          'question_ids':[c['qid'] for c in corpus['cases']],
                          'rows':{c['qid']:{'retrieved_pages':[{k:p[k] for k in ('doc_id','page_index','score')} for p in c['pages']]} for c in corpus['cases']}})
    ledger = [r for r in json.loads((package/'selected-source-ledger.json').read_text()) if r['doc_id'] in needed]
    for row in ledger:
        row['document_path'] = str(paths[row['doc_id'],'features'])
    ledger_path = output/'docprune'/'completion-ledger.json'
    write_json(ledger_path, ledger)
    # Preserve the full original IndexManifest scientific identity; relocate only
    # artifact paths and the selected completion ledger. Global index files are
    # deliberately absent and fixed-page loading never opens them.
    artifact_root = output/'docprune'
    rebased = rebase_index_manifest(index_manifest, artifact_root, ledger_path)
    feature_manifest = artifact_root/'manifest.json'
    write_json(feature_manifest, rebased.to_dict())
    hashes = {key:sha256(value) for key,value in paths.items()}
    rendered = {}
    questions=[]
    for case in corpus['cases']:
        pages=[]
        for rank,p in enumerate(case['pages']):
            key=p['doc_id'],p['page_index']
            pdf=paths[p['doc_id'],'pdf']
            if key not in rendered:
                im=render_task6_pdf_page(pdf,p['page_index']).convert('RGB')
                rendered[key]=(im.width,im.height,hashlib.sha256(im.tobytes()).hexdigest())
            width,height,rgbsha=rendered[key]
            pages.append(FixedPageRecord(rank=rank,doc_id=p['doc_id'],page_index=p['page_index'],score=float(p['score']),source_pdf_path=pdf,source_pdf_sha256=hashes[p['doc_id'],'pdf'],rendered_rgb_width=width,rendered_rgb_height=height,rendered_rgb_sha256=rgbsha,renderer_contract=TASK6_RENDERER_CONTRACT,feature_shard_path=paths[p['doc_id'],'features'],feature_shard_sha256=hashes[p['doc_id'],'features'],feature_page_index=p['page_index']))
        questions.append(FixedPageQuestion(case['qid'], hashlib.sha256(case['question'].encode()).hexdigest(),tuple(pages)))
    fixture=FixedPageFixture('correction-depth40-v1',reference,sha256(reference),eligible,sha256(eligible),feature_manifest,sha256(feature_manifest),ledger_path,sha256(ledger_path),tuple(questions))
    fixture.validate_external_bytes()
    fixture.selected_samples([c['qid'] for c in corpus['cases']])
    write_json(output/'fixture.json',fixture.to_dict())
    resources={key:{'path':str(path.resolve()),'sha256':sha256(path)} for key,path in [('config',config),('run_config',run_config),('index_manifest',feature_manifest),('fixture',output/'fixture.json')]}
    resources['mappings']={}
    write_json(output/'resources.json',resources)
    corpus['question_count'] = len(corpus['cases'])
    corpus['page_count'] = 4 * len(corpus['cases'])
    write_json(output/'corpus.json', corpus)
    return resources
