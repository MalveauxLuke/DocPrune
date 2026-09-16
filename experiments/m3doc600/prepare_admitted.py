"""Render only sealed admitted pages, reusing verified original-top4 renders."""
import argparse
import hashlib
import os
from pathlib import Path
from common import read, sha, fingerprint, publish


def run(pool, corpus, out, reuse_catalog=None):
    import pypdfium2 as pdfium
    if not os.environ.get('SLURM_JOB_ID'):
        raise RuntimeError('A compute allocation is required')
    pool, corpus, out = Path(pool), Path(corpus), Path(out)
    data = read(pool)
    questions = data['questions']
    assert len(questions) == 471 and len({q['qid'] for q in questions}) == 471
    reused = {}
    if reuse_catalog:
        old = read(reuse_catalog)
        digest = old.pop('sha256')
        assert fingerprint(old) == digest
        for page in old['pages']:
            for alias in page['aliases']:
                reused[(alias['document_id'], alias['page_number'] - 1)] = (page, alias)
    pages, identities, pdf_hashes, queries = {}, {}, {}, []
    for q in questions:
        admitted = [(p['doc_id'], p['page_index']) for p in q['admitted_pages']]
        original = [(p['doc_id'], p['page_index']) for p in q['original_top4']]
        evidence = [(p['doc_id'], p['page_index']) for p in q['evidence_pages']]
        assert admitted[:len(original)] == original and evidence
        assert set(evidence) <= set(admitted) and len(admitted) == len(set(admitted))
        links = []
        for doc, idx in admitted:
            assert isinstance(idx, int) and idx >= 0
            identity = (doc, idx)
            if identity not in identities:
                pdf_path = corpus / 'pdfs_dev' / (doc + '.pdf')
                if doc not in pdf_hashes:
                    pdf_hashes[doc] = sha(pdf_path)
                expected = next(p['pdf_sha256'] for p in q['admitted_pages']
                                if p['doc_id'] == doc and p['page_index'] == idx)
                assert pdf_hashes[doc] == expected, f'Reviewed PDF changed: {doc}'
                cached = reused.get(identity)
                if cached and cached[1]['pdf_sha256'] == pdf_hashes[doc]:
                    record, alias = cached
                    assert sha(record['image']) == record['image_sha256']
                    record = dict(record, aliases=[])
                else:
                    with pdfium.PdfDocument(str(pdf_path)) as pdf:
                        page = pdf[idx]
                        bitmap = page.render(scale=2)
                        im = bitmap.to_pil().convert('RGB')
                        w, h = im.size
                        pixel = hashlib.sha256(im.tobytes()).hexdigest()
                        key = fingerprint([w, h, pixel])
                        target = out / 'images' / (key + '.png')
                        target.parent.mkdir(parents=True, exist_ok=True)
                        if not target.exists():
                            im.save(target)
                        record = dict(key=key, image=str(target.resolve()), image_sha256=sha(target),
                                      rgb_sha256=pixel, width=w, height=h, aliases=[])
                        im.close()
                        bitmap.close()
                        page.close()
                    alias = dict(source='m3docvqa', document_id=doc, page_number=idx+1,
                                 pdf_sha256=pdf_hashes[doc], rendering='pypdfium2-144dpi-native-page')
                key = record['key']
                merged = pages.setdefault(key, record)
                if alias not in merged['aliases']:
                    merged['aliases'].append(alias)
                identities[identity] = key
            else:
                expected = next(p['pdf_sha256'] for p in q['admitted_pages']
                                if p['doc_id'] == doc and p['page_index'] == idx)
                assert pdf_hashes[doc] == expected, f'Conflicting reviewed PDF: {doc}'
            links.append(dict(key=identities[identity], page_id=f'{doc}:{idx}', page_number=idx+1))
        queries.append(dict(question_key=q['qid'], question=q['question'], source='m3docvqa',
                            pages=links, presentation_order=q['presentation_order']))
    catalog = dict(schema='docprune-m3doc600-pages-v1', manifest_sha256=sha(pool),
                   pool_path=str(pool.resolve()), stage='owner_approved_admitted471',
                   pages=list(pages.values()), questions=queries, smoke_keys=[], smoke_questions=[])
    catalog['sha256'] = fingerprint(catalog)
    publish(out / 'catalog.json', catalog)
    print({'questions': len(queries), 'unique_page_identities': len(identities),
           'unique_pixel_pages': len(pages), 'catalog_sha256': catalog['sha256']}, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pool', required=True)
    parser.add_argument('--corpus', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--reuse-catalog')
    args = parser.parse_args()
    run(args.pool, args.corpus, args.out, args.reuse_catalog)
