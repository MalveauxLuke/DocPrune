"""Collect an explicitly audited panel through the unchanged pilot teacher path."""
import argparse
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'src'), str(REPO)]
from experiments.training_pilot.run import (
    teacher, read, sha, publish, load_catalog, load_labels, VERSION,
    RETRIEVAL_SCHEMA, REVISION, SELECTOR_REV, fingerprint,
)


def panel_context(args):
    root, out = Path(args.root), Path(args.output)
    manifest = read(args.manifest)
    panel = manifest['record']
    assert fingerprint(panel) == manifest['sha256']
    assert panel['schema'] == 'audited-error-panel-v1'
    assert panel['mask_count'] == 32
    rows = panel['records']
    qids = [r['question']['qid'] for r in rows]
    assert len(qids) == len(set(qids)) == 59
    labelpath = root / 'training-pilot-v1/labels-deepseek-v4.1-flash.json'
    assert sha(labelpath) == panel['labels_sha256']
    labels = load_labels(labelpath)
    cohort = root / 'training463-evidence-filtered-v1'
    assert sha(cohort / 'pool.json') == panel['pool_sha256']
    pool = {q['qid']: q for q in read(cohort / 'pool.json')['questions']}
    for row in rows:
        q = row['question']; label = labels[q['qid']]
        assert label['verdict'] == 'incorrect'
        assert label['question'] == q['question']
        assert label['gold'] == q['gold']
        assert label['model_answer'] == q['baseline_answer']
        assert pool[q['qid']]['admitted_pages'] == q['admitted_pages']
    catalog, digest = load_catalog(root)
    contract = dict(schema='audited-error-panel-collection-v1',
                    manifest_sha256=sha(args.manifest), catalog=digest,
                    reader=REVISION, selector=SELECTOR_REV, acquisition=VERSION,
                    retrieval_schema=RETRIEVAL_SCHEMA,
                    sources={str(p.relative_to(REPO)): sha(p) for p in (
                        Path(__file__), REPO/'experiments/training_pilot/run.py',
                        REPO/'src/docprune/stage2/pilot.py')})
    publish(out / 'contract.json', contract)
    jobs = dict(qids=qids, mask_count=32, dev_qids=[],
                test_empty_interface=False, require_pairs=False)
    # Historical development banks remain immutable. This panel is acquisition,
    # not a new held-out evaluation split; every case uses the same 32-mask policy.
    return root, out, catalog, pool, labels, jobs, contract


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    for name in ('root', 'output', 'reader', 'manifest'):
        parser.add_argument('--' + name, required=True)
    args = parser.parse_args()
    import torch
    import transformers
    assert transformers.__version__ == '4.57.3' and torch.cuda.is_available()
    torch.set_num_threads(2)
    teacher(args, context_fn=panel_context)
