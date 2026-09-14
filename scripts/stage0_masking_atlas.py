"""Build a local, offline HTML atlas from the Stage 0 audit and visual notes."""
import argparse
import json
import os
from pathlib import Path


def main(root, audit, annotations):
    cases = json.loads((audit / 'cases.json').read_text())
    notes = json.loads(annotations.read_text())
    for case in cases:
        case['visual'] = notes['cases'][case['case']]
        baseline = json.loads((root / 'input/sources' / case['case'] / 'baseline.json').read_text())
        case['fixed_self'] = baseline['full_context_qwen']['answer']
        assert len(case['visual']['pages']) == len(case['pages']) == 4
        indices = {str(r['region_index']) for p in case['pages'] for r in p['regions']}
        assert set(case['visual']['regions']) <= indices
        for page in case['pages']:
            image = Path(page['image'])
            assert image.is_file(), image
            page['image'] = os.path.relpath(image, audit)
    template = Path(__file__).with_name('stage0_masking_atlas.html').read_text()
    payload = json.dumps(cases, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c')
    (audit / 'atlas.html').write_text(template.replace('__AUDIT_DATA__', payload))
    lines = ['# Stage 0 page and section review', '', notes['scope'], '']
    for case in cases:
        lines += [f"## {case['case']} — {case['question']}", '',
                  f"Gold: {case['gold']}; fixed self-answer: {case['fixed_self']}", '',
                  case['visual']['summary'], '']
        lines += [f"- **Input rank {i}:** {note}" for i, note in enumerate(case['visual']['pages'])]
        lines += ['', 'Selected sections:', '']
        lines += [f'- **Region {i}:** {note}' for i, note in case['visual']['regions'].items()]
        lines += ['']
    (audit / 'visual-review.md').write_text('\n'.join(lines))
    print(audit / 'atlas.html')


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, required=True)
    ap.add_argument('--audit', type=Path, required=True)
    ap.add_argument('--annotations', type=Path, default=Path(__file__).resolve().parents[1] /
                    'docs/experiments/corrective-selection/stage0_visual_notes.json')
    args = ap.parse_args()
    main(args.root, args.audit, args.annotations)
