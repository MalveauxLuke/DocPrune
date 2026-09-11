"""Gather only explicitly referenced PDF/OCR/provenance files when locally available."""
import argparse
import json
import shutil
from pathlib import Path
from colfeatures_package import sha


def gather(packet, destination):
    destination.mkdir(parents=True, exist_ok=False)
    rows = []
    for case_dir in sorted((packet / 'sources').glob('Q[0-9][0-9]')):
        case = json.loads((case_dir / 'case.json').read_text())
        mapping = json.loads((case_dir / 'mapping.json').read_text())
        candidates = []
        for old, page in zip(case['original_pages'], case['pages']):
            candidates.append((old['source_pdf'], page.get('source_pdf_sha256')))
        for artifact in mapping['artifacts']:
            for key, value in artifact.items():
                if key.endswith('_path') and isinstance(value, str):
                    candidates.append((value, artifact.get(key[:-5] + '_sha256')))
        for source, expected in candidates:
            p = Path(source)
            row = {'case': case_dir.name, 'source': source, 'expected_sha256': expected}
            if not p.is_file():
                row['status'] = 'unavailable_on_this_host'
            elif not expected:
                row['status'] = 'available_but_no_recorded_hash'
            else:
                actual = sha(p)
                if actual != expected:
                    raise ValueError(f'Referenced artifact changed: {source}')
                name = actual + '--' + p.name
                target = destination / 'files' / name
                target.parent.mkdir(exist_ok=True)
                if not target.exists():
                    shutil.copyfile(p, target)
                row.update(status='gathered', sha256=actual, saved=str(target.relative_to(destination)))
            rows.append(row)
    (destination / 'availability.json').write_text(json.dumps(rows, indent=2) + '\n')
    print(json.dumps({'references': len(rows), 'gathered': sum(r['status'] == 'gathered' for r in rows)}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('packet', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    gather(args.packet, args.destination)
