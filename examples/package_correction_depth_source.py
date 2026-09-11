#!/usr/bin/env python3
"""Export a focused working-tree runtime snapshot; never alter a live checkout."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile


def build(repo: Path, output: Path) -> Path:
    repo, output = repo.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(output)
    files = [p for p in (repo / 'src/docprune').rglob('*.py')]
    files += [p for p in (repo / 'h200/correction-depth').rglob('*') if p.is_file()]
    files += [repo / 'configs/docprune-m3docvqa.toml', repo / 'pyproject.toml', repo / 'LICENSE']
    files += [repo / 'examples' / name for name in (
        'package_correction_depth.py', 'run_correction_depth.py',
        'prepare_correction_depth.py', 'package_correction_depth_source.py')]
    files += [repo / 'tests' / name for name in (
        'test_correction_corpus.py', 'test_correction_depth.py',
        'test_correction_scoring.py', 'test_correction_preparation.py')]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(path)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repo, text=True).strip()
    output.mkdir(parents=True)
    for path in sorted(files):
        target = output / path.relative_to(repo)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
    (output / 'AGENTS.md').write_text(
        (output / 'h200/correction-depth/AGENTS.md').read_text().replace(
            'Read `HANDOFF.md` in this directory.',
            'Read `h200/correction-depth/HANDOFF.md`.'))
    (output / 'README.md').write_text('Read [the correction-depth handoff](h200/correction-depth/HANDOFF.md).\n')
    identity = {'schema': 'correction-depth-source-v1', 'parent_git_revision': revision,
                'revision_is_parent_not_clean_snapshot': True,
                'files': {str(p.relative_to(output)): hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(output.rglob('*')) if p.is_file()}}
    (output / 'SOURCE_IDENTITY.json').write_text(json.dumps(identity, indent=2) + '\n')
    paths = sorted(p for p in output.rglob('*') if p.is_file())
    (output / 'MANIFEST.sha256').write_text(''.join(
        f'{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.relative_to(output)}\n' for p in paths))
    archive = output.with_suffix('.tar.gz')
    with tarfile.open(archive, 'w:gz') as stream:
        stream.add(output, arcname=output.name)
    archive.with_suffix(archive.suffix + '.sha256').write_text(
        hashlib.sha256(archive.read_bytes()).hexdigest() + '  ' + archive.name + '\n')
    return archive


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', type=Path, default=Path.cwd())
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(build(args.repo, args.output))
