"""Lossless, hashed, chunked Git transfer; Python standard library only."""
import argparse
import hashlib
import json
import shutil
import tarfile
import tempfile
from pathlib import Path

CHUNK_BYTES = 40 * 1024 * 1024


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def inventory(root):
    root = Path(root)
    rows = []
    for p in sorted(root.rglob('*')):
        if p.is_symlink():
            raise ValueError(f'Symlink is not an artifact: {p}')
        if p.is_file():
            rows.append({'path': p.relative_to(root).as_posix(),
                         'bytes': p.stat().st_size, 'sha256': sha(p)})
    return rows


def verify(root, rows):
    if inventory(root) != rows:
        raise ValueError('File inventory or content hash mismatch')


def pack(source, destination, chunk_bytes=CHUNK_BYTES):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination == source or source in destination.parents:
        raise ValueError('Destination must be outside source')
    if not 0 < chunk_bytes <= CHUNK_BYTES:
        raise ValueError('Invalid chunk size')
    rows = inventory(source)
    if not rows:
        raise ValueError('Empty source')
    destination.mkdir(parents=True, exist_ok=False)
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / 'payload.tar.gz'
        with tarfile.open(archive, 'w:gz') as tar:
            for row in rows:
                tar.add(source / row['path'], arcname=row['path'], recursive=False)
        verify(source, rows)
        chunks = []
        with archive.open('rb') as stream:
            while block := stream.read(chunk_bytes):
                p = destination / f'payload.part{len(chunks):04d}'
                p.write_bytes(block)
                chunks.append({'path': p.name, 'bytes': len(block), 'sha256': sha(p)})
        manifest = {'schema': 1, 'format': 'concatenated-tar-gzip', 'files': rows,
                    'chunks': chunks, 'archive_sha256': sha(archive),
                    'archive_bytes': archive.stat().st_size}
        (destination / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return manifest


def unpack(source, destination):
    source, destination = Path(source), Path(destination)
    if destination.exists():
        raise ValueError('Refusing to replace an existing destination')
    manifest = json.loads((source / 'manifest.json').read_text())
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        archive = Path(temporary) / 'payload.tar.gz'
        with archive.open('wb') as out:
            for row in manifest['chunks']:
                if Path(row['path']).name != row['path']:
                    raise ValueError('Invalid chunk path')
                p = source / row['path']
                if p.stat().st_size != row['bytes'] or sha(p) != row['sha256']:
                    raise ValueError(f'Corrupt chunk: {p}')
                with p.open('rb') as stream:
                    shutil.copyfileobj(stream, out)
        if sha(archive) != manifest['archive_sha256']:
            raise ValueError('Archive hash mismatch')
        extracted = Path(temporary) / 'extracted'
        extracted.mkdir()
        with tarfile.open(archive, 'r:gz') as tar:
            names = set()
            for member in tar.getmembers():
                p = Path(member.name)
                if not member.isfile() or p.is_absolute() or '..' in p.parts or member.name in names:
                    raise ValueError(f'Unsafe archive member: {member.name}')
                names.add(member.name)
                target = extracted / p
                target.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(member) as src, target.open('xb') as dst:
                    shutil.copyfileobj(src, dst)
        verify(extracted, manifest['files'])
        extracted.rename(destination)
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('operation', choices=['pack', 'unpack'])
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    result = globals()[args.operation](args.source, args.destination)
    print(json.dumps({'files': len(result['files']), 'chunks': len(result['chunks']),
                      'archive_bytes': result['archive_bytes']}))
