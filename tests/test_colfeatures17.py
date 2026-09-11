"""Portable tests for transfer integrity and spatial correspondence."""
import io
import json
import sys
import tarfile
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / 'scripts'
sys.path.insert(0, str(SCRIPTS))
from colfeatures_package import pack, sha, unpack
from extract_colfeatures17 import region_membership


def test_transfer_roundtrip_and_corruption(tmp_path):
    source = tmp_path / 'source'
    source.mkdir()
    (source / 'image.png').write_bytes(bytes(range(256)) * 8)
    (source / 'nested').mkdir()
    (source / 'nested' / 'raw.json').write_text('{"gold":1,"self":2}')
    bundle = tmp_path / 'bundle'
    manifest = pack(source, bundle, chunk_bytes=128)
    assert len(manifest['chunks']) > 1
    restored = tmp_path / 'restored'
    unpack(bundle, restored)
    assert (restored / 'image.png').read_bytes() == (source / 'image.png').read_bytes()
    first = bundle / manifest['chunks'][0]['path']
    first.write_bytes(b'corrupt')
    with pytest.raises(ValueError, match='Corrupt chunk'):
        unpack(bundle, tmp_path / 'invalid')
    assert not (tmp_path / 'invalid').exists()


def test_archive_traversal_rejected(tmp_path):
    bundle = tmp_path / 'bundle'
    bundle.mkdir()
    archive = bundle / 'payload.part0000'
    with tarfile.open(archive, 'w:gz') as tar:
        member = tarfile.TarInfo('../escape')
        member.size = 1
        tar.addfile(member, io.BytesIO(b'x'))
    manifest = {'chunks': [{'path': archive.name, 'bytes': archive.stat().st_size,
                           'sha256': sha(archive)}], 'archive_sha256': sha(archive), 'files': []}
    (bundle / 'manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match='Unsafe archive'):
        unpack(bundle, tmp_path / 'invalid')
    assert not (tmp_path / 'escape').exists()


def test_overlapping_regions_preserve_many_to_many_and_empty():
    sources = [{'input_page_index': 0, 'page_size': [100, 200], 'bbox': [0, 0, 50, 200]},
               {'input_page_index': 0, 'page_size': [100, 200], 'bbox': [25, 0, 75, 100]},
               {'input_page_index': 0, 'page_size': [100, 200], 'bbox': [100, 200, 100, 200]},
               {'input_page_index': 1, 'page_size': [100, 200], 'bbox': [0, 0, 100, 200]}]
    regions, boxes, overlap = region_membership(sources, 0, 2, 2)
    assert len(regions) == 3
    np.testing.assert_allclose(boxes[2], [0, .5, .5, 1])
    np.testing.assert_allclose(overlap, [[1, 0, 1, 0], [.5, .5, 0, 0], [0, 0, 0, 0]])
