"""CPU checks for preparation resume identities and immutable source reuse."""
import runpy
import tempfile
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

PREP = runpy.run_path(str(Path(__file__).resolve().parents[1]/'examples/prepare_correction_depth.py'))


def test_existing_sealed_inputs_reject_changed_fixture():
    module = ModuleType('docprune.task8_runtime')
    module.seal_task8_smoke_inputs = lambda **kw: (_ for _ in ()).throw(AssertionError('must reuse'))
    module.load_task8_smoke_inputs = lambda p: {'qid': 'q', 'fixed_page_fixture_sha256': 'original'}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root/'inputs').mkdir()
        (root/'inputs'/'smoke-input-manifest.json').write_text('{}')
        with patch.dict('sys.modules', {'docprune.task8_runtime': module}):
            assert PREP['seal']({'qid': 'q'}, root/'fixture', 'original', root, 'a'*40).exists()
            try:
                PREP['seal']({'qid': 'q'}, root/'fixture', 'changed', root, 'a'*40)
            except ValueError:
                pass
            else:
                raise AssertionError('changed fixture accepted')


def test_completed_segmentation_reuses_authenticated_artifact_without_launch():
    module = ModuleType('docprune.task8_runtime')
    module.prepare_task8_mineru_smoke = lambda **kw: (_ for _ in ()).throw(AssertionError('must not prepare'))
    module.finalize_task8_mineru_smoke = lambda **kw: (_ for _ in ()).throw(AssertionError('must not finalize'))
    module.load_task8_mineru_completion = lambda p, **kw: {'run_manifest_path': str(p.parent/'run-manifest.json')}
    module._load_signed_run_manifest = lambda p: {'qid': 'q'}
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        (root/'mineru').mkdir()
        path = root/'mineru'/'completion-manifest.json'
        path.write_text('{}')
        with patch.dict('sys.modules', {'docprune.task8_runtime': module}):
            assert PREP['segment']({'qid': 'q'}, root/'smoke', root, 'a'*40,
                                  root/'template', root/'mineru-exe', {}) == path


def test_canonical_manifest_writer_does_not_replace():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'manifest.json'
        PREP['canonical_write'](path, {'b': 2, 'a': 1})
        assert path.read_text() == '{"a":1,"b":2}'
        try:
            PREP['canonical_write'](path, {'a': 3})
        except FileExistsError:
            pass
        else:
            raise AssertionError('immutable manifest replaced')
