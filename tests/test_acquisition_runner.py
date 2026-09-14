import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from test_adaptive_acquisition import fixture
from docprune.acquisition_io import digest, read_record, write_new, validate_case
from docprune.acquisition_runner import run_case, run_lock, SyntheticReader


class RunnerTests(unittest.TestCase):
    def test_interrupted_resume_matches_uninterrupted_without_repeating_scores(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            case = fixture()
            calls = []
            class Counting(SyntheticReader):
                def score(self, ids):
                    calls.append(tuple(ids))
                    return super().score(ids)
            partial = run_case(case, root/'resumed', Counting, Counting.identity, max_new_observations=37)
            self.assertFalse(partial['complete'])
            resumed = run_case(case, root/'resumed', Counting, Counting.identity)
            self.assertEqual(len(calls), len(set(calls)))
            self.assertEqual(len(calls), resumed['unique_physical_masks'])
            complete = run_case(case, root/'clean', SyntheticReader, SyntheticReader.identity)
            self.assertEqual(resumed, complete)
            for arm in ('R', 'A', 'adaptive'):
                for slot in range(32):
                    self.assertEqual(read_record(root/f'resumed/Q01/{arm}/{slot:02}.request.json'),
                                     read_record(root/f'clean/Q01/{arm}/{slot:02}.request.json'))
            self.assertEqual(resumed['logical_observations'], {'R':32, 'A':32, 'adaptive':32})
            self.assertLess(resumed['unique_physical_masks'], 96)

    def test_crash_after_physical_cache_before_logical_event_reuses_completed_score(self):
        from docprune import acquisition_runner as module
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            calls = []
            class Counting(SyntheticReader):
                def score(self, ids):
                    calls.append(tuple(ids))
                    return super().score(ids)
            original = module.write_new
            def interrupted(path, value):
                if str(path).endswith('00.result.json'):
                    raise RuntimeError('simulated interruption')
                original(path, value)
            with patch.object(module, 'write_new', interrupted), self.assertRaisesRegex(RuntimeError, 'interruption'):
                run_case(fixture(), root, Counting, Counting.identity)
            self.assertEqual(len(calls), 1)
            run_case(fixture(), root, Counting, Counting.identity, max_new_observations=1)
            self.assertEqual(len(calls), 1)

    def test_identity_drift_rejects_resume(self):
        with tempfile.TemporaryDirectory() as d:
            run_case(fixture(), d, SyntheticReader, SyntheticReader.identity, max_new_observations=1)
            changed = fixture()
            changed['public']['priority'][0] += 1
            with self.assertRaisesRegex(ValueError, 'Conflicting immutable'):
                run_case(changed, d, SyntheticReader, SyntheticReader.identity)
            with self.assertRaisesRegex(ValueError, 'Conflicting immutable'):
                run_case(fixture(), d, SyntheticReader, {'synthetic': True, 'name': 'different'})

    def test_corrupt_physical_cache_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            run_case(fixture(), d, SyntheticReader, SyntheticReader.identity, max_new_observations=1)
            path = next((Path(d)/'Q01/physical').glob('*.json'))
            content = json.loads(path.read_text())
            content['G'] += 1
            path.write_text(json.dumps(content))
            with self.assertRaisesRegex(ValueError, 'Corrupt record'):
                run_case(fixture(), d, SyntheticReader, SyntheticReader.identity)

    def test_stable_replay_never_calls_reader_again(self):
        with tempfile.TemporaryDirectory() as d:
            first = run_case(fixture(), d, SyntheticReader, SyntheticReader.identity)
            def forbidden(_):
                raise AssertionError('Completed cache must not instantiate reader')
            second = run_case(fixture(), d, forbidden, SyntheticReader.identity)
            self.assertEqual(first, second)

    def test_nonfinite_score_does_not_publish_measurement(self):
        class Broken(SyntheticReader):
            def score(self, ids):
                return {'likelihoods': [float('nan'), 0]}
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(ValueError):
                run_case(fixture(), d, Broken, Broken.identity)
            self.assertEqual(list(Path(d).rglob('*.result.json')), [])
            self.assertEqual(list((Path(d)/'Q01/physical').glob('*')), [])

    def test_input_and_atomic_record_contracts(self):
        bad = fixture()
        bad['public']['owners'][0] = 1
        with self.assertRaisesRegex(ValueError, 'costs disagree'):
            validate_case(bad)
        bad = fixture()
        bad['reader']['expected_generated_ids'] = [99]
        with self.assertRaisesRegex(ValueError, 'actual original'):
            validate_case(bad)
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)/'record.json'
            write_new(path, {'value': 1})
            write_new(path, {'value': 1})
            with self.assertRaises(ValueError):
                write_new(path, {'value': 2})
            with run_lock(d):
                with self.assertRaises(RuntimeError):
                    with run_lock(d):
                        pass


if __name__ == '__main__':
    unittest.main()
