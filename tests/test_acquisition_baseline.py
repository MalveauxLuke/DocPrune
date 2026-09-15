import tempfile
from pathlib import Path
from unittest import TestCase
from unittest.mock import patch
from types import SimpleNamespace
from contextlib import nullcontext
import sys
from test_adaptive_acquisition import fixture
from docprune.acquisition_runner import run_case, SyntheticReader
from docprune.acquisition_io import read_record
from docprune.acquisition_reader import QuestionCheckpoint

class BaselineTests(TestCase):
    def test_current_baseline_drives_controller_and_all_summaries(self):
        class Current(SyntheticReader):
            def __init__(self, case):
                super().__init__(case)
                self.baseline_likelihoods = [-4., -5.]
            def score(self, ids):
                return {'likelihoods': [-3., -6.]}
        with tempfile.TemporaryDirectory() as d:
            with patch('docprune.acquisition_runner.Controller', wraps=__import__('docprune.adaptive_acquisition', fromlist=['Controller']).Controller) as controller:
                result = run_case(fixture(), d, Current, Current.identity)
                self.assertEqual(controller.call_args.args[1], (-4., -5.))
            self.assertEqual(result['baseline_G_S'], [-4., -5.])
            for arm in ('R','A','adaptive'):
                winner = result['arms'][arm]['32']['winners']['G']
                self.assertEqual(winner['delta_G'], 1.)
                self.assertEqual(winner['delta_S'], -1.)
                self.assertTrue(winner['admissible'])
            record = read_record(Path(d)/'Q01/baseline.json')
            self.assertEqual(record['historical_likelihoods'], [-1., -2.])
            self.assertEqual(record['likelihoods'], [-4., -5.])

    def test_resume_rejects_changed_current_baseline(self):
        class Changed(SyntheticReader):
            def __init__(self, case):
                super().__init__(case)
                self.baseline_likelihoods = [-4., -5.]
        with tempfile.TemporaryDirectory() as d:
            run_case(fixture(), d, SyntheticReader, SyntheticReader.identity, max_new_observations=1)
            with self.assertRaisesRegex(ValueError, 'parity failed'):
                run_case(fixture(), d, Changed, Changed.identity)

    def test_guard_accepts_historical_shift_but_rejects_cached_scoring_error(self):
        import numpy as np
        q = QuestionCheckpoint.__new__(QuestionCheckpoint)
        q.population=10032
        q.case=fixture()
        q.case['reader']['expected_terminal_eos']=9
        q.targets=((1,), (2,))
        q.preparation_seconds=0
        q.guarded=False
        q.batch={'input_ids': np.zeros((1,2),dtype=int)}
        q.teacher=SimpleNamespace(runtime={'max_new_tokens':4, 'eos_ids':[9]}, processor=SimpleNamespace(decode=lambda ids, **kw: str(ids)))
        q.model=SimpleNamespace(generate=lambda **kw: np.array([[0,0,2,9]]))
        q._measure=lambda ids: {'likelihoods': [-4., -5.]}
        q._legacy_measure=lambda ids: [-4., -5.]
        torch=SimpleNamespace(inference_mode=nullcontext,cuda=SimpleNamespace(synchronize=lambda:None))
        with patch.dict(sys.modules, {'torch':torch}):
            q._guard()
            self.assertTrue(q.guarded)
            self.assertEqual(q.baseline_likelihoods, [-4., -5.])
            self.assertEqual(q.guard['historical_all_keep_error'], 3.)
            q.guarded=False
            q._legacy_measure=lambda ids: [-4.1,-5.]
            with self.assertRaisesRegex(ValueError,'parity failed'):
                q._guard()
            self.assertFalse(q.guarded)
