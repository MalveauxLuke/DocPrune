from contextlib import nullcontext
import sys
from types import SimpleNamespace, ModuleType
import unittest
from unittest.mock import patch
from docprune.acquisition_reader import (QuestionCheckpoint, validate_contract, validate_generation,
    validate_parity, validate_retained, validate_trace)


class ReaderTests(unittest.TestCase):
    def test_parity_and_generation_fail_closed(self):
        self.assertEqual(validate_parity([-1, -2], [-1, -2]), 0)
        for actual in ([float('nan'), -2], [-1], [-1.01, -2]):
            with self.assertRaises(ValueError):
                validate_parity(actual, [-1, -2])
        reader = {'expected_generated_ids': [32,450,273], 'expected_terminal_eos':151645}
        validate_generation([32,450,273], 151645, reader)
        for ids, eos in (([3959],151645), ([32,450,273],151643)):
            with self.assertRaises(ValueError):
                validate_generation(ids, eos, reader)

    def test_prompt_grid_and_original_token_contract(self):
        reader = {'prompt_sha256': 'p', 'input_ids_sha256': 'i'}
        runtime = {'processor': {'grid_thw':[1,114,88]}}
        validate_contract(reader, prompt_sha256='p', input_ids_sha256='i', grids=[[1,114,88]]*4, runtime=runtime)
        with self.assertRaises(ValueError):
            validate_contract(reader, prompt_sha256='p', input_ids_sha256='changed', grids=[[1,114,88]]*4, runtime=runtime)
        with self.assertRaises(ValueError):
            validate_contract(reader, prompt_sha256='p', input_ids_sha256='i', grids=[[1,88,114]]*4, runtime=runtime)
        validate_retained([0,3], 4, 2)
        for ids in ([0,0], [3,0], [0,4], [0]):
            with self.assertRaises(ValueError):
                validate_retained(ids, 4, 2)

    def test_adapter_invokes_frozen_checkpoint_and_exact_targets(self):
        calls = []
        checkpoint = object()
        targets = ((7,8), (9,), (32,450,273))
        trace = {'boundary':'B_input', 'mode':'physical_delete', 'visual_population':4,
                 'requested_budget':2, 'achieved_budget':2, 'retained_visual_ids':[0,3]}
        def resume(model, supplied_checkpoint, intervention):
            self.assertIs(supplied_checkpoint, checkpoint)
            self.assertEqual(intervention.boundary, 'input')
            self.assertEqual(intervention.mode, 'physical_delete')
            self.assertEqual(intervention.retained_visual_ids, (0,3))
            calls.append('resume')
            return SimpleNamespace(forced=SimpleNamespace(to_dict=lambda: trace))
        def teacher(model, head, prefill, supplied_targets):
            self.assertEqual(supplied_targets, targets)
            calls.append('teacher')
            return (-3., -1., -2.)
        torch = SimpleNamespace(inference_mode=nullcontext, cuda=SimpleNamespace(synchronize=lambda: None, max_memory_allocated=lambda: 123))
        decoder = SimpleNamespace(ForcedVisualIntervention=lambda **kw: SimpleNamespace(**kw), resume_forced_boundary_checkpoint=resume)
        model_module = SimpleNamespace(teacher_forced_sequence_loglikelihoods=teacher)
        q = QuestionCheckpoint.__new__(QuestionCheckpoint)
        q.model = SimpleNamespace(model=object(), lm_head=object())
        q.checkpoint, q.targets, q.population, q.budget = checkpoint, targets, 4, 2
        q.guarded, q.guard = True, {'test':True}
        with patch.dict(sys.modules, {'torch':torch, 'docprune.qwen2vl.decoder':decoder, 'docprune.qwen2vl.model':model_module}):
            result = q.score([0,3])
            self.assertEqual(result['likelihoods'], [-3., -1., -2.])
            self.assertEqual(calls, ['resume', 'teacher'])
            trace['boundary'] = 'B13'
            with self.assertRaises(ValueError):
                q.score([0,3])
            q.guarded = False
            with self.assertRaises(RuntimeError):
                q.score([0,3])


    def test_gpu_entrypoint_requires_authority_paths_and_idle_coral_device(self):
        import importlib.util
        from pathlib import Path
        spec = importlib.util.spec_from_file_location('stage0_test_cli', Path(__file__).resolve().parents[1]/'scripts/stage0_acquisition.py')
        cli = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cli)
        args = SimpleNamespace(execute_gpu=False, physical_gpu=4, coral_channel_or_relay=True,
                               package=Path('/mnt/data1/eunwooim/DocPrune/outputs/inputs'),
                               output=Path('/mnt/data1/eunwooim/DocPrune/outputs/test'))
        with patch.object(cli.subprocess, 'check_output') as query:
            with self.assertRaisesRegex(ValueError, '--execute-gpu'):
                cli.gpu_preflight(args)
            query.assert_not_called()
        args.execute_gpu = True
        args.physical_gpu = 0
        with self.assertRaisesRegex(ValueError, 'CoRAL'):
            cli.gpu_preflight(args)
        args.physical_gpu = 4
        env = {k: '/mnt/data2/eunwooim/test' for k in ('HF_HOME','TRANSFORMERS_CACHE','PIP_CACHE_DIR',
               'UV_CACHE_DIR','TORCH_HOME','XDG_CACHE_HOME','CONDA_PKGS_DIRS','TMPDIR')}
        with patch.dict(cli.os.environ, env), patch.object(cli.subprocess, 'check_output', side_effect=['GPU-test, NVIDIA H200, 100, 0', '123']):
            with self.assertRaisesRegex(RuntimeError, 'occupied'):
                cli.gpu_preflight(args)
        with patch.dict(cli.os.environ, env), patch.object(cli.subprocess, 'check_output', side_effect=['GPU-test, NVIDIA H200, 0, 0', '']):
            self.assertEqual(cli.gpu_preflight(args)['physical_gpu'], 4)


if __name__ == '__main__':
    unittest.main()
