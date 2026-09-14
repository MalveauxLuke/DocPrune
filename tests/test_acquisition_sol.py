import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from docprune.acquisition_sol import sol_preflight
class SOLTests(unittest.TestCase):
 def setUp(self):
  self.args=SimpleNamespace(command='smoke',physical_gpu=None,package=Path('/scratch/test/inputs'),output=Path('/scratch/test/output'))
  self.env={k:'/scratch/test/cache' for k in ('HF_HOME','TRANSFORMERS_CACHE','PIP_CACHE_DIR','UV_CACHE_DIR','TORCH_HOME','XDG_CACHE_HOME','CONDA_PKGS_DIRS','TMPDIR')}
  self.env.update(SLURM_JOB_ID='123',CUDA_VISIBLE_DEVICES='2')
  self.user=patch('docprune.acquisition_sol.pwd.getpwuid',return_value=SimpleNamespace(pw_name='test')); self.user.start(); self.addCleanup(self.user.stop)
 def test_requires_slurm_and_scratch(self):
  with patch.dict(os.environ,{},clear=True):
   with self.assertRaises(ValueError): sol_preflight(self.args)
  with patch.dict(os.environ,self.env,clear=True):
   self.args.output=Path('/home/test/output')
   with self.assertRaisesRegex(ValueError,'scratch'): sol_preflight(self.args)
 def test_rejects_lightwork_wrong_node_and_occupied(self):
  valid='JobState=RUNNING Partition=htc UserId=test(123) AllocTRES=cpu=2,gres/gpu=1 NodeList=gpu01'
  for responses in [[valid.replace('htc','lightwork')],[valid,'othernode'],[valid,'gpu01','GPU-abc, NVIDIA H100, 0, 81000, 0','1234']]:
   with patch.dict(os.environ,self.env,clear=True),patch('docprune.acquisition_sol.socket.gethostname',return_value='gpu01'),patch('docprune.acquisition_sol.subprocess.check_output',side_effect=responses):
    with self.assertRaises((ValueError,RuntimeError)): sol_preflight(self.args)
 def test_preserves_slurm_device_and_rejects_full_scoring(self):
  responses=['JobState=RUNNING Partition=htc UserId=test(123) AllocTRES=cpu=2,gres/gpu=1 NodeList=gpu01','gpu01','GPU-abc, NVIDIA H100, 0, 81000, 0','']
  with patch.dict(os.environ,self.env,clear=True),patch('docprune.acquisition_sol.socket.gethostname',return_value='gpu01'),patch('docprune.acquisition_sol.subprocess.check_output',side_effect=responses):
   result=sol_preflight(self.args); self.assertEqual(result['assigned_device'],'2'); self.assertEqual(os.environ['CUDA_VISIBLE_DEVICES'],'2')
   self.args.command='score'
   with self.assertRaisesRegex(ValueError,'one-question'): sol_preflight(self.args)
if __name__=='__main__': unittest.main()
