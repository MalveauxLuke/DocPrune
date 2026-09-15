"""Validate completed smoke artifacts before admitting production arrays."""
import argparse
from pathlib import Path
from common import load_catalog,read,publish,sha,fingerprint,pages_for,query_for

def validate(root,receipt):
 if receipt.get('schema')=='docprune-initial300-region-admission-v2':return validate_region(root,receipt)
 root=Path(root);c,digest=load_catalog(root)
 assert receipt['catalog_sha256']==digest and receipt['visual_review']=='passed'
 for engine in ('colqwen','mineru'):
  contract=read(root/engine/'contract.json')
  assert contract['sha256']==receipt['contracts'][engine]
  run=read(root/receipt['runs'][engine])
  assert run['smoke'] and run['contract_sha256']==contract['sha256']
  assert run['requested_pages']==len(c['smoke_keys'])
  for p in pages_for(c,0,1,True):
   r=read(root/engine/'pages'/f"{p['key']}.json")
   assert r['image_sha256']==p['image_sha256']
   if engine=='colqwen':
    assert r['contract_sha256']==contract['sha256']
    assert sha(root/engine/'pages'/f"{p['key']}.safetensors")==r['tensor_sha256']
   else:assert r['provenance']['contract_sha256']==contract['sha256']
  if engine=='colqwen':
   assert run['adapter_check']['status']=='passed' and run['adapter_check']['verified_tensors']==506
   assert len(run['batch_singleton_checks'])>=min(4,len(c['smoke_keys']))
   assert all(x['min_cosine']>.995 for x in run['batch_singleton_checks'])
   for q in query_for(c,0,1,True):
    key=fingerprint(q['question_key']);r=read(root/engine/'queries'/f'{key}.json')
    assert r['contract_sha256']==contract['sha256'] and r['question_sha256']==fingerprint(q['question'])
    assert sha(root/engine/'queries'/f'{key}.safetensors')==r['tensor_sha256']
 return receipt


def validate_region(root,receipt):
 root=Path(root);c,digest=load_catalog(root)
 assert receipt['catalog_sha256']==digest and receipt['visual_review']=='passed'
 assert receipt['owner_accepted_batching'] is True
 for engine in ('colqwen','mineru'):
  contract=read(root/engine/'contract.json')
  assert contract['sha256']==receipt['contracts'][engine]
 col=read(root/'colqwen/contract.json')
 assert col['code_sha256']==sha(Path(__file__).with_name('colqwen.py'))
 diag_path=root/receipt['diagnostic'];assert sha(diag_path)==receipt['diagnostic_sha256']
 diag=read(diag_path)
 assert diag['catalog_sha256']==digest
 assert diag['code_sha256']==sha(Path(__file__).with_name('diagnose_colqwen_batch.py'))
 assert diag['adapter']['status']=='passed' and diag['adapter']['verified_tensors']==506
 expected=pages_for(c,0,1,True)[:4]
 assert [r['page_key'] for r in diag['pages']]==[p['key'] for p in expected]
 for row,p in zip(diag['pages'],expected):
  assert row['image_sha256']==p['image_sha256'] and row['same_token_ids'] and row['same_image_grid']
  region=row['region_comparison'];assert region['rankings']['single'][:5]==region['rankings']['batch'][:5]
  assert region['max_rank_shift']<=1
  assert sha(root/'mineru/pages'/f"{p['key']}.json")==region['layout_sha256']
 run_path=root/receipt['miner_run'];assert sha(run_path)==receipt['miner_run_sha256']
 run=read(run_path)
 assert run['smoke'] and run['contract_sha256']==receipt['contracts']['mineru']
 assert run['requested_pages']==len(c['smoke_keys']) and not run['empty_layout_pages']
 for p in pages_for(c,0,1,True):
  r=read(root/'mineru/pages'/f"{p['key']}.json")
  assert r['image_sha256']==p['image_sha256'] and r['provenance']['contract_sha256']==receipt['contracts']['mineru']
 return receipt

def main(a):
 root=Path(a.root)
 if a.check:
  validate(root,read(a.check));print('Smoke admission verified');return
 assert a.visual_review=='passed'
 c,digest=load_catalog(root)
 if a.region_job:
  diag=f'diagnostics/regions-{a.region_job}/completion.json';miner=f'mineru/runs/smoke-{a.job}.json'
  record={'schema':'docprune-initial300-region-admission-v2','catalog_sha256':digest,'visual_review':'passed','owner_accepted_batching':True,'acceptance_basis':'Owner approved exploratory preprocessing after stable ordered top five on four fixed pages; token-level equivalence not claimed','diagnostic':diag,'diagnostic_sha256':sha(root/diag),'miner_run':miner,'miner_run_sha256':sha(root/miner),'contracts':{e:read(root/e/'contract.json')['sha256'] for e in ('colqwen','mineru')}}
  validate(root,record);publish(root/'region-admission.json',record);print('Region-based production admission created');return
 runs={e:str((root/e/'runs'/f'smoke-{a.job}.json').relative_to(root)) for e in ('colqwen','mineru')}
 record={'schema':'docprune-initial300-smoke-admission-v1','catalog_sha256':digest,'visual_review':'passed','job':a.job,'runs':runs,'contracts':{e:read(root/e/'contract.json')['sha256'] for e in runs}}
 validate(root,record);publish(root/'smoke-admission.json',record);print('Production admission created')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--check');p.add_argument('--job');p.add_argument('--region-job');p.add_argument('--visual-review',choices=['passed']);main(p.parse_args())
