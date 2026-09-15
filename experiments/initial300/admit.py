"""Validate completed smoke artifacts before admitting production arrays."""
import argparse
from pathlib import Path
from common import load_catalog,read,publish,sha,fingerprint,pages_for,query_for

def validate(root,receipt):
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

def main(a):
 root=Path(a.root)
 if a.check:
  validate(root,read(a.check));print('Smoke admission verified');return
 assert a.visual_review=='passed'
 c,digest=load_catalog(root)
 runs={e:str((root/e/'runs'/f'smoke-{a.job}.json').relative_to(root)) for e in ('colqwen','mineru')}
 record={'schema':'docprune-initial300-smoke-admission-v1','catalog_sha256':digest,'visual_review':'passed','job':a.job,'runs':runs,'contracts':{e:read(root/e/'contract.json')['sha256'] for e in runs}}
 validate(root,record);publish(root/'smoke-admission.json',record);print('Production admission created')
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--check');p.add_argument('--job');p.add_argument('--visual-review',choices=['passed']);main(p.parse_args())
