"""Explicitly combine owner-approved batch-four and batch-eight layout sources."""
from pathlib import Path
from common import read,fingerprint

def contracts(root):
 root=Path(root);result={}
 for name in ['mineru','mineru-batch8']:
  path=root/name/'contract.json'
  if path.exists():
   c=read(path);body={k:v for k,v in c.items() if k!='sha256'}
   assert fingerprint(body)==c['sha256']
   result[name]=c
 assert 'mineru' in result
 if 'mineru-batch8' in result:
  a,b=result['mineru'],result['mineru-batch8']
  assert a['batch_size']==4 and b['batch_size']==8
  excluded={'batch_size','code_sha256','sha256'}
  assert {k:v for k,v in a.items() if k not in excluded}=={k:v for k,v in b.items() if k not in excluded}
 return result

def page(root,p,cs):
 found=[]
 for name,c in cs.items():
  path=Path(root)/name/'pages'/(p['key']+'.json')
  if path.exists():
   row=read(path)
   assert row['image_sha256']==p['image_sha256']
   assert row['provenance']['contract_sha256']==c['sha256']
   found.append(row)
 assert len(found)==1, f"Expected exactly one layout for {p['key']}, got {len(found)}"
 return found[0]
