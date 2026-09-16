"""Reusable page/query vectors; no answerer, token pruning, or training."""
import argparse,time,sys,os,json
from pathlib import Path
from common import *

def run(a):
 import torch
 import numpy as np
 from safetensors.torch import save_file,load_file
 from colpali_engine.models import ColQwen2_5,ColQwen2_5_Processor
 sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
 from extract_colfeatures17 import load_verified_adapter
 from importlib.metadata import version
 assert version('colpali-engine')=='0.3.12' and version('transformers')=='4.53.3'
 assert Path(a.base).name==BASE[1] and Path(a.adapter).name==ADAPTER[1]
 assert torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0]>=8
 torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats();start=time.monotonic()
 root=Path(a.root);catalog,digest=load_catalog(root)
 model=ColQwen2_5.from_pretrained(a.base,torch_dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa',local_files_only=True)
 adapter_check=load_verified_adapter(model,a.adapter);model.eval()
 processor=ColQwen2_5_Processor.from_pretrained(a.adapter,local_files_only=True);processor.query_prefix='Query: '
 assert processor.image_processor.max_pixels==602112
 contract={'schema':'docprune-colqwen-cache-v1','catalog_sha256':digest,'base':BASE,'adapter':ADAPTER,'max_pixels':602112,'dtype':'bfloat16','storage':'float32','attention':'sdpa','query_prefix':'Query: ','query_augmentation':'<|endoftext|>'*10,'code_sha256':sha(__file__),'common_sha256':sha(Path(__file__).with_name('common.py')),'adapter_loader_sha256':sha(Path(__file__).resolve().parents[2]/'scripts/extract_colfeatures17.py'),'torch':torch.__version__,'transformers':version('transformers'),'batch_size':a.batch}
 identity=fingerprint(contract);target=root/'colqwen';target.mkdir(parents=True,exist_ok=True)
 publish(target/'contract.json',dict(contract,sha256=identity))
 def complete(kind,key):
  p=target/kind/(key+'.json')
  if not p.exists():return False
  r=read(p);assert r['contract_sha256']==identity
  assert sha(target/kind/(key+'.safetensors'))==r['tensor_sha256']
  return True
 def write(kind,key,tensors,metadata):
  d=target/kind;d.mkdir(exist_ok=True);f=d/(key+'.safetensors')
  if f.exists():raise ValueError(f'Orphan tensor needs inspection: {f}')
  tmp=d/(key+'.tmp-'+str(os.getpid()))
  save_file({k:v.contiguous().cpu() for k,v in tensors.items()},str(tmp));os.link(tmp,f);tmp.unlink()
  publish(d/(key+'.json'),dict(metadata,contract_sha256=identity,tensor_sha256=sha(f)))
 todo=[p for p in pages_for(catalog,a.shard,a.shards,a.smoke) if not complete('pages',p['key'])]
 parity=[];count=0;batch_timings=[]
 for offset in range(0,len(todo),a.batch):
  torch.cuda.synchronize();batch_start=time.monotonic()
  ps=todo[offset:offset+a.batch];images=[open_image(root,p) for p in ps];batch=processor.process_images(images)
  with torch.inference_mode():emb=model(**batch.to('cuda:0')).float().cpu()
  # Cache only real positions; padding must never participate in MaxSim.
  ids=batch['input_ids'].cpu();mask=batch['attention_mask'].cpu().bool();grid=batch['image_grid_thw'].cpu()
  for i,p in enumerate(ps):
   valid=mask[i];e=emb[i,valid];tokens=ids[i,valid];pos=torch.where(tokens==model.config.image_token_id)[0]
   t,gh,gw=grid[i].tolist();h,w=gh//model.spatial_merge_size,gw//model.spatial_merge_size
   assert t==1 and len(pos)==h*w and e.shape[1]==128
   assert torch.isfinite(e).all() and torch.allclose(e.norm(dim=-1),torch.ones(len(e)),atol=.03)
   boxes=torch.tensor([(x/w,y/h,(x+1)/w,(y+1)/h) for y in range(h) for x in range(w)],dtype=torch.float32)
   if a.smoke and offset==0:
    solo=processor.process_images([images[i]])
    with torch.inference_mode():s=model(**solo.to('cuda:0'))[0].float().cpu()[solo['attention_mask'][0].cpu().bool()]
    assert s.shape==e.shape
    similarity=torch.nn.functional.cosine_similarity(s,e,dim=-1)
    parity.append({'page':p['key'],'max_abs':float((s-e).abs().max()),'min_cosine':float(similarity.min())})
    # Record numerical differences; initial300 established batch sensitivity.
    # Admission compares page ranks/features explicitly, not an arbitrary cosine gate.
   write('pages',p['key'],{'embeddings':e,'input_ids':tokens,'image_positions':pos,'image_grid_thw':grid[i],'patch_boxes':boxes},{'page_key':p['key'],'image_sha256':p['image_sha256'],'grid_after_merge':[h,w],'tokens':len(e),'image_tokens':len(pos),'actual_batch_size':len(ps)})
   images[i].close();count+=1
  del batch,emb
  torch.cuda.synchronize();batch_timings.append({'pages':len(ps),'seconds':time.monotonic()-batch_start})
  print('colqwen pages',offset+len(ps),'/',len(todo),flush=True)
 queries=query_for(catalog,a.shard,a.shards,a.smoke)
 for q in queries:
  key=fingerprint(q['question_key'])
  if complete('queries',key):continue
  b=processor.process_queries([q['question']]);ids=b['input_ids'][0].clone();mask=b['attention_mask'][0].bool().clone()
  with torch.inference_mode():e=model(**b.to('cuda:0'))[0].float().cpu()[mask.cpu()]
  ids=ids.cpu()[mask.cpu()];assert torch.isfinite(e).all()
  write('queries',key,{'embeddings':e,'input_ids':ids},{'question_key':q['question_key'],'question_sha256':fingerprint(q['question']),'tokens':processor.tokenizer.convert_ids_to_tokens(ids.tolist())})
 result={'engine':'colqwen','shard':a.shard,'shards':a.shards,'smoke':a.smoke,'contract_sha256':identity,'processed_pages':count,'requested_pages':len(pages_for(catalog,a.shard,a.shards,a.smoke)),'requested_queries':len(queries),'adapter_check':adapter_check,'batch_singleton_checks':parity,'batch_timings':batch_timings,'stats':stats(start,torch)}
 name=('smoke' if a.smoke else f'shard-{a.shard:02d}')+'-'+os.environ.get('SLURM_JOB_ID','local')+'.json'
 publish(target/'runs'/name,result);print(json.dumps(result),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--base',required=True);p.add_argument('--adapter',required=True);p.add_argument('--batch',type=int,default=4);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--smoke',action='store_true');run(p.parse_args())
