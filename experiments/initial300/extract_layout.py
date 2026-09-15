"""MinerU layout detection only: types/boxes/order, without content transcription."""
import argparse,time,os,math
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from common import *

def run(a):
 import torch
 from transformers import Qwen2VLForConditionalGeneration,AutoProcessor
 from mineru_vl_utils import MinerUClient
 from importlib.metadata import version
 assert version('mineru')=='3.0.9' and version('mineru-vl-utils')=='0.2.8'
 assert Path(a.model).name==MINER[1]
 assert torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0]>=8
 torch.set_num_threads(2);torch.cuda.reset_peak_memory_stats();start=time.monotonic()
 root=Path(a.root);catalog,digest=load_catalog(root)
 model=Qwen2VLForConditionalGeneration.from_pretrained(a.model,torch_dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa',local_files_only=True).eval()
 processor=AutoProcessor.from_pretrained(a.model,use_fast=True,local_files_only=True)
 executor=ThreadPoolExecutor(max_workers=2)
 client=MinerUClient(backend='transformers',model=model,processor=processor,batch_size=a.batch,executor=executor,use_tqdm=False,abandon_list=False,abandon_paratext=False)
 contract={'schema':'docprune-mineru-layout-v1','catalog_sha256':digest,'model':MINER,'mineru':version('mineru'),'mineru-vl-utils':version('mineru-vl-utils'),'torch':torch.__version__,'transformers':version('transformers'),'backend':'transformers','dtype':'bfloat16','attention':'sdpa','layout_image_size':[1036,1036],'stage':'batch_layout_detect','content_transcription':False,'retain_paratext':True,'batch_size':a.batch,'code_sha256':sha(__file__),'common_sha256':sha(Path(__file__).with_name('common.py'))}
 identity=fingerprint(contract);target=root/'mineru';publish(target/'contract.json',dict(contract,sha256=identity))
 todo=[]
 for p in pages_for(catalog,a.shard,a.shards,a.smoke):
  f=target/'pages'/(p['key']+'.json')
  if f.exists():
   record=read(f);assert record['provenance']['contract_sha256']==identity and record['image_sha256']==p['image_sha256']
  else:todo.append(p)
 total_regions=0;blank=[]
 for offset in range(0,len(todo),a.batch):
  ps=todo[offset:offset+a.batch];ims=[open_image(root,p) for p in ps]
  with torch.inference_mode():results=client.batch_layout_detect(ims)
  assert len(results)==len(ps)
  for p,im,result in zip(ps,ims,results):
   raw=[dict(x) for x in result];regions=[]
   for i,block in enumerate(raw):
    box=block['bbox'];assert len(box)==4 and all(math.isfinite(v) and 0<=v<=1 for v in box) and box[0]<box[2] and box[1]<box[3]
    regions.append({'id':f'r{i:04d}','type':block['type'],'bbox':box,'angle':block.get('angle'),'merge_prev':block.get('merge_prev',False)})
   if not regions:blank.append(p['key'])
   total_regions+=len(regions)
   publish(target/'pages'/(p['key']+'.json'),{'schema':'docprune-layout-boxes-v1','page_id':p['key'],'image_sha256':p['image_sha256'],'provenance':{'contract_sha256':identity,'stage':'mineru_layout_detection_only','actual_batch_size':len(ps)},'regions':regions,'raw_layout':raw,'empty_layout':not bool(regions)})
   im.close()
  print('mineru pages',offset+len(ps),'/',len(todo),'regions',total_regions,flush=True)
 executor.shutdown()
 result={'engine':'mineru','shard':a.shard,'shards':a.shards,'smoke':a.smoke,'contract_sha256':identity,'processed_pages':len(todo),'requested_pages':len(pages_for(catalog,a.shard,a.shards,a.smoke)),'regions':total_regions,'empty_layout_pages':blank,'stats':stats(start,torch)}
 name=('smoke' if a.smoke else f'shard-{a.shard:02d}')+'-'+os.environ.get('SLURM_JOB_ID','local')+'.json';publish(target/'runs'/name,result);print(json.dumps(result),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--model',required=True);p.add_argument('--batch',type=int,default=4);p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1);p.add_argument('--smoke',action='store_true');run(p.parse_args())
