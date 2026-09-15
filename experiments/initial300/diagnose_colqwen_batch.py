"""Measure four-page batch differences, without production cache/admission writes."""
import argparse,os,sys,time
from pathlib import Path
from common import load_catalog,pages_for,open_image,sha,publish,stats

def main(a):
 import torch
 from colpali_engine.models import ColQwen2_5,ColQwen2_5_Processor
 sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
 from extract_colfeatures17 import load_verified_adapter
 torch.set_num_threads(2);start=time.monotonic();root=Path(a.root)
 out=root/'diagnostics'/(('regions-' if a.regions else 'batch-')+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True,exist_ok=False)
 catalog,digest=load_catalog(root);pages=pages_for(catalog,0,1,True)[:4]
 images=[open_image(root,p) for p in pages]
 model=ColQwen2_5.from_pretrained(a.base,torch_dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa',local_files_only=True).eval()
 adapter=load_verified_adapter(model,a.adapter)
 proc=ColQwen2_5_Processor.from_pretrained(a.adapter,local_files_only=True);proc.query_prefix='Query: '
 batch=proc.process_images(images)
 ids=batch['input_ids'].clone();masks=batch['attention_mask'].bool().clone();grids=batch['image_grid_thw'].clone()
 with torch.inference_mode():mixed=model(**batch.to('cuda:0')).float().cpu()
 del batch
 def difference(x,y):
  cosine=torch.nn.functional.cosine_similarity(x,y,dim=-1)
  return {'max_absolute_difference':float((x-y).abs().max()),'mean_absolute_difference':float((x-y).abs().mean()),'minimum_cosine':float(cosine.min()),'mean_cosine':float(cosine.mean()),'tokens_at_or_below_0995':int((cosine<=.995).sum()),'token_count':len(cosine)}
 rows=[]
 for i,p in enumerate(pages):
  single=proc.process_images([images[i]])
  sid=single['input_ids'][0].clone();smask=single['attention_mask'][0].bool().clone();grid=single['image_grid_thw'][0].clone()
  with torch.inference_mode():reference=model(**single.to('cuda:0'))[0].float().cpu()[smask]
  candidate=mixed[i,masks[i]];tokens=ids[i,masks[i]]
  assert torch.equal(tokens,sid[smask]) and torch.equal(grid,grids[i])
  assert candidate.shape==reference.shape and torch.isfinite(candidate).all() and torch.isfinite(reference).all()
  image=tokens==model.config.image_token_id
  row={'page_key':p['key'],'aliases':p['aliases'],'image_sha256':p['image_sha256'],'same_token_ids':True,'same_image_grid':True,'padding_tokens':int((~masks[i]).sum()),'all_tokens':difference(candidate,reference),'image_tokens':difference(candidate[image],reference[image]),'nonimage_tokens':difference(candidate[~image],reference[~image])}
  publish(out/f'page-{i}-embeddings.json',row)
  # One real question associated with this page; no cross-document retrieval.
  q=next(q for q in catalog['questions'] if any(x['key']==p['key'] for x in q['pages']))
  query=proc.process_queries([q['question']]);qm=query['attention_mask'][0].bool().clone()
  with torch.inference_mode():qe=model(**query.to('cuda:0'))[0].float().cpu()[qm]
  row['question_key']=q['question_key'];row['retrieval_scores']={}
  for name,selected in [('full_page',torch.ones(len(tokens),dtype=torch.bool)),('image_only',image)]:
   base=float((qe@reference[selected].T).max(dim=1).values.sum());batched=float((qe@candidate[selected].T).max(dim=1).values.sum())
   row['retrieval_scores'][name]={'single':base,'batch':batched,'delta':batched-base}
  if a.regions:
   import json
   from safetensors.torch import save_file
   layout_path=root/'mineru/pages'/f"{p['key']}.json"
   layout=json.loads(layout_path.read_text())
   assert layout['image_sha256']==p['image_sha256']
   t,gh,gw=grid.tolist();h,w=gh//model.spatial_merge_size,gw//model.spatial_merge_size
   assert t==1 and h*w==int(image.sum())
   boxes=torch.tensor([(x/w,y/h,(x+1)/w,(y+1)/h) for y in range(h) for x in range(w)])
   sims=[qe@reference[image].T,qe@candidate[image].T];rr=[]
   for region in layout['regions']:
    rb=torch.tensor(region['bbox']);members=(torch.minimum(boxes[:,2:],rb[2:])-torch.maximum(boxes[:,:2],rb[:2])).clamp_min(0).prod(dim=1)>0
    if not members.any():
     rr.append(dict(region,patch_count=0,single=None,batch=None));continue
    scores=[float(sim[:,members].max(dim=1).values.sum()) for sim in sims]
    rr.append(dict(region,patch_count=int(members.sum()),single=scores[0],batch=scores[1],delta=scores[1]-scores[0]))
   valid=[x for x in rr if x['single'] is not None]
   orders={name:[x['id'] for x in sorted(valid,key=lambda x:(-x[name],x['id']))] for name in ['single','batch']}
   rankmaps={name:{rid:k+1 for k,rid in enumerate(order)} for name,order in orders.items()}
   for x in valid:
    x['single_rank']=rankmaps['single'][x['id']];x['batch_rank']=rankmaps['batch'][x['id']]
   overlap={str(k):{'k':min(k,len(valid)),'shared':len(set(orders['single'][:k])&set(orders['batch'][:k]))} for k in [1,3,5]}
   changed=sum(1 for j,x in enumerate(valid) for y in valid[j+1:] if (x['single']-y['single'])*(x['batch']-y['batch'])<0)
   row['region_comparison']={'mapping':'positive-area patch overlap; sum of per-query-token regional maxima, same as combine profiles','layout_sha256':sha(layout_path),'regions':rr,'rankings':orders,'top_overlap':overlap,'strict_pairwise_reversals':changed,'pair_count':len(valid)*(len(valid)-1)//2,'max_rank_shift':max([abs(x['single_rank']-x['batch_rank']) for x in valid],default=0)}
   save_file({'single':reference.contiguous(),'batch':candidate.contiguous(),'query':qe.contiguous(),'input_ids':tokens.contiguous(),'patch_boxes':boxes.contiguous()},str(out/f'page-{i}.safetensors'))
   print('REGION_COMPARISON',i,overlap,'reversals',changed,flush=True)
  publish(out/f'page-{i}-complete.json',row);rows.append(row);images[i].close()
  print('PAGE_COMPARISON',i,row,flush=True)
 result={'scope':'four original smoke pages; one mixed batch and four singletons; associated-question score comparison','catalog_sha256':digest,'code_sha256':sha(__file__),'adapter':adapter,'pages':rows,'original_threshold':.995,'original_threshold_passed':all(r['all_tokens']['minimum_cosine']>.995 for r in rows),'production_admission':False,'stats':stats(start,torch)}
 publish(out/'completion.json',result);print('DIAGNOSTIC_COMPLETE',str(out),result['original_threshold_passed'],result['stats'],flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--regions',action='store_true');p.add_argument('--root',required=True);p.add_argument('--base',required=True);p.add_argument('--adapter',required=True);main(p.parse_args())
