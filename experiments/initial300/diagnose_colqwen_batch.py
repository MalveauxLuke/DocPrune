"""Bounded diagnostic; never writes production feature caches or admissions."""
import argparse,contextlib,json,os,sys,time
from pathlib import Path
from common import load_catalog,pages_for,open_image,sha,publish,fingerprint,stats

def main(a):
 import torch
 from colpali_engine.models import ColQwen2_5,ColQwen2_5_Processor
 from safetensors.torch import save_file
 from torch.nn.attention import sdpa_kernel,SDPBackend
 sys.path.insert(0,str(Path(__file__).resolve().parents[2]/'scripts'))
 from extract_colfeatures17 import load_verified_adapter
 torch.set_num_threads(2);start=time.monotonic();root=Path(a.root)
 out=root/'diagnostics'/('batch-'+os.environ['SLURM_JOB_ID']);out.mkdir(parents=True,exist_ok=False)
 c,digest=load_catalog(root);pages=pages_for(c,0,1,True)[:4];images=[open_image(root,p) for p in pages]
 model=ColQwen2_5.from_pretrained(a.base,torch_dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa',local_files_only=True).eval()
 adapter=load_verified_adapter(model,a.adapter)
 proc=ColQwen2_5_Processor.from_pretrained(a.adapter,local_files_only=True);proc.query_prefix='Query: '
 captured={}
 def pre_lm(module,args,kw):
  if 'position_ids' in kw and kw['position_ids'] is not None:captured['position_ids']=kw['position_ids'].detach().cpu()
 def vision_hook(module,args,res):captured['vision']=res.detach().float().cpu()
 def projection_hook(module,args,res):captured['projection']=res.detach().float().cpu()
 handles=[model.language_model.register_forward_pre_hook(pre_lm,with_kwargs=True),model.visual.register_forward_hook(vision_hook),model.custom_text_proj.register_forward_hook(projection_hook)]
 report={'catalog_sha256':digest,'code_sha256':sha(__file__),'adapter':adapter,'pages':[{'key':p['key'],'aliases':p['aliases'],'width':p['width'],'height':p['height'],'image_sha256':p['image_sha256']} for p in pages],'padding_side':proc.tokenizer.padding_side,'variants':{}}
 def emit(tag,value):
  publish(out/(tag+'.json'),value);report['variants'][tag]=value;print(tag,json.dumps(value),flush=True)
 def run(tag,which,math=False,explicit=False):
  batch=proc.process_images([images[i] for i in which]);cpu={k:v.clone() for k,v in batch.items()};captured.clear();t=time.monotonic()
  inp=batch.to('cuda:0')
  if explicit:
   inp['position_ids']=model.get_rope_index(inp['input_ids'],inp['image_grid_thw'],attention_mask=inp['attention_mask'])[0]
  with torch.inference_mode(),(sdpa_kernel(SDPBackend.MATH) if math else contextlib.nullcontext()):e=model(**inp).float().cpu()
  torch.cuda.synchronize();elapsed=time.monotonic()-t;result=[];start_v=0;save={}
  for j,i in enumerate(which):
   valid=cpu['attention_mask'][j].bool();ids=cpu['input_ids'][j,valid];image_pos=ids==model.config.image_token_id;count=int(image_pos.sum());grid=cpu['image_grid_thw'][j];n=int(grid[1]*grid[2]);pos=captured.get('position_ids')
   row={'e':e[j,valid],'ids':ids,'image':image_pos,'grid':grid,'pixels':cpu['pixel_values'][j,:n],'positions':pos[:,j,valid] if pos is not None else None,'vision':captured['vision'][start_v:start_v+count],'raw_projection':captured['projection'][j,valid],'padding':int((~valid).sum())}
   start_v+=count;result.append(row)
   for k in ('e','ids','positions','raw_projection'):
    if row[k] is not None:save[f'{j}_{k}']=row[k].contiguous()
  assert start_v==len(captured['vision'])
  save_file(save,str(out/(tag+'.safetensors')))
  print('completed',tag,round(elapsed,3),'seconds',flush=True)
  return result,elapsed
 def metric(x,y):
  delta=(x-y).abs();cos=torch.nn.functional.cosine_similarity(x,y,dim=-1)
  return {'max_abs':float(delta.max()),'mean_abs':float(delta.mean()),'min_cosine':float(cos.min()),'mean_cosine':float(cos.mean()),'below_0995':int((cos<.995).sum()),'vectors':len(cos)}
 def compare(tag,rows,refs,elapsed):
  values=[]
  for j,(r,b) in enumerate(zip(rows,refs)):
   same_ids=torch.equal(r['ids'],b['ids']);same_grid=torch.equal(r['grid'],b['grid']);same_pixels=torch.equal(r['pixels'],b['pixels']);same_pos=r['positions'] is not None and b['positions'] is not None and torch.equal(r['positions'],b['positions'])
   v={'index':j,'same_ids':same_ids,'same_grid':same_grid,'same_pixels':same_pixels,'pixel_max_abs':float((r['pixels']-b['pixels']).abs().max()) if r['pixels'].shape==b['pixels'].shape else None,'same_positions':same_pos,'padding':r['padding'],'reference_padding':b['padding']}
   if same_ids:
    v['all']=metric(r['e'],b['e']);v['image']=metric(r['e'][r['image']],b['e'][b['image']]);v['nonimage']=metric(r['e'][~r['image']],b['e'][~b['image']]);v['vision']=metric(r['vision'],b['vision']);v['raw_projection']=metric(r['raw_projection'],b['raw_projection'])
    cs=torch.nn.functional.cosine_similarity(r['e'],b['e'],dim=-1);worst=torch.argsort(cs)[:8]
    v['worst_tokens']=[{'index':int(k),'token':proc.tokenizer.convert_ids_to_tokens(int(r['ids'][k])),'cosine':float(cs[k]),'image':bool(r['image'][k]),'reference_projection_norm':float(b['raw_projection'][k].norm())} for k in worst]
   values.append(v)
  emit(tag,{'seconds':elapsed,'comparisons':values})
 # Reproduce the exact original order: mixed batch, then singleton pages.
 mixed,dt=run('native_mixed',list(range(4)));refs=[]
 for i in range(4):
  rows,_=run('native_single_'+str(i),[i]);refs+=rows
 compare('native_mixed_comparison',mixed,refs,dt)
 repeat,dt=run('native_single_repeat',[0]);compare('single_repeat_comparison',repeat,[refs[0]],dt)
 repeat,dt=run('native_mixed_repeat',list(range(4)));compare('mixed_repeat_comparison',repeat,mixed,dt)
 identical,dt=run('native_identical',[0]*4);compare('identical_comparison',identical,[refs[0]]*4,dt)
 reverse,dt=run('native_reversed',[3,2,1,0]);compare('reversed_comparison',list(reversed(reverse)),refs,dt)
 explicit,dt=run('native_explicit_positions',list(range(4)),explicit=True);compare('explicit_positions_comparison',explicit,refs,dt)
 mathmixed,dt=run('math_mixed',list(range(4)),math=True);mathrefs=[]
 for i in range(4):
  rows,_=run('math_single_'+str(i),[i],math=True);mathrefs+=rows
 compare('math_mixed_comparison',mathmixed,mathrefs,dt)
 compare('math_vs_native_singletons',mathrefs,refs,0)
 # Measure retrieval-score effects for four actual questions, held fixed across page variants.
 for h in handles:h.remove()
 query=[]
 for p in pages:
  q=next(q for q in c['questions'] if any(x['key']==p['key'] for x in q['pages']))
  b=proc.process_queries([q['question']]);mask=b['attention_mask'][0].bool()
  with torch.inference_mode():qe=model(**b.to('cuda:0'))[0].float().cpu()[mask]
  query.append((q['question_key'],qe))
 scores=[]
 for qid,q in query:
  for tag,rows in [('native_mixed',mixed),('math_mixed',mathmixed),('native_identical_first',[identical[0]])]:
   for i,r in enumerate(rows):
    b=refs[i];full=float((q@r['e'].T).max(1).values.sum());ref=float((q@b['e'].T).max(1).values.sum());im=float((q@r['e'][r['image']].T).max(1).values.sum());imref=float((q@b['e'][b['image']].T).max(1).values.sum())
    scores.append({'question':qid,'variant':tag,'page':i,'full_delta':full-ref,'image_only_delta':im-imref,'reference_full':ref})
 emit('retrieval_score_deltas',scores)
 report['stats']=stats(start,torch);publish(out/'completion.json',report)
 print('DIAGNOSTIC_COMPLETE',str(out),json.dumps(report['stats']),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--base',required=True);p.add_argument('--adapter',required=True);main(p.parse_args())
