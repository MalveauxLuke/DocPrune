"""Two new conditions per question. Immutable banks, one reader load, no retrieval."""
import argparse, gc, json, sys, time
from pathlib import Path
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO),str(REPO/'src'),str(REPO/'experiments/m3doc600')]
import torch
from transformers import GenerationConfig
from experiments.selector_smoke.run import pages,load_model,to_prompt
from experiments.m3doc600.baseline import PROMPT,REVISION
from experiments.m3doc600.common import read,publish,load_catalog,fingerprint,stats
from docprune.stage2.answerer import FrozenAnswerer
from docprune.stage2.data import load_example
from docprune.stage2.qwen import prepare_prompt

def main(a):
 root=Path(a.root);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
 spec=read(Path(__file__).with_name('spec.json'));assert len(spec['records'])==9
 assert Path(a.reader).name==REVISION
 catalog,_=load_catalog(root)
 audit=read(root/'training-pilot-v1/bank-audit-63555309.json')['record'];folders={r['qid']:Path(r['directory']) for r in audit['rows']}
 processor,model=load_model(a.reader);reader=FrozenAnswerer(model)
 eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
 generation=GenerationConfig(do_sample=False,max_new_tokens=256,repetition_penalty=1.,use_cache=True,eos_token_id=eos,pad_token_id=processor.tokenizer.pad_token_id,bos_token_id=model.generation_config.bos_token_id)
 started=time.monotonic();torch.cuda.reset_peak_memory_stats()
 for r in spec['records']:
  qid=r['qid'];folder=folders[qid];ex,bank=load_example(folder/'example')
  assert bank.baseline_correct and list(ex.identity.ordered_pages)==r['ordered_pages']
  anchor=read(folder/'anchor.json');reference=read(folder/'reference.json');assert anchor==r['anchor'] and reference==r['reference']
  own=list(anchor['continuation_ids'])
  while own and own[-1] in eos:own.pop()
  assert own
  target=torch.tensor(own,device='cuda')
  record=dict(**r,reader_revision=REVISION,full_context_reused=True)
  dest=out/qid;dest.mkdir(exist_ok=True)
  if not (dest/'question-only.json').exists():
   # Same textual instruction/query; no image objects, placeholders or vision call.
   text=processor.apply_chat_template([{'role':'user','content':[{'type':'text','text':PROMPT+r['question']}]}],tokenize=False,add_generation_prompt=True)
   inputs=processor.tokenizer(text,return_tensors='pt').to('cuda');ids=inputs['input_ids'];assert not (ids==model.config.image_token_id).any()
   full=torch.cat((ids,target[:-1][None]),dim=1)
   with torch.inference_mode():
    logits=model(input_ids=full,attention_mask=torch.ones_like(full),use_cache=False,logits_to_keep=len(own)).logits.float()
    ll=torch.log_softmax(logits,dim=-1)[0].gather(1,target[:,None]).squeeze(1)
    generated=model.generate(**inputs,generation_config=generation,use_model_defaults=False)
   tokens=generated[0,ids.shape[1]:].tolist()
   publish(dest/'question-only.json',dict(S=float(ll.mean()),token_loglikelihoods=ll.tolist(),answer=processor.tokenizer.decode(tokens,skip_special_tokens=True).strip(),token_ids=tokens,truncated=not tokens or tokens[-1] not in eos,prompt=text,input_token_ids=ids[0].tolist(),images=0))
   del logits,generated,full,inputs;gc.collect();torch.cuda.empty_cache()
  if not (dest/'evidence-removed.json').exists():
   q,assets,images,regions=pages(root,catalog,qid);assert q['question']==r['question']
   prompt=to_prompt(prepare_prompt(processor,PROMPT+q['question'],images),'cuda')
   for im in images:im.close()
   baseline=read(root/'baseline-qwen3-8b-admitted-v2/answers'/f'{fingerprint(qid)}.json')
   assert prompt.input_ids[0].tolist()==baseline['input_token_ids'] and prompt.image_grid_thw.tolist()==baseline['image_grid_thw']
   assert [ex.layout.region_ids[i] for i in r['remove_indices']]==r['remove_region_ids']
   mask=torch.ones(len(ex.layout.region_ids),dtype=torch.bool);mask[r['remove_indices']]=False
   retained=ex.layout.retained_tokens(mask).cuda();memory=reader.vision(prompt,'evidence9-'+qid)
   with torch.inference_mode():
    score=reader.likelihood(prompt,memory,retained,own)
    answer=reader.generate(prompt,memory,retained,max_new_tokens=256,eos_ids=eos,repetition_penalty=1.)
   publish(dest/'evidence-removed.json',dict(S=score['mean'],token_loglikelihoods=score['token_loglikelihoods'],answer=processor.tokenizer.decode(answer['token_ids'],skip_special_tokens=True).strip(),generation=answer,mask=mask.tolist(),removed_region_ids=r['remove_region_ids'],retained_tokens=len(retained),removed_tokens=int(ex.layout.costs[r['remove_indices']].sum())))
   del memory,prompt;gc.collect();torch.cuda.empty_cache()
  publish(dest/'record.json',record);del ex,bank;gc.collect();torch.cuda.empty_cache()
  print(json.dumps(dict(qid=qid,status='two_conditions_complete')),flush=True)
 # Final capacity smoke: largest full prefix plus target in this admitted nine-case scope.
 smoke_path=out/'largest-context-smoke.json'
 if not smoke_path.exists():
  candidates=[]
  for r in spec['records']:
   b=read(root/'baseline-qwen3-8b-admitted-v2/answers'/f"{fingerprint(r['qid'])}.json")
   candidates.append((len(b['input_token_ids'])+len(r['anchor']['continuation_ids']),r))
  r=max(candidates,key=lambda x:x[0])[1];qid=r['qid'];ex,bank=load_example(folders[qid]/'example')
  q,assets,images,regions=pages(root,catalog,qid)
  prompt=to_prompt(prepare_prompt(processor,PROMPT+q['question'],images),'cuda')
  for im in images:im.close()
  own=list(r['anchor']['continuation_ids'])
  while own and own[-1] in eos:own.pop()
  baseline=read(root/'baseline-qwen3-8b-admitted-v2/answers'/f'{fingerprint(qid)}.json')
  assert prompt.input_ids[0].tolist()==baseline['input_token_ids']
  assert prompt.image_grid_thw.tolist()==baseline['image_grid_thw']
  gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats()
  device=torch.cuda.get_device_properties(0)
  smoke=dict(qid=qid,scope='largest full-prefix-plus-target of nine diagnostic questions; not whole corpus',gpu=device.name,total_vram_bytes=device.total_memory,actual_20gb_class=device.total_memory<=21*1024**3,stages=[])
  try:
   memory=reader.vision(prompt,'evidence9-capacity-'+qid)
   smoke['stages'].append('vision_encoded')
   full=torch.arange(len(ex.layout.owner),device='cuda')
   result=reader.likelihood(prompt,memory,full,own)
   smoke.update(full_visual_tokens=len(full),full_prefix_tokens=prompt.input_ids.shape[1],full_S=result['mean'],saved_full_S=r['reference']['s'],full_S_delta=result['mean']-r['reference']['s'])
   smoke['stages'].append('full_context_scored')
   # All32 saved masks sequentially: validates allocator behavior as well as one-mask peak.
   sequential=[]
   for i in range(32):
    row=read(folders[qid]/'measurements'/f'{i:02}.json')
    ix=ex.layout.retained_tokens(torch.tensor(row['mask'],dtype=torch.bool)).cuda()
    v=reader.likelihood(prompt,memory,ix,own)['mean']
    sequential.append(dict(index=i,retained_tokens=len(ix),S=v,saved_S=row['s'],delta=v-row['s']))
   smoke['sequential_masks']=sequential;smoke['stages'].append('32_masks_scored_sequentially')
   generated=reader.generate(prompt,memory,full,max_new_tokens=256,eos_ids=eos,repetition_penalty=1.)
   smoke['generation']=generated;smoke['answer']=processor.tokenizer.decode(generated['token_ids'],skip_special_tokens=True).strip()
   smoke['stages'].append('full_context_generated');smoke['status']='passed'
  except torch.cuda.OutOfMemoryError:
   smoke['status']='out_of_memory'
   raise
  finally:
   smoke['peak_allocated_bytes']=torch.cuda.max_memory_allocated();smoke['peak_reserved_bytes']=torch.cuda.max_memory_reserved()
   publish(smoke_path,smoke)
  del memory,prompt,ex,bank;gc.collect();torch.cuda.empty_cache()
 assert read(smoke_path)['status']=='passed'
 publish(out/'complete.json',dict(status='passed',questions=9,new_conditions=18,capacity_smoke=str(smoke_path),spec=spec,resources=stats(started,torch)))
if __name__=='__main__':
 p=argparse.ArgumentParser()
 for n in ['root','output','reader']:p.add_argument('--'+n,required=True)
 a=p.parse_args();torch.set_num_threads(2);main(a)
