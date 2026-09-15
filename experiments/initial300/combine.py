"""CPU-only scoped MaxSim ranking and question-associated region profiles."""
import argparse,time,json,os
from pathlib import Path
from common import *
def main(root):
 import torch
 from safetensors.torch import load_file,save_file
 from layout_sources import contracts as layout_contracts,page as layout_page
 torch.set_num_threads(2);start=time.monotonic();root=Path(root);c,digest=load_catalog(root);pc={p['key']:p for p in c['pages']}
 col=read(root/'colqwen/contract.json');miner=read(root/'mineru/contract.json')
 assert col['catalog_sha256']==miner['catalog_sha256']==digest
 layouts=layout_contracts(root);layout_identity=fingerprint({k:v['sha256'] for k,v in layouts.items()})
 for p in c['pages']:
  m=layout_page(root,p,layouts);r=read(root/'colqwen/pages'/f"{p['key']}.json")
  assert m['image_sha256']==r['image_sha256']==p['image_sha256']
  assert m['provenance']['contract_sha256'] in {v['sha256'] for v in layouts.values()} and r['contract_sha256']==col['sha256']
  assert sha(root/'colqwen/pages'/f"{p['key']}.safetensors")==r['tensor_sha256']
 output=root/'combined';output.mkdir(parents=True,exist_ok=True);rankings={};index={};regions_total=0;empty_total=0
 for q in c['questions']:
  qhash=fingerprint(q['question_key']);qm=read(root/'colqwen/queries'/f'{qhash}.json')
  assert qm['question_key']==q['question_key'] and qm['question_sha256']==fingerprint(q['question']) and qm['contract_sha256']==col['sha256']
  qpath=root/'colqwen/queries'/f'{qhash}.safetensors';assert sha(qpath)==qm['tensor_sha256'];query=load_file(str(qpath))['embeddings']
  scores=[]
  # Full document scope, including every low-ranked page; no gold-conditioned insertion.
  for logical in q['pages']:
   f=load_file(str(root/'colqwen/pages'/f"{logical['key']}.safetensors"));sim=query@f['embeddings'].T
   scores.append(dict(logical,score=float(sim.max(dim=1).values.sum()),image_only_score=float(sim[:,f['image_positions']].max(dim=1).values.sum())))
  scores.sort(key=lambda x:(-x['score'],x['page_id']));rankings[q['question_key']]={'question_sha256':fingerprint(q['question']),'retrieval_identity':col['sha256'],'scope':'within_supplied_document','ranked_pages':scores,'page_ids':[x['page_id'] for x in scores]}
  selected=scores[:20];patch_scores=[];boxes=[];patch_pages=[];profiles=[];valid=[];region_meta=[]
  for rank,logical in enumerate(selected):
   f=load_file(str(root/'colqwen/pages'/f"{logical['key']}.safetensors"));ps=query@f['embeddings'][f['image_positions']].T;b=f['patch_boxes'];patch_scores.append(ps);boxes.append(b);patch_pages.append(torch.full((len(b),),rank,dtype=torch.int64))
   m=layout_page(root,pc[logical['key']],layouts)
   for r in m['regions']:
    rb=torch.tensor(r['bbox']);area=(torch.minimum(b[:,2:],rb[2:])-torch.maximum(b[:,:2],rb[:2])).clamp_min(0).prod(dim=1);members=area>0
    profiles.append(ps[:,members].max(dim=1).values if members.any() else torch.zeros(len(query)));valid.append(bool(members.any()));region_meta.append(dict(r,page_id=logical['page_id'],page_rank=rank,geometric_patch_count=int(members.sum())))
  target=output/'profiles'/f'{qhash}.safetensors';target.parent.mkdir(exist_ok=True)
  tensors={'query_vectors':query,'query_patch_scores':torch.cat(patch_scores,dim=1),'patch_boxes':torch.cat(boxes),'patch_pages':torch.cat(patch_pages),'region_query_maxsim':torch.stack(profiles) if profiles else torch.empty((0,len(query))),'region_profile_valid':torch.tensor(valid,dtype=torch.bool)}
  if target.exists():
   old=read(target.with_suffix('.json'));assert sha(target)==old['tensor_sha256']
  else:
   tmp=target.with_suffix('.tmp-'+str(os.getpid()));save_file(tensors,str(tmp));os.link(tmp,target);tmp.unlink()
  metadata={'schema':'docprune-colqwen-region-profile-v1','question_key':q['question_key'],'question_sha256':fingerprint(q['question']),'ordered_pages':[x['page_id'] for x in selected],'retrieval_identity':col['sha256'],'layout_identity':layout_identity,'layout_contracts':{k:v['sha256'] for k,v in layouts.items()},'tensor_sha256':sha(target),'regions':region_meta,'missing_profiles':sum(not x for x in valid),'mapping':'positive-area many-to-many intersection; no fabricated nearest region','zero_profile_rule':'zero entries are missing only where region_profile_valid is false','all_page_rankings_saved':True}
  publish(target.with_suffix('.json'),metadata);index[q['question_key']]=str(target.relative_to(root));regions_total+=len(region_meta);empty_total+=metadata['missing_profiles']
 publish(output/'rankings.json',rankings);publish(output/'profile-index.json',{'entries':index,'catalog_sha256':digest})
 completion={'status':'complete','questions':len(index),'logical_pages':c['logical_page_count'],'unique_pages':c['unique_page_count'],'top20_region_instances':regions_total,'regions_without_overlapping_colqwen_patch':empty_total,'catalog_sha256':digest,'colqwen_identity':col['sha256'],'mineru_identity':layout_identity,'mineru_contracts':{k:v['sha256'] for k,v in layouts.items()},'answerer_calls':0,'training_steps':0,'stats':stats(start)}
 if (output/'completion.json').exists():
  old=read(output/'completion.json');completion['stats']=old['stats']
 publish(output/'completion.json',completion)
 print(json.dumps(read(output/'completion.json')),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(p.parse_args().root)
