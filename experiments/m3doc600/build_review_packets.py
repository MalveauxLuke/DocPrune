"""Package a deterministic mixed calibration batch; no new model calls."""
import argparse,hashlib,json,os,shutil
from pathlib import Path
from common import read,publish,sha

CHALLENGES=['89317cdd73ec3b7a3b1ba99c3d9d4e82','48f3645e358ead94a77793bbf865148e','e234125d2e799b7bc11eb2d8196fcce3','7275a1bc421f71a683f41129b3daea17']
def select(pool):
 rank=lambda q:hashlib.sha256(('m3doc600-review-calibration-v1\0'+q['qid']).encode()).hexdigest()
 selected=[]
 for t in sorted(pool['type_quotas']):selected.extend(sorted((q for q in pool['questions'] if q['metadata']['type']==t),key=rank)[:2])
 byid={q['qid']:q for q in pool['questions']};seen={q['qid'] for q in selected}
 for k in CHALLENGES:
  if k in byid and k not in seen:selected.append(byid[k]);seen.add(k)
 for q in sorted(pool['questions'],key=rank):
  if len(selected)>=12:break
  if q['qid'] not in seen:selected.append(q);seen.add(q['qid'])
 return selected

def run(root,corpus,all_cases=False):
 assert os.environ.get('SLURM_JOB_ID'),'Compute allocation required'
 root=Path(root);corpus=Path(corpus);pool=read(root/'cohort/pool.json');poolsha=sha(root/'cohort/pool.json')
 assert poolsha=='caabb62c3c0e875f58ceadd86c8b09a4364a57617edd01782503e215cceff452'
 selected=pool['questions'] if all_cases else select(pool);out=root/'review-packets'/('all-v2' if all_cases else 'calibration-v1');out.mkdir(parents=True,exist_ok=True)
 docs={s['doc_id'] for q in selected for s in q['supporting_context']}|{p['doc_id'] for q in selected for p in q['original_top4']}
 files=[]
 def include(src,relative):
  src=Path(src);dst=out/relative;dst.parent.mkdir(parents=True,exist_ok=True)
  if dst.exists():assert sha(src)==sha(dst)
  else:os.link(src,dst)
  r={'path':relative,'bytes':dst.stat().st_size,'sha256':sha(dst)};files.append(r);return r
 include(root/'cohort/pool.json','pool.json')
 pdfs={d:include(corpus/'pdfs_dev'/(d+'.pdf'),'pdfs/'+d+'.pdf') for d in sorted(docs)}
 sources={}
 needed=set(docs)
 for q in selected:
  needed.update(q['metadata'].get('image_doc_ids',[]));needed.update(q['metadata'].get('text_doc_ids',[]))
  if q['metadata'].get('table_id'):needed.add(q['metadata']['table_id'])
 for name in ['MMQA_texts.jsonl','MMQA_tables.jsonl','MMQA_images.jsonl']:
  sources[name]={}
  for line in (corpus/'multimodalqa'/name).open():
   row=json.loads(line)
   if row['id'] in needed:sources[name][row['id']]=row
 publish(out/'source_contexts.json',sources)
 imrefs={r['doc_id']:r for r in read(root/'original-images/manifest.json')['images']};images={}
 for d in sorted(docs&set(imrefs)):
  ref=imrefs[d];assert sha(ref['file'])==ref['sha256'];images[d]=include(ref['file'],'original-images/'+Path(ref['file']).name)
 for q in selected:
  ids={s['doc_id'] for s in q['supporting_context']}|{p['doc_id'] for p in q['original_top4']}
  data={'qid':q['qid'],'question':q['question'],'type':q['metadata']['type'],'pool_sha256':poolsha,'original_top4':q['original_top4'],'supporting_context':q['supporting_context'],'metadata':{k:v for k,v in q['metadata'].items() if k!='intermediate_answers'},'pdfs':{d:pdfs[d] for d in sorted(ids)},'original_images':{d:images[d] for d in sorted(ids&set(images))},'additional_candidate_contexts_in_source_contexts':True,'warning':'Support annotations are localization clues, not proof of sufficient or exhaustive evidence. Negative/list candidates may need additional PDFs; report missing assets explicitly.'}
  publish(out/'cases'/(q['qid']+'.json'),data)
 publish(out/'gold.json',{q['qid']:q['answers'] for q in selected})
 manifest={'schema':'m3doc600-review-packet-v1','pool_sha256':poolsha,'selection':'All frozen V2 cases; no reader outcome filtering' if all_cases else 'Two seeded cases per type, four deliberate changed/equivalent-photo challenges, deterministic fill to12; no reader outcomes','qids':[q['qid'] for q in selected],'files':files,'source_contexts_sha256':sha(out/'source_contexts.json'),'gold_sha256':sha(out/'gold.json'),'case_hashes':{q['qid']:sha(out/'cases'/(q['qid']+'.json')) for q in selected},'unique_pdfs':len(pdfs),'total_linked_bytes':sum(f['bytes'] for f in files),'validated_questions':0}
 publish(out/'manifest.json',manifest)
 print(json.dumps({'packet':str(out),'questions':len(selected),'unique_pdfs':len(pdfs),'linked_bytes':manifest['total_linked_bytes'],'manifest_sha256':sha(out/'manifest.json')}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--corpus',required=True);p.add_argument('--all',action='store_true');a=p.parse_args();run(a.root,a.corpus,a.all)
