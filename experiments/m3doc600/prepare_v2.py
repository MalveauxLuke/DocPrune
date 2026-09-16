"""Preserve rejected V1 and reuse immutable source assets in corrected V2."""
import argparse,os
from pathlib import Path
from common import read,publish,sha

def run(parent):
 assert os.environ.get('SLURM_JOB_ID')
 parent=Path(parent);old=parent/'20260915-v1';new=parent/'20260915-v2'
 pool=read(new/'cohort/pool.json');assert len(pool['questions'])==600
 assert len(pool['older600_reused_qids'])==6
 assert all(q['metadata']['type']!='TextQ' or len(q['supporting_context'])==1 for q in pool['questions'])
 assert all(set(q['prior_selected_cohorts'])<= {'older600'} for q in pool['questions'])
 publish(old/'cohort/REJECTED.json',{'status':'rejected_for_single_hop_eligibility','pool_sha256':sha(old/'cohort/pool.json'),'reason':'Modality-only screen admitted multi-paragraph TextQ, including a verified two-step bridge question.','ineligible_text_rows':172,'example_qid':'4e0256e74443d82b6a1870b776d0ec37','source':'https://arxiv.org/html/2104.06039v1#S2','replacement_pool':str(new/'cohort/pool.json'),'replacement_sha256':sha(new/'cohort/pool.json'),'smoke_measurements_reusable':True})
 counts={}
 for sub in ['evidence/pdf-text','original-images']:
  dest=new/sub;dest.mkdir(parents=True,exist_ok=True);count=0
  for f in (old/sub).iterdir():
   if not f.is_file() or sub=='original-images' and f.suffix=='.json':continue
   t=dest/f.name
   if t.exists():assert sha(t)==sha(f)
   else:os.link(f,t)
   count+=1
  counts[sub]=count
 (new/'logs').mkdir(exist_ok=True)
 publish(new/'cohort/preparation.json',{'pool_sha256':sha(new/'cohort/pool.json'),'reused_source_files':counts,'resource_smoke_root':str(old/'smoke'),'resource_smoke_jobs':['63356461','63356462'],'selected_colqwen_batch':4,'selected_mineru_batch':16,'full_production_started':False,'evidence_policy':'awaiting_owner_decision_for_changed_source_photographs'})
 print(counts,flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--parent',required=True);run(p.parse_args().parent)
