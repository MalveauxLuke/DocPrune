"""Score frozen baseline answers; never select/filter questions by outcome."""
import argparse,json
from pathlib import Path
from common import read,load_catalog,fingerprint,publish,sha
from answer_metrics import list_em,list_f1

def run(root):
 root=Path(root); c,d=load_catalog(root)
 out=root/'baseline-qwen3-8b-admitted-v1'; contract=read(out/'contract.json')
 pool={q['qid']:q for q in read(c['pool_path'])['questions']}; rows=[]
 assert len(pool)==471 and contract['catalog_sha256']==d
 for q in c['questions']:
  r=read(out/'answers'/(fingerprint(q['question_key'])+'.json'))
  assert r['contract_sha256']==contract['sha256'] and r['question_key']==q['question_key']
  assert r['ordered_pages']==[q['pages'][i]['page_id'] for i in q['presentation_order']]
  gold=[str(a['answer']) for a in pool[q['question_key']]['answers']]
  prediction=r['answer']
  try:
   parsed=json.loads(prediction)
   if isinstance(parsed,list) and all(isinstance(v,(str,int,float)) for v in parsed):prediction=[str(v) for v in parsed]
  except (ValueError,TypeError):pass
  em=list_em(prediction,gold); f1=list_f1(prediction,gold)
  rows.append(dict(qid=q['question_key'],answer=r['answer'],gold=gold,em=em,f1=f1,
    context_stratum=r['context_stratum'],modality=r['modality'],hit_generation_limit=r['hit_generation_limit'],
    assessment='exact_match' if em else 'non_exact_match_semantic_review_not_performed',answer_sha256=sha(out/'answers'/(fingerprint(q['question_key'])+'.json'))))
 groups={}
 for name in ['all','original_four','supplemented','TextQ','TableQ','ImageQ']:
  g=[r for r in rows if name=='all' or name in (r['context_stratum'],r['modality'])]
  groups[name]=dict(n=len(g),em=sum(r['em'] for r in g)/len(g),f1=sum(r['f1'] for r in g)/len(g),truncated=sum(r['hit_generation_limit'] for r in g))
 result=dict(status='complete',contract_sha256=contract['sha256'],questions=len(rows),groups=groups,rows=rows,metrics_sha256=sha(Path(__file__).with_name('answer_metrics.py')))
 publish(out/'assessment.json',result); print(json.dumps(groups),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);run(p.parse_args().root)
