"""Render a deterministic 32-page resource smoke from frozen original top four."""
import argparse, collections, hashlib, json, os
from pathlib import Path
from common import publish, sha, fingerprint

def run(pool,corpus,out):
 import pypdfium2 as pdfium
 assert os.environ.get('SLURM_JOB_ID')
 pool=Path(pool);root=Path(out);d=json.loads(pool.read_text());qs=[]
 for t in sorted(d['type_quotas']):
  rows=sorted((q for q in d['questions'] if q['metadata']['type']==t),key=lambda q:(len(q['question']),q['qid']))
  qs.extend([rows[0],rows[-1]])
 pages={};questions=[];pdf_hash={}
 for q in qs:
  links=[]
  for p in q['original_top4']:
   doc=p['doc_id'];idx=p['page_index'];path=Path(corpus)/'pdfs_dev'/f'{doc}.pdf'
   if doc not in pdf_hash:pdf_hash[doc]=sha(path)
   with pdfium.PdfDocument(str(path)) as pdf:
    page=pdf[idx];bitmap=page.render(scale=2);im=bitmap.to_pil().convert('RGB');w,h=im.size
    pixel=hashlib.sha256(im.tobytes()).hexdigest();key=fingerprint([w,h,pixel]);f=root/'images'/f'{key}.png'
    f.parent.mkdir(parents=True,exist_ok=True)
    if not f.exists():im.save(f)
    record=pages.setdefault(key,{'key':key,'image':str(f.resolve()),'image_sha256':sha(f),'rgb_sha256':pixel,'width':w,'height':h,'aliases':[]})
    record['aliases'].append({'source':'m3docvqa','document_id':doc,'page_number':idx+1,'pdf_sha256':pdf_hash[doc],'rendering':'pypdfium2-144dpi-native-page'})
    links.append({'key':key,'page_id':f'{doc}:{idx}','page_number':idx+1})
    im.close();bitmap.close();page.close()
  questions.append({'question_key':q['qid'],'question':q['question'],'source':'m3docvqa','pages':links})
 c={'schema':'docprune-m3doc600-pages-v1','manifest_sha256':sha(pool),'pool_path':str(pool.resolve()),'stage':'resource_smoke_original_top4_only','pages':list(pages.values()),'questions':questions,'smoke_keys':sorted(pages),'smoke_questions':[q['question_key'] for q in questions]}
 c['sha256']=fingerprint(c);publish(root/'catalog.json',c)
 for engine,batches in [('colqwen',[4,8,16]),('mineru',[8,16])]:
  for batch in batches:publish(root/f'{engine}-b{batch}'/'catalog.json',c)
 print(json.dumps({'pages':len(pages),'questions':len(questions),'catalog_sha256':c['sha256']}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--pool',required=True);p.add_argument('--corpus',required=True);p.add_argument('--out',required=True);a=p.parse_args();run(a.pool,a.corpus,a.out)
