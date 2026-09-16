"""Conservative source-to-PDF evidence alignment; unresolved is never a hit."""
import argparse, collections, json, os, re, unicodedata
from pathlib import Path
from common import publish, read, sha

def norm(s):
 s=re.sub(r'\[\d+(?:[,–-]\d+)*\]','',s)
 return ''.join(c for c in unicodedata.normalize('NFKC',s).casefold() if c.isalnum())
def exact_windows(needle,pages):
 n=norm(needle)
 if len(n)<24:return []
 hits=[]
 for size in [1,2,3]:
  for start in range(len(pages)-size+1):
   if n in ''.join(norm(pages[k]) for k in range(start,start+size)):
    hit=list(range(start,start+size))
    if not any(set(old)<=set(hit) for old in hits):hits.append(hit)
 return hits

def run(root,corpus):
 import pypdfium2 as pdfium
 assert os.environ.get('SLURM_JOB_ID')
 root=Path(root);corpus=Path(corpus);pool=read(root/'cohort/pool.json');inv=read(root/'inventory/inventory.json')
 source=inv['sources'];out=root/'evidence';records=[]
 def pages(doc):
  p=out/'pdf-text'/f'{doc}.json'
  if p.exists():return read(p)
  path=corpus/'pdfs_dev'/f'{doc}.pdf'
  if not path.exists():return {'doc_id':doc,'error':'missing_pdf','pages':[]}
  texts=[]
  with pdfium.PdfDocument(str(path)) as pdf:
   for i in range(len(pdf)):
    page=pdf[i];tp=page.get_textpage();texts.append(tp.get_text_bounded());tp.close();page.close()
  r={'doc_id':doc,'pdf_sha256':sha(path),'pages':texts};publish(p,r);return r
 for ix,q in enumerate(pool['questions']):
  target=out/'questions'/f"{q['qid']}.json"
  if target.exists():records.append(read(target));continue
  t=q['metadata']['type'];retr={(p['doc_id'],p['page_index']) for p in q['original_top4']}
  supports={s['doc_id']:pages(s['doc_id']) for s in q['supporting_context']}
  verified=[];details=[];candidates=[]
  if t=='TextQ':
   for answer in q['answers']:
    alternatives=[]
    for inst in answer.get('text_instances',[]):
     doc=inst['doc_id'];src=source['MMQA_texts.jsonl'].get(doc)
     if not src or norm(str(inst.get('text',''))) not in norm(src['text']):continue
     ps=pages(doc)['pages'];hits=exact_windows(src['text'],ps)
     # A source passage match, not an answer-only string hit.
     for hit in hits:
      evidence=[(doc,k) for k in hit]
      if norm(src['title']) not in ''.join(norm(ps[k]) for k in hit) and ps:
       if norm(src['title']) in norm(ps[0]):evidence.append((doc,0))
       else:continue
      alternatives.append(sorted(set(evidence)))
     an=norm(str(inst.get('text','')))
     candidates.extend({'doc_id':doc,'page_index':i,'reason':'answer_anchor_only_not_verified'} for i,p in enumerate(ps) if an and an in norm(p))
    if alternatives:
     best=min(alternatives,key=lambda e:(len(set(e)-retr),len(e),e));verified.extend(best)
     details.append({'answer':answer['answer'],'method':'full_annotated_source_passage_exact_normalized_match_with_title','pages':best})
    else:details.append({'answer':answer['answer'],'method':'unresolved'})
   # Preserve every annotated supporting passage, not just passages with answer strings.
   for doc in supports:
    src=source['MMQA_texts.jsonl'].get(doc)
    if not src:
     details.append({'doc_id':doc,'method':'unresolved'});continue
    ps=supports[doc]['pages'];hits=exact_windows(src['text'],ps)
    if not hits or not ps or norm(src['title']) not in norm(ps[0]):
     details.append({'doc_id':doc,'method':'unresolved'});continue
    hit=min(hits,key=lambda h:(sum((doc,k) not in retr for k in h),len(h),h))
    verified.extend((doc,k) for k in hit+[0])
    details.append({'doc_id':doc,'method':'full_supporting_passage_and_source_title_exact_match','pages':hit})
   complete=bool(details) and all(x['method']!='unresolved' for x in details)
  elif t=='TableQ':
   doc=q['metadata'].get('table_id');src=source['MMQA_tables.jsonl'].get(doc);complete=False
   if src:
    ps=pages(doc)['pages'];table=src['table'];hits=[];unmatched=[]
    for ri,row in enumerate(table['table_rows']):
     text=' '.join(c['text'] for c in row)
     h=exact_windows(text,ps)
     if h:hits.extend(min(h,key=lambda v:(sum((doc,k) not in retr for k in v),len(v),v)))
     elif norm(text):unmatched.append(ri)
    header=' '.join(c['column_name'] for c in table['header']);hh=exact_windows(header,ps)
    title_ok=bool(ps) and norm(src['title']) in norm(ps[0])
    complete=bool(hits) and not unmatched and bool(hh) and title_ok
    if complete:verified=[(doc,k) for k in sorted(set(hits+hh[0]+[0]))]
    details=[{'method':'entire_annotated_table_rows_headers_and_title_exact_match' if complete else 'unresolved_table_alignment','unmatched_rows':unmatched,'total_rows':len(table['table_rows']),'header_matched':bool(hh),'title_matched':title_ok}]
    for answer in q['answers']:
     for ri,ci in answer.get('table_indices',[]):
      row=table['table_rows'][ri];anchors=[norm(c['text']) for c in row if len(norm(c['text']))>=3]
      for i,p in enumerate(ps):
       n=norm(p);matched=sum(a in n for a in anchors)
       if anchors and matched>=max(2,len(anchors)-1):candidates.append({'doc_id':doc,'page_index':i,'reason':'annotated_answer_row_candidate_not_verified','row':ri,'matched_cells':matched,'cells':len(anchors)})
  else:
   complete=False
   details=[{'method':'requires_original_image_alignment_or_visual_verification','type':t}]
   for doc,rec in supports.items():
    candidates.extend({'doc_id':doc,'page_index':i,'reason':'support_pdf_page_not_verified'} for i in range(len(rec['pages'])))
  verified=sorted(set(verified)) if complete else []
  missing=[{'doc_id':d,'page_index':i} for d,i in verified if (d,i) not in retr]
  status=('verified_present' if not missing else 'verified_missing') if complete else 'unresolved'
  r={'qid':q['qid'],'type':t,'pool_sha256':sha(root/'cohort/pool.json'),'status':status,
    'gold_pages':[{'doc_id':d,'page_index':i} for d,i in verified],'missing_pages':missing,
    'details':details,'candidates':candidates,'support_pdf_hashes':{d:r.get('pdf_sha256') for d,r in supports.items()},
    'verification_scope':'annotated_source_input_alignment; no reader outcomes used'}
  publish(target,r);records.append(r)
  if ix%25==0:print('localized',ix+1,dict(collections.Counter(r['status'] for r in records)),flush=True)
 summary={'questions':len(records),'pool_sha256':sha(root/'cohort/pool.json'),'statuses':dict(collections.Counter(r['status'] for r in records)),
  'by_type':{t:dict(collections.Counter(r['status'] for r in records if r['type']==t)) for t in sorted(pool['type_quotas'])},'no_model_calls':True}
 publish(out/'alignment-summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--corpus',required=True);a=p.parse_args();run(a.root,a.corpus)
