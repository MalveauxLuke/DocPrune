import json,pathlib,collections,hashlib
B=pathlib.Path('/scratch/lmalveau/docprune');O=pathlib.Path(__file__).parent
meta={r['qid']:r for r in map(json.loads,open(B/'datasets/m3docvqa/multimodalqa/MMQA_dev.jsonl'))}
protected_q=set();protected_d=set();sources=[]
for name in ['task9-baseline-wrong100-inputs-c0c9bee-v1/fixture.json','task9-shared-probe-random600-inputs-v1/fixture.json','task9-preliminary-mapping-inputs-5f7ff93-v1/fixture.json']:
 p=B/name;sources.append(str(p))
 for q in json.load(open(p))['questions']:protected_q.add(q['qid']);protected_d.update(p['doc_id'] for p in q['pages'])
for q in protected_q:protected_d.update(p['doc_id'] for p in meta[q]['supporting_context'])
base=B/'benchmark-4e2473b/attempt-2';raw=base/'eval-quality/docprune/top4/run/results.jsonl';sources.append(str(raw));counts=collections.Counter();rows=[]
for r in map(json.loads,open(raw)):
 q=r['question_id'];counts['source_questions']+=1
 if q in protected_q:counts['excluded_qid']+=1;continue
 docs={p['doc_id'] for p in r['retrieved_pages']}|{p['doc_id'] for p in meta[q]['supporting_context']}
 if docs&protected_d:counts['excluded_document']+=1;continue
 counts['document_disjoint_remaining']+=1
 support={p['doc_id'] for p in meta[q]['supporting_context']}
 if not support<={p['doc_id'] for p in r['retrieved_pages']}:counts['support_document_missing']+=1;continue
 rows.append({'qid':q,'question':r['question'],'gold_original':r['answers'],'question_type':meta[q]['metadata']['type'],'modalities':meta[q]['metadata']['modalities'],'supporting_context':meta[q]['supporting_context'],'pages':[{**p,'source_pdf':str(B/'datasets/m3docvqa/pdfs_dev'/f"{p['doc_id']}.pdf")} for p in r['retrieved_pages']],'native_answer':r['predicted_answer'],'native_trace':r['trace'],'native_source':str(raw),'baseline':None,'all_kept_reference':None,'all_document_ids':sorted(docs),'evidence_status':'support_documents_in_top4_only; exact_evidence_page_and_readability_unverified','admission':'pending','review':None})
byid={r['qid']:r for r in rows}
roots=json.load(open(base/'diagnostics/stage245-v1-incremental/stage245-analysis.json'))['source_roots']
paths=[pathlib.Path(p) for p in roots['btp_qtp_first64']]+sorted((pathlib.Path(roots['incremental'])/'btp-qtp').glob('shard-*/run/results.jsonl'))
allkept=base/'eval-quality/all-kept/top4/run/results.jsonl'
for path in paths+[allkept]:
 if not path.exists():continue
 sources.append(str(path))
 for r in map(json.loads,open(path)):
  q=r['question_id']
  if q not in byid:continue
  c=byid[q];same=[(p['doc_id'],p['page_index']) for p in c['pages']]==[(p['doc_id'],p['page_index']) for p in r['retrieved_pages']]
  if not same:continue
  tr=r['trace'];k='all_kept_reference' if path==allkept else 'baseline'
  if k=='baseline':assert tr['post_qtp_visual_tokens']==tr['post_ctp_visual_tokens'] and tr['ctp_layer'] is None
  c[k]={'answer':r['predicted_answer'],'source':str(path),'trace':tr,'ordered_pages_match':True,'kind':'all_visual_kept' if k=='all_kept_reference' else 'btp_qtp_no_ctp'}
# Questions sharing ANY support/retrieved document must travel together.
parent={r['qid']:r['qid'] for r in rows}
def find(q):
 while parent[q]!=q:parent[q]=parent[parent[q]];q=parent[q]
 return q
owner={}
for r in rows:
 for d in r['all_document_ids']:
  if d in owner:parent[find(r['qid'])]=find(owner[d])
  else:owner[d]=r['qid']
for r in rows:r['document_component']=find(r['qid'])
rows.sort(key=lambda r:(r['baseline'] is None,hashlib.sha256(('correction-v1'+r['qid']).encode()).hexdigest()))
json.dump(rows,open(O/'candidates.json','w'),indent=2)
counts.update({'candidates':len(rows),'matching_no_ctp_baseline':sum(r['baseline'] is not None for r in rows),'matching_all_kept':sum(r['all_kept_reference'] is not None for r in rows),'document_components':len({r['document_component'] for r in rows})})
summary={'counts':dict(counts),'question_types':dict(collections.Counter(r['question_type'] for r in rows)),'protected_question_count':len(protected_q),'protected_document_count':len(protected_d),'source_sha256':{p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() for p in sources},'admitted_correction_cases':0,'status':'candidate_collection; no new model inference; no question rewritten yet','target_verified_wrong':40,'review_buffer_target':80}
json.dump(summary,open(O/'manifest.json','w'),indent=2);print(json.dumps({k:v for k,v in summary.items() if k!='source_sha256'},indent=2))
for r in rows:
 if r['baseline']:print(r['qid'],r['question'],'BASE:',r['baseline']['answer'],'GOLD:',r['gold_original'])
