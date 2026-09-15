"""Seal canonical pages once; no model imports or inference."""
import argparse,hashlib,json,math,os,time
from pathlib import Path
from common import MANIFEST_SHA,sha,read,publish,fingerprint,stats

def main(root):
 from PIL import Image
 import pypdfium2 as pdfium
 import importlib.metadata as metadata
 root=Path(root);c=root/'input/cohorts/initial-300-v1';assert sha(c/'manifest.json')==MANIFEST_SHA
 if (root/'prepare-completion.json').exists():
  from common import load_catalog
  catalog,digest=load_catalog(root);receipt=read(root/'prepare-completion.json')
  assert receipt['catalog_sha256']==digest
  for page in catalog['pages']:assert sha(root/page['image'])==page['image_sha256']
  print('Verified existing completed catalog',digest,flush=True);return
 manifest=read(c/'manifest.json');pages={};documents={};questions=[];start=time.monotonic()
 for source in ['slidevqa','dude','tatdqa']:
  material=read(c/f'{source}_materialization.json');assert material['cohort_manifest_sha256']==MANIFEST_SHA
  for doc_id,d in material['documents'].items():
   doc_key=f'{source}:{doc_id}';logical=[];pdf=None
   if source=='dude':
    f=root/'input'/d['pdf'];assert sha(f)==d['pdf_sha256'];pdf=pdfium.PdfDocument(str(f))
    assert len(pdf)==len(d['pages'])
   for item in d['pages']:
    n=item['page_number'];pid=f'{doc_key}:{n}'
    if source=='dude':
     page=pdf[n-1];bitmap=page.render(scale=200/72);image=bitmap.to_pil().convert('RGB');bitmap.close();page.close()
     provenance={'engine':'pypdfium2','version':metadata.version('pypdfium2'),'dpi':200,'pdf_sha256':d['pdf_sha256'],'pdf_page_index':n-1}
     image_file=None
    else:
     image_file=root/'input'/item['path'];assert sha(image_file)==item['sha256']
     with Image.open(image_file) as im:
      assert im.getexif().get(274,1)==1,('orientation',pid)
      image=im.convert('RGB')
     provenance={'engine':'released_image','source_image_sha256':item['sha256']}
    w,h=image.size;rgb_hash=hashlib.sha256(image.tobytes()).hexdigest();key=fingerprint([w,h,rgb_hash])
    if key not in pages:
     if image_file is None:
      image_file=root/'rendered'/f'{key}.png';image_file.parent.mkdir(parents=True,exist_ok=True)
      if image_file.exists():
       with Image.open(image_file) as old:assert hashlib.sha256(old.convert('RGB').tobytes()).hexdigest()==rgb_hash
      else:image.save(image_file)
     pages[key]={'key':key,'image':str(image_file.relative_to(root)),'image_sha256':sha(image_file),'rgb_sha256':rgb_hash,'width':w,'height':h,'aliases':[]}
    pages[key]['aliases'].append({'page_id':pid,'source':source,'document_id':doc_id,'page_number':n,'rendering':provenance})
    logical.append({'page_id':pid,'key':key,'page_number':n});image.close()
   if pdf is not None:pdf.close()
   documents[doc_key]=logical
  print('Prepared',source,len(pages),'unique pages',flush=True)
 for q in manifest['questions']:
  questions.append({'question_key':q['question_key'],'source':q['source'],'document_id':q['document_id'],'question':q['question'],'pages':documents[f"{q['source']}:{q['document_id']}"]})
 assert len(questions)==300 and sum(len(p) for p in documents.values())==3743
 # General smoke coverage: each source, portrait/landscape, and smallest/largest raster.
 smoke=set();smoke_q=[]
 for source in ['slidevqa','dude','tatdqa']:
  candidates=[p for p in pages.values() if any(a['source']==source for a in p['aliases'])]
  for key in [lambda p:p['width']/p['height'],lambda p:p['width']*p['height']]:
   ordered=sorted(candidates,key=key);smoke.update([ordered[0]['key'],ordered[-1]['key']])
  source_q=sorted((q for q in questions if q['source']==source),key=lambda q:q['question_key'])
  smoke_q.extend(q['question_key'] for q in source_q if any(p['key'] in smoke for p in q['pages']))
  smoke_q.append(max(source_q,key=lambda q:len(q['question']))['question_key'])
 result={'schema':'docprune-initial300-pages-v1','manifest_sha256':MANIFEST_SHA,'pages':sorted(pages.values(),key=lambda p:p['key']),'questions':questions,'documents':documents,'logical_page_count':3743,'unique_page_count':len(pages),'smoke_keys':sorted(smoke),'smoke_questions':sorted(set(smoke_q))}
 result['sha256']=fingerprint(result);publish(root/'catalog.json',result)
 publish(root/'prepare-completion.json',{'catalog_sha256':result['sha256'],'logical_pages':3743,'unique_pages':len(pages),'questions':300,'stats':stats(start),'models_run':0})
 print(json.dumps({'unique_pages':len(pages),'logical_pages':3743,'smoke_pages':len(smoke),'catalog_sha256':result['sha256']}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);main(p.parse_args().root)
