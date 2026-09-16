"""CPU inspection of completed smoke tensors, ranking differences and layouts."""
import argparse,json,os
from pathlib import Path
from common import read,publish,fingerprint,sha

def run(root):
 import torch
 from safetensors.torch import load_file
 from PIL import Image,ImageDraw,ImageFont
 assert os.environ.get('SLURM_JOB_ID');torch.set_num_threads(1)
 root=Path(root);c=read(root/'smoke/catalog.json');summary={'colqwen':{},'mineru':{},'ranking_comparisons':[]}
 for engine,batches in [('colqwen',[4,8,16]),('mineru',[8,16])]:
  for b in batches:
   base=root/'smoke'/f'{engine}-b{b}'/engine;runs=list((base/'runs').glob('*.json'));assert len(runs)==1
   r=read(runs[0]);assert r['processed_pages']==len(c['pages'])
   summary[engine][str(b)]={'batch_seconds':sum(x['seconds'] for x in r['batch_timings']),'stats':r['stats'],'run_sha256':sha(runs[0])}
 scores={}
 for b in [4,8,16]:
  base=root/'smoke'/f'colqwen-b{b}'/'colqwen';scores[b]={}
  for q in c['questions']:
   query=load_file(str(base/'queries'/(fingerprint(q['question_key'])+'.safetensors')))['embeddings']
   values={}
   for p in q['pages']:
    page=load_file(str(base/'pages'/(p['key']+'.safetensors')))['embeddings'];assert torch.isfinite(page).all()
    values[p['key']]=float((query@page.T).max(dim=1).values.sum())
   scores[b][q['question_key']]=values
 for q in c['questions']:
  key=q['question_key'];base=scores[4][key];order=sorted(base,key=lambda k:(-base[k],k))
  for b in [8,16]:
   other=scores[b][key];o=sorted(other,key=lambda k:(-other[k],k))
   summary['ranking_comparisons'].append({'qid':key,'batch':b,'same_order':o==order,'same_top1':o[0]==order[0],'max_score_difference':max(abs(base[k]-other[k]) for k in base),'reference_scores':base,'scores':other})
 differing=[]
 for p in c['pages']:
  a=read(root/'smoke/mineru-b8/mineru/pages'/(p['key']+'.json'));b=read(root/'smoke/mineru-b16/mineru/pages'/(p['key']+'.json'))
  if a['regions']!=b['regions']:differing.append(p['key'])
 summary['mineru_pages_with_layout_differences']=differing
 out=root/'review-smoke';out.mkdir(exist_ok=True);font=ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans.ttf',14)
 chosen=[]
 for q in c['questions']:
  chosen.append(next(p for p in c['pages'] if p['key']==q['pages'][0]['key']))
 for group in range(2):
  sheet=Image.new('RGB',(1200,1500),'white');draw=ImageDraw.Draw(sheet)
  for i,p in enumerate(chosen[group*4:group*4+4]):
   with Image.open(p['image']) as src:im=src.convert('RGB');im.thumbnail((590,700))
   d=ImageDraw.Draw(im);w,h=im.size;r=read(root/'smoke/mineru-b16/mineru/pages'/(p['key']+'.json'))
   for j,reg in enumerate(r['regions']):
    x0,y0,x1,y1=reg['bbox'];d.rectangle((x0*w,y0*h,x1*w,y1*h),outline=['red','blue','green','purple'][j%4],width=2)
   x=(i%2)*600;y=(i//2)*750;draw.text((x,y),f'{p["aliases"][0]["document_id"][:12]} page {p["aliases"][0]["page_number"]}: {len(r["regions"])} regions',font=font,fill='black');sheet.paste(im,(x,y+30));im.close()
  sheet.save(out/f'layout-sheet-{group}.png')
 publish(out/'smoke-inspection.json',summary)
 print(json.dumps({'colqwen':summary['colqwen'],'mineru':summary['mineru'],'rank_changes':sum(not r['same_order'] for r in summary['ranking_comparisons']),'top1_changes':sum(not r['same_top1'] for r in summary['ranking_comparisons']),'layout_difference_pages':len(differing)},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(a.root)
