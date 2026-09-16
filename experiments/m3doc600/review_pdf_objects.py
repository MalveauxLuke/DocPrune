"""Inspect every meaningful PDF image object for a specified source-image QID."""
import argparse
from pathlib import Path
from common import read,publish

def run(root,corpus,qid):
 import pypdfium2 as pdfium
 from PIL import Image,ImageDraw,ImageFont,ImageOps
 root=Path(root);q=next(q for q in read(root/'cohort/pool.json')['questions'] if q['qid']==qid)
 refs={r['doc_id']:r for r in read(root/'original-images/manifest.json')['images']}
 font=ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans.ttf',14);records=[];tiles=[]
 def rgb(im):
  im=ImageOps.exif_transpose(im).convert('RGBA');return Image.alpha_composite(Image.new('RGBA',im.size,'white'),im).convert('RGB')
 for s in q['supporting_context']:
  doc=s['doc_id']
  if doc not in refs:continue
  with Image.open(refs[doc]['file']) as im:tiles.append(('ANNOTATED SOURCE '+doc[:8],rgb(im)))
  with pdfium.PdfDocument(str(Path(corpus)/'pdfs_dev'/(doc+'.pdf'))) as pdf:
   for pi in range(len(pdf)):
    page=pdf[pi]
    for oi,obj in enumerate(page.get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_IMAGE])):
     try:
      bm=obj.get_bitmap(render=True);im=rgb(bm.to_pil());bm.close();w,h=im.size
      records.append({'doc_id':doc,'page_index':pi,'object_index':oi,'width':w,'height':h})
      if min(w,h)>=48 and max(w,h)>=100:tiles.append((f'{doc[:8]} PDF page {pi+1} object {oi}',im))
      else:im.close()
     except Exception as e:records.append({'doc_id':doc,'page_index':pi,'object_index':oi,'error':str(e)})
    page.close()
 out=root/'review-all-objects';out.mkdir(exist_ok=True)
 for start in range(0,len(tiles),12):
  part=tiles[start:start+12];sheet=Image.new('RGB',(1200,120+260*((len(part)+3)//4)),'white');draw=ImageDraw.Draw(sheet)
  draw.text((5,5),q['question'][:145],fill='black',font=font);draw.text((5,30),'GOLD: '+str([a['answer'] for a in q['answers']]),fill='black',font=font)
  for i,(label,im) in enumerate(part):
   x=(i%4)*300;y=90+(i//4)*260;draw.text((x,y),label,fill='black',font=font);im.thumbnail((290,230));sheet.paste(im,(x,y+25));im.close()
  sheet.save(out/f'{qid}-{start//12}.png')
 publish(out/(qid+'.json'),{'question':q,'objects':records,'tiles':len(tiles)})
 print(qid,'objects',len(records),'visible tiles',len(tiles),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--corpus',required=True);p.add_argument('--qid',required=True);a=p.parse_args();run(a.root,a.corpus,a.qid)
