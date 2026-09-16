"""Small visual inspection sheets, preserving source/PDF image distinctions."""
import argparse,json,os
from pathlib import Path
from common import read,sha,publish

def run(root,corpus):
 import pypdfium2 as pdfium
 from PIL import Image,ImageDraw,ImageFont,ImageOps
 assert os.environ.get('SLURM_JOB_ID')
 root=Path(root);refs=read(root/'original-images/manifest.json')['images'];font=ImageFont.truetype('/usr/share/fonts/dejavu/DejaVuSans.ttf',14)
 items=[]
 for ref in refs:
  r=read(root/'evidence/image-matches-v2'/(ref['doc_id']+'.json'))
  if r['best_candidates']:items.append((r['best_candidates'][0]['mae_rgb64'],ref,r))
 items.sort(key=lambda x:x[0]);chosen=items[:6]+items[len(items)//2:len(items)//2+6]+items[-6:]
 out=root/'review-v2';out.mkdir(exist_ok=True);ledger=[]
 for si in range(3):
  sheet=Image.new('RGB',(1100,1500),'white');draw=ImageDraw.Draw(sheet)
  for j,(err,ref,r) in enumerate(chosen[si*6:(si+1)*6]):
   y=j*250;hit=r['best_candidates'][0]
   with Image.open(ref['file']) as im:
    rgba=ImageOps.exif_transpose(im).convert('RGBA');src=Image.alpha_composite(Image.new('RGBA',rgba.size,'white'),rgba).convert('RGB');src.thumbnail((430,210));sheet.paste(src,(0,y+35))
   with pdfium.PdfDocument(str(Path(corpus)/'pdfs_dev'/(ref['doc_id']+'.pdf'))) as pdf:
    page=pdf[hit['page_index']];objs=list(page.get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_IMAGE]));bm=objs[hit['object_index']].get_bitmap(render=True);rgba=bm.to_pil().convert('RGBA');im=Image.alpha_composite(Image.new('RGBA',rgba.size,'white'),rgba).convert('RGB');im.thumbnail((430,210));sheet.paste(im,(550,y+35));im.close();bm.close();page.close()
   title=ref['source_title'];draw.text((4,y),f'{title[:58]} | {ref["doc_id"][:8]} | error {err:.4f}',fill='black',font=font)
   draw.text((4,y+18),'Original annotated image',fill='black',font=font);draw.text((554,y+18),f'Closest PDF object: page {hit["page_index"]+1}',fill='black',font=font)
   ledger.append({'sheet':si,'row':j,'doc_id':ref['doc_id'],'mae':err,'page_index':hit['page_index']})
  sheet.save(out/f'image-comparison-{si}.png')
 publish(out/'image-comparison-index.json',ledger)
 print('Wrote three source-versus-PDF inspection sheets',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--corpus',required=True);a=p.parse_args();run(a.root,a.corpus)
