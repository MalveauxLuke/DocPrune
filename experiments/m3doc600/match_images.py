"""Locate original source photographs in PDF image objects without a learned model."""
import argparse, json, os
from pathlib import Path
from common import read,publish,sha

def run(root,corpus):
 import numpy as np
 import pypdfium2 as pdfium
 from PIL import Image,ImageOps
 assert os.environ.get('SLURM_JOB_ID')
 root=Path(root);manifest=read(root/'original-images/manifest.json');out=root/'evidence/image-matches-v2';rows=[]
 for ix,ref in enumerate(manifest['images']):
  target=out/(ref['doc_id']+'.json')
  if target.exists():rows.append(read(target));continue
  path=Path(corpus)/'pdfs_dev'/(ref['doc_id']+'.pdf')
  def canonical(im):
   im=ImageOps.exif_transpose(im).convert('RGBA');bg=Image.new('RGBA',im.size,'white');return Image.alpha_composite(bg,im).convert('RGB')
  with Image.open(ref['file']) as im:source=canonical(im)
  sw,sh=source.size;original=np.asarray(source.resize((64,64),Image.Resampling.LANCZOS),dtype=np.float32)/255
  candidates=[]
  with pdfium.PdfDocument(str(path)) as pdf:
   for pi in range(len(pdf)):
    page=pdf[pi]
    for oi,obj in enumerate(page.get_objects(filter=[pdfium.raw.FPDF_PAGEOBJ_IMAGE])):
     try:
      bm=obj.get_bitmap(render=True);im=canonical(bm.to_pil());w,h=im.size
      if min(w,h)<24 or abs((w/h)/(sw/sh)-1)>.03:im.close();bm.close();continue
      arr=np.asarray(im.resize((64,64),Image.Resampling.LANCZOS),dtype=np.float32)/255
      diff=np.abs(arr-original);mae=float(diff.mean());p99=float(np.quantile(diff,.99))
      exact=im.size==source.size and im.tobytes()==source.tobytes()
      candidates.append({'page_index':pi,'object_index':oi,'width':w,'height':h,'mae_rgb64':mae,'p99_error':p99,'exact_rgb_identity':exact})
      im.close();bm.close()
     except Exception as e:
      # Extraction failures remain explicit and never count as a match.
      candidates.append({'page_index':pi,'object_index':oi,'error':type(e).__name__+': '+str(e)[:180]})
    page.close()
  source.close();valid=sorted((c for c in candidates if 'mae_rgb64' in c),key=lambda c:c['mae_rgb64'])
  exact=[c for c in valid if c['exact_rgb_identity']]
  near=[c for c in valid if c['mae_rgb64']<=.01 and c['p99_error']<=.08]
  r={'doc_id':ref['doc_id'],'pdf_sha256':sha(path),'source_image_sha256':ref['sha256'],'source_size':[sw,sh],'canonicalization':'EXIF orientation and white alpha compositing; PDF rendered image object with masks',
     'status':'exact_pixel_identity' if exact else 'near_pixel_identity_requires_visual_review' if near else 'no_strict_match',
     'exact_matches':exact,'near_matches':near,'best_candidates':valid[:5],'extraction_errors':[c for c in candidates if 'error' in c]}
  publish(target,r);rows.append(r)
  if ix%20==0:print('matched images',ix+1,'/',len(manifest['images']),flush=True)
 from collections import Counter
 summary={'images':len(rows),'statuses':dict(Counter(r['status'] for r in rows)),'models_run':0}
 publish(out/'summary.json',summary);print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--corpus',required=True);a=p.parse_args();run(a.root,a.corpus)
