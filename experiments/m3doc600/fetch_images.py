"""Fetch only selected original MMQA images from the official ZIP using HTTP ranges."""
import argparse, io, json, os, urllib.request, zipfile
from pathlib import Path
from common import read,publish,sha
URL='https://multimodalqa-images.s3-us-west-2.amazonaws.com/final_dataset_images/final_dataset_images.zip'
class RemoteZip(io.RawIOBase):
 def __init__(self,url):
  self.url=url;self.pos=0;self.bytes=0
  with urllib.request.urlopen(urllib.request.Request(url,method='HEAD'),timeout=30) as r:
   self.size=int(r.headers['Content-Length']);self.etag=r.headers['ETag']
 def seekable(self):return True
 def readable(self):return True
 def tell(self):return self.pos
 def seek(self,offset,whence=0):
  self.pos=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
  if self.pos<0:raise ValueError('negative seek')
  return self.pos
 def read(self,n=-1):
  if n<0:n=self.size-self.pos
  n=min(n,self.size-self.pos)
  if n<=0:return b''
  start=self.pos;end=start+n-1
  req=urllib.request.Request(self.url,headers={'Range':f'bytes={start}-{end}','If-Match':self.etag})
  with urllib.request.urlopen(req,timeout=60) as r:
   assert r.status==206 and r.headers['Content-Range']==f'bytes {start}-{end}/{self.size}'
   b=r.read();assert len(b)==n
  self.pos+=n;self.bytes+=n;return b

def run(root):
 assert os.environ.get('SLURM_JOB_ID')
 root=Path(root);pool=read(root/'cohort/pool.json');inv=read(root/'inventory/inventory.json')
 wanted={s['doc_id'] for q in pool['questions'] if q['metadata']['type'] in ('ImageQ','ImageListQ') for s in q['supporting_context'] if s['doc_part']=='image'}
 source=inv['sources']['MMQA_images.jsonl'];out=root/'original-images';out.mkdir(exist_ok=True)
 remote=RemoteZip(URL);records=[]
 with zipfile.ZipFile(remote) as z:
  names=z.namelist()
  for i,doc in enumerate(sorted(wanted)):
   ref=source[doc];matches=[n for n in names if n==ref['path'] or n.endswith('/'+ref['path'])]
   if len(matches)!=1:raise ValueError((doc,ref['path'],matches))
   f=out/(doc+Path(ref['path']).suffix)
   if not f.exists():
    data=z.read(matches[0])
    with f.open('xb') as stream:stream.write(data)
   records.append({'doc_id':doc,'file':str(f),'sha256':sha(f),'zip_member':matches[0],'source_title':ref['title']})
   if i%25==0:print('original images',i+1,'/',len(wanted),flush=True)
 result={'url':URL,'etag':remote.etag,'archive_bytes':remote.size,'transferred_bytes':remote.bytes,'pool_sha256':sha(root/'cohort/pool.json'),'images':records}
 publish(out/'manifest.json',result);print(json.dumps({'images':len(records),'transferred_bytes':remote.bytes,'archive_bytes':remote.size}),flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();run(a.root)
