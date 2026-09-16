"""Shared immutable identities for M3Doc600 preprocessing."""
import hashlib,json,os,tempfile,platform,resource,time
from pathlib import Path

MANIFEST_SHA='f1942130e3b585ede4474515089dda2e3df67bb78a53f8f3acf9cd8b5322a495'
BASE=('vidore/colqwen2.5-base','92908120384b7a2110c5beda3ab29cbdb2c08e49')
ADAPTER=('vidore/colqwen2.5-v0.2','dcbe8d9cede518bce830488364ba0e40c873645b')
MINER=('opendatalab/MinerU2.5-Pro-2604-1.2B','d3f5e08d073c21466bbabe21c71bb1e9c2e595da')
def fingerprint(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def publish(p,value):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 data=(json.dumps(value,indent=2,allow_nan=False)+'\n').encode()
 if p.exists():
  if p.read_bytes()!=data:raise ValueError(f'Conflicting existing record: {p}')
  return
 fd,tmp=tempfile.mkstemp(prefix='.'+p.name,dir=p.parent)
 try:
  with os.fdopen(fd,'wb') as f:f.write(data);f.flush();os.fsync(f.fileno())
  try:os.link(tmp,p)
  except FileExistsError:
   if p.read_bytes()!=data:raise ValueError(f'Conflicting concurrent record: {p}')
 finally:os.unlink(tmp)
def load_catalog(root):
 c=read(Path(root)/'catalog.json');digest=c.pop('sha256');assert fingerprint(c)==digest
 assert c['schema']=='docprune-m3doc600-pages-v1'
 assert sha(c['pool_path'])==c['manifest_sha256']
 return c,digest
def pages_for(c,shard,shards,smoke=False):
 pages=c['pages']
 if smoke:return [p for p in pages if p['key'] in c['smoke_keys']]
 assert 0<=shard<shards
 # Sort similar aspect ratios together inside each balanced page-count shard.
 pages=sorted(pages,key=lambda p:(p['width']/p['height'],p['key']))
 return pages[shard::shards]
def query_for(c,shard,shards,smoke=False):
 qs=c['questions']
 if smoke:return [q for q in qs if q['question_key'] in c['smoke_questions']]
 return sorted(qs,key=lambda q:q['question_key'])[shard::shards]
def open_image(root,p):
 from PIL import Image
 f=Path(root)/p['image'];assert sha(f)==p['image_sha256'],f
 with Image.open(f) as im:return im.convert('RGB')
def stats(start,torch=None):
 r={'elapsed_seconds':time.monotonic()-start,'host_peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'hostname':platform.node(),'slurm_job_id':os.environ.get('SLURM_JOB_ID')}
 if torch is not None:r.update(gpu=torch.cuda.get_device_name(0),gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved())
 return r
