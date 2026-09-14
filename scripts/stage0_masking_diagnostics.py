"""Exploratory matched-null and pair diagnostics over an existing Stage 0 audit."""
import argparse
import sys,json,csv
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent))
from stage0_masking_audit import read,dump,csvout,coverage
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--root',type=Path,required=True)
parser.add_argument('--audit',type=Path,required=True)
args=parser.parse_args()
out=args.audit;root=args.root
rows=list(csv.DictReader((out/'regions.csv').open())); rng=np.random.default_rng(60311)
null=[]
for n in range(1,18):
 q=f'Q{n:02d}';rs=sorted([r for r in rows if r['case']==q],key=lambda r:int(r['region_index']));cost=np.array([int(r['token_cost']) for r in rs]);kind=np.array([r['kind'] for r in rs]);bins=np.searchsorted(np.quantile(cost,[.2,.4,.6,.8]),cost);groups=[np.where((kind==k)&(bins==b))[0] for k in sorted(set(kind)) for b in range(5)];k=int(np.ceil(.2*len(rs)))
 for score in ['all_maxsim','lex_maxsim','focus','exclusive_centroid','query_peak','automatic_mix']:
  p=np.array([float(r[score]) for r in rs]);sel=np.argsort(-p,kind='stable')[:k];masses={'negative_G':np.maximum(-np.array([float(r['input_G']) for r in rs]),0),'negative_C':np.maximum(-np.array([float(r['input_C']) for r in rs]),0),'positive_S':np.maximum(np.array([float(r['input_S']) for r in rs]),0)};values={t:[] for t in masses}
  for z in range(400):
   perm=p.copy()
   for g in groups:perm[g]=rng.permutation(p[g])
   ii=np.argsort(-perm,kind='stable')[:k]
   for t,m in masses.items(): values[t].append(coverage(m,ii))
  for t,m in masses.items():null.append({'case':q,'score':score,'target':t,'actual':coverage(m,sel),'matched_kind_cost_null':float(np.mean(values[t])),'null_q025':float(np.quantile(values[t],.025)),'null_q975':float(np.quantile(values[t],.975))})
csvout(out/'size-kind-matched-null.csv',null)
pairs=[('Q02',75,76,'snow photograph','canopy caption'),('Q07',24,2,'49-species lead','66-species-and-subspecies paragraph'),('Q11',11,7,'Championship 1998-2002 table','Series 2012+ table'),('Q12',22,59,'football photograph','Adele references'),('Q13',29,28,'theme table','Paramore-to-Decode bridge paragraph'),('Q04',27,24,'screenplay infobox','screenplay prose'),('Q15',38,6,'Greyhound highball field','Rickey lead')]
inter=[]
for q,a,b,an,bn in pairs:
 for depth in ['input','intermediate']:
  d=read(root/'input/sources'/q/depth/'comparison-rp105.json');x=np.array([m['vector'] for m in d['design']['fit_masks']],bool);gs=np.array(d['raw_likelihoods']);y=np.c_[gs,gs[:,0]-gs[:,1]]
  cells={f'{aa}{bb}':y[(x[:,a]==aa)&(x[:,b]==bb)] for aa in [0,1] for bb in [0,1]};means={k:v.mean(0) for k,v in cells.items()}
  contrasts={'B_effect_A_present':means['11']-means['10'],'B_effect_A_absent':means['01']-means['00'],'interaction':means['11']-means['10']-means['01']+means['00']}
  boots=[]
  for _ in range(1500):
   mm={k:v[rng.integers(len(v),size=len(v))].mean(0) for k,v in cells.items()};boots.append(mm['11']-mm['10']-mm['01']+mm['00'])
  boot=np.array(boots)
  row={'case':q,'depth':depth,'A_index':a,'B_index':b,'A':an,'B':bn,'cell_counts':{k:len(v) for k,v in cells.items()},'cell_means_GSC':{k:v.tolist() for k,v in means.items()},'contrasts_GSC':{k:v.tolist() for k,v in contrasts.items()},'interaction_bootstrap_95':np.quantile(boot,[.025,.975],axis=0).tolist(),'interpretation':'Bernoulli-bank average over varying backgrounds; not a fixed-background factorial replay; exploratory selected pairs'};inter.append(row)
dump(out/'pair-diagnostics.json',inter)
for target in ['negative_G','negative_C','positive_S']:
 a=[r for r in null if r['score']=='all_maxsim' and r['target']==target];print(target,'raw',np.mean([r['actual'] for r in a]),'size_kind_null',np.mean([r['matched_kind_cost_null'] for r in a]))
for row in inter:
 if row['depth']=='input':print(row['case'],'B added with A',np.round(row['contrasts_GSC']['B_effect_A_present'],3).tolist(),'interaction',np.round(row['contrasts_GSC']['interaction'],3).tolist(),'CI',np.round(row['interaction_bootstrap_95'],2).tolist())
