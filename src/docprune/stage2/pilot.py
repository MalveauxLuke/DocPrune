"""Versioned variable-retention pilot; historical exact-budget contracts stay intact."""
from contextlib import nullcontext
from dataclasses import replace
import hashlib
import math
from itertools import combinations
from collections import OrderedDict
import numpy as np
import torch
from torch import nn
from .contracts import VisualMemory, fingerprint
from .policy import PolicyEncoding
from .qwen import assemble_prefill, extract_vision

VERSION = 'bernoulli-shared-reference-v1'
RETRIEVAL_SCHEMA = 'colqwen2.5-query128-owned-patch-maxsim-v1'

def seed_for(qid, stream='acquisition'):
    return int.from_bytes(hashlib.sha256(f'{VERSION}|{qid}|1729|{stream}'.encode()).digest()[:8], 'big')

def fit_omp(x, y):
    x, y = np.asarray(x,float), np.asarray(y,float)
    if y.ndim == 1: y=y[:,None]
    if x.ndim!=2 or len(x)!=len(y) or y.shape[1] not in (1,2) or not np.isfinite(y).all():
        raise ValueError('Invalid one/two-channel OMP observations')
    xm,ym=x.mean(0),y.mean(0);xc=x-xm;yc=y-ym
    norm=np.linalg.norm(xc,axis=0);z=xc/np.where(norm>0,norm,1)
    residual=yc.copy();support=[];beta=np.zeros((x.shape[1],y.shape[1]))
    for _ in range(min(4,x.shape[1])):
        a=np.linalg.norm(z.T@residual,axis=1);a[norm==0]=-1;a[support]=-1
        j=int(a.argmax())
        if a[j]<1e-10:break
        if np.linalg.matrix_rank(z[:,support+[j]])<=len(support):break
        support.append(j);c=np.linalg.lstsq(z[:,support],yc,rcond=None)[0]
        residual=yc-z[:,support]@c;beta[support]=c/norm[support,None]
    association=np.linalg.norm(z.T@yc,axis=1)
    return dict(coefficients=beta.tolist(),intercept=(ym-xm@beta).tolist(),support=support,
                association=association.tolist(),constant_columns=np.flatnonzero(norm==0).tolist(),
                rank=int(np.linalg.matrix_rank(xc)),channels=y.shape[1])

def next_probe(rows, costs, qid, correct, reference, *, count=32, random_only=False):
    """Pure replayable next decision; predictions are journaled before measuring."""
    r=len(costs);n=len(rows)
    if n>=min(count,2**r):return None
    seen={tuple(z['mask']) for z in rows};rng=np.random.default_rng(seed_for(qid))
    def random_probe(reason):
        for _ in range(10000):
            z=tuple(bool(v) for v in rng.integers(0,2,r))
            if z not in seen:return dict(mask=list(z),kind='broad',reason=reason)
        return None
    if n<22 or random_only:return random_probe('independent Bernoulli; no retention repair')
    channels=lambda row:[row['s']] if correct else [row['g'],row['s']]
    fit=fit_omp([z['mask'] for z in rows[:22]],[channels(z) for z in rows[:22]])
    beta=np.asarray(fit['coefficients']);intercept=np.asarray(fit['intercept'])
    ties=lambda j:hashlib.sha256(f'{seed_for(qid)}|{j}'.encode()).hexdigest()
    support=sorted(fit['support'],key=lambda j:(-float(np.linalg.norm(beta[j])),ties(j)))
    others=sorted((j for j in range(r) if j not in support and j not in fit['constant_columns']),key=lambda j:(-fit['association'][j],ties(j)))
    nominees=(support+others)[:4]
    if n<24:
        p=random_probe('fresh check of frozen 22-mask OMP')
        if p is not None:p.update(kind='fresh_check',prediction=(np.asarray(p['mask'])@beta+intercept).tolist(),omp=fit)
        return p
    base=rows[:24];costs=np.asarray(costs);half=costs.sum()/2
    admissible=lambda row:row['g']>=reference['g']-.1
    near=lambda i:abs(np.asarray(base[i]['mask'])@costs-half)
    eligible=([i for i,z in enumerate(base) if z['s']>=max(t['s'] for t in base)-.05] if correct else [i for i,z in enumerate(base) if admissible(z)])
    bi=min(eligible,key=lambda i:(near(i),ties(i))) if eligible else min(range(24),key=lambda i:(-base[i]['g'],near(i),ties(i)))
    b=np.array(base[bi]['mask'],bool);outside=[j for j in range(r) if j not in nominees]
    eligible2=list(range(24)) if correct else ([i for i,z in enumerate(base) if admissible(z)] or list(range(24)))
    bj=min(eligible2,key=lambda i:(-sum(b[outside]!=np.array(base[i]['mask'])[outside]),ties(i)))
    bp=np.array(base[bj]['mask'],bool);background_distance=int(sum(b[outside]!=bp[outside]))
    lookup={tuple(z['mask']):z for z in rows}
    def flip(bits,ix):
        z=bits.copy();z[ix]=~z[ix];return z
    effects={};conflicts={}
    for j in nominees:
        endpoint=lookup.get(tuple(flip(b,[j])))
        delta=np.asarray(channels(endpoint))-channels(base[bi]) if endpoint else np.zeros(beta.shape[1])
        effects[j]=float(abs(delta).max())
        oriented=delta*(1 if not b[j] else -1)
        conflicts[j]=bool(((oriented*beta[j]<0)&(abs(oriented)>.05)).any())
    kind='single_flip';background=b; candidates=[[j] for j in nominees]
    if n>=28:
        slot=n-28
        if slot==0:
            kind='square';ranked=sorted(nominees,key=lambda j:(-effects[j],nominees.index(j)))
            candidates=[list(pair) for pair in combinations(ranked,2)]
        elif slot in (1,2):
            kind='background_flip';background=bp
            ordered=sorted(nominees,key=lambda j:(not conflicts[j],effects[j]>.05,-effects[j],nominees.index(j)))
            candidates=[[j] for j in ordered] if background_distance else []
        else:
            kind='outside_flip';remaining=[j for j in outside if tuple(flip(b,[j])) not in seen]
            rng2=np.random.default_rng(seed_for(qid,'outside'))
            ordered=sorted(remaining,key=lambda j:(-fit['association'][j],ties(j)))
            if remaining and rng2.random()<.5:ordered=list(rng2.permutation(remaining))
            candidates=[[int(j)] for j in ordered]
    for ix in candidates:
        z=flip(background,ix)
        if tuple(z) not in seen:
            return dict(mask=z.tolist(),kind=kind,flipped=ix,reference=bi if kind!='background_flip' else bj,
                        nominees=nominees,background_distance=background_distance,
                        prediction=(z@beta+intercept).tolist(),omp=fit,
                        reason='fixed 22+2+4+4 policy; cached duplicates skipped')
    p=random_probe('no new eligible endpoint of scheduled type; broad fallback')
    if p:p['fallback_from']=kind
    return p

def encoding_cache_key(inputs,prompt,native_layout=None):
    layout=native_layout if native_layout is not None else inputs.layout
    return fingerprint((inputs.identity.key,prompt.input_ids.cpu().tolist(),prompt.image_grid_thw.cpu().tolist(),layout.owner.cpu().tolist()))


class CachedPilotSelector(nn.Module):
    """Cache frozen vision across phases; language memories only while LoRA is identity."""
    def __init__(self, base, cache_entries=2, disk_cache=None):
        super().__init__();self.base=base;self.phase='lora'
        self.cache_entries=cache_entries;self.disk_cache=disk_cache;self.vision_cache=OrderedDict();self.hidden_cache=OrderedDict()
        self.calls={'vision':0,'language':0,'vision_hits':0,'hidden_hits':0}
    @property
    def correction(self):return self.base.correction
    def set_phase(self,phase):
        if phase not in ('warmup','frozen','lora'):raise ValueError('Unknown training phase')
        if phase != self.phase:self.hidden_cache.clear()
        if phase!='lora' and any(torch.count_nonzero(p).item() for n,p in self.base.backbone.named_parameters() if 'lora_B' in n):
            raise ValueError('Frozen hidden cache requires restored identity LoRA')
        self.phase=phase
        for n,p in self.base.backbone.named_parameters():p.requires_grad_('lora_' in n and phase=='lora')
        if phase=='lora':self.hidden_cache.clear()
    def forward(self,inputs,prompt,*,native_layout=None,ledger=None,return_encoding=False,bypass_cache=False):
        inputs.validate()
        if inputs.retrieval is None or inputs.retrieval.schema!=RETRIEVAL_SCHEMA:raise ValueError('Required ColQwen fusion missing')
        if inputs.budget!=inputs.layout.owner.numel():raise ValueError('Pilot utility capacity must be full')
        layout=replace(native_layout,metadata=inputs.layout.metadata) if native_layout is not None else inputs.layout
        if layout.region_ids!=inputs.layout.region_ids:raise ValueError("Native action identity differs")
        key=encoding_cache_key(inputs,prompt,layout)
        device=prompt.input_ids.device
        if self.disk_cache is not None and not bypass_cache:
            if self.phase!='lora' and key not in self.hidden_cache:
                saved=self.disk_cache.get('identity-language',key)
                if saved is not None:self.hidden_cache[key]=(saved['visual'],saved['question'])
            if key not in self.vision_cache and key not in self.hidden_cache:
                saved=self.disk_cache.get('native-vision',key)
                if saved is not None:self.vision_cache[key]=VisualMemory(saved['merged'],tuple(saved[k] for k in sorted(saved) if k.startswith('deepstack_')),saved['grid'],key)
        if self.phase!='lora' and key in self.hidden_cache and not bypass_cache:
            visual,question=(v.to(device) for v in self.hidden_cache[key]);self.calls['hidden_hits']+=1
        else:
            with torch.no_grad() if self.phase!='lora' else nullcontext():
                if key in self.vision_cache and not bypass_cache:
                    v=self.vision_cache[key];vision=VisualMemory(v.merged.to(device),tuple(x.to(device) for x in v.deepstack),v.grid_thw.to(device),v.provenance);self.calls['vision_hits']+=1
                else:
                    vision=extract_vision(self.base.backbone,prompt,provenance=key);self.calls['vision']+=1
                    self.vision_cache[key]=VisualMemory(vision.merged.cpu(),tuple(v.cpu() for v in vision.deepstack),vision.grid_thw.cpu(),key)
                    if self.disk_cache is not None:self.disk_cache.put('native-vision',key,dict(merged=vision.merged,grid=vision.grid_thw,**{f'deepstack_{i}':v for i,v in enumerate(vision.deepstack)}))
                packed,_,_=assemble_prefill(self.base.backbone,prompt,vision,question_first=True)
                output=self.base.backbone.model.language_model(**packed,use_cache=False,output_hidden_states=False,return_dict=True)
                hidden=output.last_hidden_state[0];self.calls['language']+=1
                visual=hidden[prompt.input_ids[0]==self.base.backbone.config.image_token_id];question=hidden[prompt.question_positions]
                if self.phase!='lora':
                    self.hidden_cache[key]=(visual.detach().cpu(),question.detach().cpu())
                    if self.disk_cache is not None:self.disk_cache.put('identity-language',key,dict(visual=visual,question=question))
        for cache in (self.vision_cache,self.hidden_cache):
            if key in cache:cache.move_to_end(key)
            while len(cache)>self.cache_entries:cache.popitem(last=False)
        e=self.base.reader(visual,question,layout,1.,inputs.retrieval)
        scores=self.base.head(e)
        return PolicyEncoding(e,scores) if return_encoding else scores
    def cache_bytes(self):
        return sum(t.numel()*t.element_size() for v in self.vision_cache.values() for t in (v.merged,*v.deepstack))+sum(t.numel()*t.element_size() for pair in self.hidden_cache.values() for t in pair)

def phase_optimizer(model,phase):
    model.set_phase(phase);groups={}
    for n,p in model.named_parameters():
        if not p.requires_grad:continue
        lr=2e-5 if 'lora_' in n else (3e-4 if phase=='warmup' else 1e-4)
        decay=0. if p.ndim<2 or n.endswith('.bias') else .01
        groups.setdefault((lr,decay),[]).append(p)
    return torch.optim.AdamW([dict(params=p,lr=lr,weight_decay=decay) for (lr,decay),p in groups.items()],betas=(.9,.999),eps=1e-8)

def train_epoch(model,batches,config,optimizer,*,accumulation=4,scheduler=None):
    from .training import objective
    model.train();optimizer.zero_grad(set_to_none=True);pending=0;updates=[];losses=[]
    def step():
        for p in model.parameters():
            if p.grad is not None:p.grad.div_(pending)
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        if not torch.isfinite(norm):raise ValueError('Nonfinite gradients')
        optimizer.step()
        if scheduler:scheduler.step()
        updates.append(float(norm));optimizer.zero_grad(set_to_none=True)
    for batch in batches:
        loss=objective(model,config=config,**batch)
        if loss is None:continue
        if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
        loss.backward();pending+=1;losses.append(float(loss.detach()))
        if pending==accumulation:step();pending=0
    if pending:step()
    return dict(losses=losses,updates=len(updates),gradient_norms=updates,clipped=sum(n>1 for n in updates))


def branch_scheduler(optimizer, total_updates):
    """Ten-percent linear warm-up then cosine; invoked per optimizer update."""
    warmup=max(1,math.ceil(total_updates*.1))
    def scale(step):
        if step<warmup:return (step+1)/warmup
        return .5*(1+math.cos(math.pi*min(1.,(step-warmup)/max(1,total_updates-warmup))))
    return torch.optim.lr_scheduler.LambdaLR(optimizer,scale)
