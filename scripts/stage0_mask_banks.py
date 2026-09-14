"""Generate fixed-cost, structurally matched Stage 0 R/A banks without reader calls."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from stage0_owned_overlap import ranks, read, sha


CONFIG = {
    'schema': 'stage0-matched-ra-v1', 'seed': 20260913, 'boundary': 'B_input',
    'retained_token_fraction': .5, 'lambda': 1., 'slots_per_bank': 32,
    'shared_contexts': 8, 'additional_global_contexts': 16, 'new_local_contexts': 8,
    'local_pool_token_fractions': [.15, .5, .15, .5], 'max_proposal_attempts': 128,
    'mapping': 'positive-area-owned-support', 'A_priority': 'automatic_mix',
    'logit_transform': 'lambda * (2 * (midrank + 0.5) / n - 1)',
    'measurement_prefixes': [8,16,32],
    'epsilon_grid': [0.,.1,.25,.5], 'preference_tie_margins': [0.,.01,.05],
    'provisional_decoded_selection_epsilon': .1,
    'decoded_pool': 'Union of distinct G-only, pure-C and gold-aware(.1) winners from R and A, plus shared slot 0, R slot 1 and A slot 2 as predetermined controls; at most 9 unique masks per question.',
    'decoded_review': 'Generate once per unique mask; strip bank/teacher/family labels from adjudication packet; retain a separate lookup ledger.',
    'local_semantics': 'Each of four shared bases supplies two new equal-cost masks; outside nominated pool is fixed.',
    'status': 'unmeasured proposals; no reader outcomes or learned selector',
}


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def generator(*parts):
    seed = int(digest([CONFIG['seed'],*parts])[:16],16)
    return np.random.default_rng(seed)


def common_budget(costs, cap):
    reachable = 1
    truncate = (1 << (int(cap)+1))-1
    for cost in costs:
        reachable = (reachable | (reachable << int(cost))) & truncate
    return reachable.bit_length()-1


class CostConditionalSampler:
    """Exact subset law proportional to exp(theta @ z), conditional on cost B.

    The dynamic program is O(number_of_actions * B). It is a proposal mechanism,
    not an optimizer of reader utility. theta=0 is uniform over feasible subsets,
    which does not imply uniform inclusion of unequal-cost actions.
    """
    def __init__(self,costs,theta,budget):
        self.costs=np.asarray(costs,dtype=int)
        self.theta=np.asarray(theta,dtype=float)
        self.budget=int(budget)
        if self.costs.ndim!=1 or self.theta.shape!=self.costs.shape or np.any(self.costs<=0):
            raise ValueError('Invalid costs or weights')
        if not np.isfinite(self.theta).all() or self.budget<0:
            raise ValueError('Nonfinite weights or negative budget')
        n=len(self.costs)
        self.logz=np.full((n+1,self.budget+1),-np.inf)
        self.logz[0,0]=0
        for i,(cost,weight) in enumerate(zip(self.costs,self.theta)):
            self.logz[i+1]=self.logz[i]
            if cost<=self.budget:
                self.logz[i+1,cost:]=np.logaddexp(self.logz[i,cost:],weight+self.logz[i,:self.budget-cost+1])
        if not np.isfinite(self.logz[-1,self.budget]):
            raise ValueError('Requested cost is not achievable')

    def draw(self,rng):
        mask=np.zeros(len(self.costs),dtype=bool)
        remaining=self.budget
        for i in range(len(self.costs)-1,-1,-1):
            c=int(self.costs[i])
            log_include=self.theta[i]+self.logz[i,remaining-c] if remaining>=c else -np.inf
            probability=float(np.exp(log_include-self.logz[i+1,remaining]))
            if rng.random()<min(1.,probability):
                mask[i]=True
                remaining-=c
        if remaining or int(self.costs@mask)!=self.budget:
            raise RuntimeError('Conditional sampler violated exact cost')
        return mask


def mask_key(mask):
    return np.packbits(mask).tobytes()


def new_draw(sampler,rng,seen,base=None,pool=None):
    for attempt in range(1,CONFIG['max_proposal_attempts']+1):
        sampled=sampler.draw(rng)
        candidate=sampled if base is None else base.copy()
        if base is not None:
            candidate[pool]=sampled
        key=mask_key(candidate)
        if key not in seen:
            seen.add(key)
            return candidate,attempt
    raise ValueError('Could not produce a distinct context within the fixed proposal-attempt cap')


def local_pair(base,costs,priority,bank,case,round_index,seen):
    cap=int(CONFIG['local_pool_token_fractions'][round_index]*costs.sum())
    logits=np.zeros(len(base)) if bank=='R' else priority
    rng=generator(case,'local',round_index)  # common random numbers for R/A
    for pool_attempt in range(1,CONFIG['max_proposal_attempts']+1):
        # Soft nomination with a uniform component. Same cost-packing rule in R/A.
        soft=np.exp(logits-logits.max());soft/=soft.sum()
        weights=.5/len(base)+.5*soft
        order=np.argsort(-(np.log(weights)+rng.gumbel(size=len(base))),kind='stable')
        pool=[];used=0
        for i in order:
            if used+costs[i]<=cap:
                pool.append(int(i));used+=int(costs[i])
        pool=np.array(sorted(pool),dtype=int)
        if len(pool)<2 or not base[pool].any() or base[pool].all():
            continue
        local_budget=int(costs[pool]@base[pool])
        sampler=CostConditionalSampler(costs[pool],np.zeros(len(pool)),local_budget)
        # At least base + two alternatives, checked before making new contexts.
        if sampler.logz[-1,local_budget]<np.log(3)-1e-10:
            continue
        tentative=set(seen)
        try:
            one,a=new_draw(sampler,rng,tentative,base,pool)
            two,b=new_draw(sampler,rng,tentative,base,pool)
        except ValueError:
            continue
        seen.update(tentative)
        return [one,two],{'pool_region_indices':pool.tolist(),'pool_token_cap':cap,
            'pool_token_count':used,'pool_attempts':pool_attempt,'draw_attempts':[a,b],
            'local_retained_budget':local_budget}
    raise ValueError(f'No viable local comparison pool: {case}/{bank}/{round_index}')


def generate_banks(costs,priority,case):
    costs=np.asarray(costs,dtype=int)
    theta=CONFIG['lambda']*(2*(ranks(priority)+.5)/len(costs)-1)
    budget=common_budget(costs,int(CONFIG['retained_token_fraction']*costs.sum()))
    zero=CostConditionalSampler(costs,np.zeros(len(costs)),budget)
    seen=set();shared=[];shared_attempts=[]
    rng=generator(case,'shared')
    for _ in range(8):
        mask,attempts=new_draw(zero,rng,seen);shared.append(mask);shared_attempts.append(attempts)
    banks={}
    for bank in ['R','A']:
        seen={mask_key(m) for m in shared}
        samples={};attempts={}
        for sign,name in [(1,'positive'),(-1,'negative')]:
            sampler=zero if bank=='R' else CostConditionalSampler(costs,sign*theta,budget)
            rng=generator(case,'global',name)
            samples[name]=[];attempts[name]=[]
            for _ in range(8):
                mask,count=new_draw(sampler,rng,seen);samples[name].append(mask);attempts[name].append(count)
        locals_=[];local_meta=[]
        for round_index in range(4):
            pair,meta=local_pair(shared[2*round_index],costs,theta,bank,case,round_index,seen)
            locals_.append(pair);local_meta.append(meta)
        slots=[];comparisons=[]
        for r in range(4):
            start=len(slots)
            for j in range(2):
                slots.extend([
                    {'family':'shared','mask':shared[2*r+j],'attempts':shared_attempts[2*r+j]},
                    {'family':'global_positive','mask':samples['positive'][2*r+j],'attempts':attempts['positive'][2*r+j]},
                    {'family':'global_negative','mask':samples['negative'][2*r+j],'attempts':attempts['negative'][2*r+j]},
                    {'family':'local','mask':locals_[r][j],'base_slot':start,'pool':local_meta[r],'variant':j},
                ])
            # Predetermined diagnostic/training-comparison proposal: 24 pairs.
            # No training has occurred; report this set separately from all pairs.
            comparisons.extend([[start,start+3],[start,start+7],[start+3,start+7],
                                [start+1,start+2],[start+5,start+6],[start,start+4]])
        assert len(slots)==32 and len({mask_key(s['mask']) for s in slots})==32
        assert all(int(costs@s['mask'])==budget for s in slots)
        banks[bank]={'slots':slots,'planned_comparison_slots':comparisons}
    return budget,theta,banks


def bank_diagnostics(slots,costs,kinds):
    x=np.array([s['mask'] for s in slots],dtype=bool)
    pairs=np.triu_indices(len(x),1)
    distance=((x[:,None,:]!=x[None,:,:])*costs).sum(axis=2)[pairs]
    frequency=x.mean(axis=0)
    signatures={}
    for i in range(x.shape[1]):
        signatures.setdefault(tuple(x[:,i]),[]).append(i)
    return {'unique_masks':len({mask_key(m) for m in x}),
        'retained_region_counts':x.sum(axis=1).tolist(),
        'retained_semantic_region_counts':x[:,kinds=='mineru-region'].sum(axis=1).tolist(),
        'retained_residual_region_counts':x[:,kinds!='mineru-region'].sum(axis=1).tolist(),
        'retention_frequencies':frequency.tolist(),
        'always_retained':np.flatnonzero(frequency==1).tolist(),'always_removed':np.flatnonzero(frequency==0).tolist(),
        'co_toggled_groups':[group for pattern,group in signatures.items() if len(group)>1 and any(pattern) and not all(pattern)],
        'token_symmetric_difference_quantiles':np.quantile(distance,[0,.25,.5,.75,1]).tolist(),
        'local_token_distances_to_base':[int(costs@(s['mask']!=slots[s['base_slot']]['mask'])) for s in slots if s['family']=='local']}


def main(root,owned,out):
    out.mkdir(parents=True,exist_ok=False)
    (out/'config.json').write_text(json.dumps(CONFIG,indent=2)+'\n')
    summary=[]
    for number in range(1,18):
        q=f'Q{number:02d}'
        meta=read(owned/q/'manifest.json')
        mapping_path=root/'input/sources'/q/'mapping.json'
        if sha(mapping_path)!=meta['source_mapping_sha256']:
            raise ValueError(f'Changed action mapping: {q}')
        mapping=read(mapping_path);ids=meta['source_ids'];index={sid:i for i,sid in enumerate(ids)}
        regions={r['source_id']:r for r in mapping['sources']}
        costs=np.array([regions[sid]['token_cost'] for sid in ids],dtype=int)
        kinds=np.array([regions[sid]['source_kind'] for sid in ids])
        owners=np.array([index[sid] for sid in mapping['token_to_source']],dtype=int)
        np.testing.assert_equal(np.bincount(owners,minlength=len(ids)),costs)
        with np.load(owned/q/'profiles.npz',allow_pickle=False) as profile:
            priority=profile['automatic_mix']
        budget,theta,banks=generate_banks(costs,priority,q)
        measurements={};order=[]
        for bank,value in banks.items():
            value['diagnostics']=bank_diagnostics(value['slots'],costs,kinds)
            for slot,s in enumerate(value['slots']):
                mask=s.pop('mask')
                token_hash=hashlib.sha256(np.packbits(mask[owners]).tobytes()).hexdigest()
                mid=q+'-'+token_hash[:20]
                s['slot']=slot;s['mask_id']=mid
                if mid not in measurements:
                    measurements[mid]={'mask_id':mid,'source_mask':mask.astype(int).tolist(),
                        'retained_token_count':int(costs@mask),'token_mask_sha256':token_hash}
                else:
                    assert measurements[mid]['source_mask']==mask.astype(int).tolist()
        # Fixed round-robin measurement order; duplicate shared contexts scored once.
        for slot in range(32):
            for bank in (['R','A'] if number%2 else ['A','R']):
                mid=banks[bank]['slots'][slot]['mask_id']
                if mid not in order:order.append(mid)
        record={'case':q,'config_sha256':digest(CONFIG),'source_ids':ids,'costs':costs.tolist(),
            'token_budget':budget,'source_mapping_sha256':sha(mapping_path),
            'case_file_sha256':sha(root/'input/sources'/q/'case.json'),
            'baseline_file_sha256':sha(root/'input/sources'/q/'baseline.json'),
            'owned_profile_file_sha256':sha(owned/q/'profiles.npz'),
            'priority':priority.tolist(),'A_logits':theta.tolist(),'banks':banks,
            'unique_measurement_order':order,'measurements':measurements,
            'reader_measurements_completed':0}
        record['bank_sha256']=digest(record)
        (out/f'{q}.json').write_text(json.dumps(record,indent=2)+'\n')
        summary.append({'case':q,'token_budget':budget,'unique_masks':len(measurements),
            'R_always_removed':len(banks['R']['diagnostics']['always_removed']),
            'A_always_removed':len(banks['A']['diagnostics']['always_removed'])})
        print(q,budget,len(measurements),'unmeasured masks',flush=True)
    (out/'summary.json').write_text(json.dumps({'cases':summary,'script_sha256':sha(__file__),
        'unique_masks':sum(x['unique_masks'] for x in summary),'reader_measurements_completed':0},indent=2)+'\n')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root',type=Path,required=True)
    ap.add_argument('--owned',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();main(args.root,args.owned,args.output)
