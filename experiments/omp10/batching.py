"""Length-aware mask batching, bounded tuning and explicit numerical fallback."""
import gc
import time
import torch
from torch.nn import functional as F
from docprune.stage2.qwen import assemble_prefill


@torch.no_grad()
def score_batch(reader, prompt, memory, retained_sets, gold, own):
    """Independent full-prefix scoring; left padding never changes real positions.

    All rows are masks of one question. DeepStack features follow boolean indexing
    order (batch row, then original retained-token order). No KV branch reuse.
    """
    if len(retained_sets)==1:
        return [reader.teacher_scores(prompt,memory,retained_sets[0],[gold],own,reuse_prefill=False)]
    model=reader.model
    prefixes=[assemble_prefill(model,prompt,memory,r) for r in retained_sets]
    answers=[]
    for target in [gold,own]:
        target=torch.tensor(target,dtype=torch.long,device=prompt.input_ids.device)
        tail=target[:-1]; width=max(p[0]['inputs_embeds'].shape[1] for p in prefixes)+len(tail)
        template=prefixes[0][0]['inputs_embeds']; count=len(prefixes)
        emb=template.new_zeros((count,width,template.shape[-1]))
        attention=torch.zeros((count,width),dtype=torch.long,device=emb.device)
        positions=torch.zeros((3,count,width),dtype=torch.long,device=emb.device)
        visual=torch.zeros((count,width),dtype=torch.bool,device=emb.device)
        tail_emb=model.model.get_input_embeddings()(tail[None])
        for row,(packed,_,start) in enumerate(prefixes):
            length=packed['inputs_embeds'].shape[1]; offset=width-length-len(tail)
            emb[row,offset:offset+length]=packed['inputs_embeds'][0]
            emb[row,offset+length:]=tail_emb[0]
            attention[row,offset:]=1
            positions[:,row,offset:offset+length]=packed['position_ids'][:,0]
            positions[:,row,offset+length:]=torch.arange(start,start+len(tail),device=emb.device)[None]
            visual[row,offset:offset+length]=packed['visual_pos_masks'][0]
        streams=[torch.cat([p[0]['deepstack_visual_embeds'][i] for p in prefixes]) for i in range(len(memory.deepstack))]
        out=model.model.language_model(inputs_embeds=emb,attention_mask=attention,position_ids=positions,
             visual_pos_masks=visual,deepstack_visual_embeds=streams,cache_position=torch.arange(width,device=emb.device),use_cache=False,return_dict=True)
        logits=model.lm_head(out.last_hidden_state[:,-len(target):]).float()
        ll=F.log_softmax(logits,dim=-1).gather(2,target[None,:,None].expand(count,-1,-1)).squeeze(-1)
        answers.append([dict(mean=float(v.mean()),token_loglikelihoods=v.cpu().tolist()) for v in ll])
        del out,logits,ll,emb,streams
    return [dict(G=g['mean'],S=s['mean'],C=g['mean']-s['mean'],accepted=[g],fixed_self=s) for g,s in zip(*answers)]


def tune(reader,prompt,memory,retained_sets,gold,own,*,atol=.05,max_batch=4):
    """Tune only on two longest masks, repeated for bounded shape probes.

    Choice minimizes measured seconds/mask on the allocated GPU. First regression
    or parity/OOM failure stops growth. No model-name batch-size table. Benchmark
    calls are recorded separately, never used as extra OMP observations.
    """
    pairs=sorted(retained_sets,key=len,reverse=True)[:2]
    trials=[]; best=1; best_time=float('inf'); peak_per_row=None
    reference=None
    for size in [1,2,4,8]:
        if size>max_batch:break
        free,total=torch.cuda.mem_get_info(); spare=max(512*1024**2,int(total*.05))
        if size>1 and peak_per_row is not None and peak_per_row*size>free-spare:
            trials.append(dict(batch=size,status='memory_estimate_skip'));break
        work=(pairs*((size+1)//2))[:size] if size>1 else pairs
        torch.cuda.synchronize(); base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        failed=None
        try:
            if size==1:
                scores=[score_batch(reader,prompt,memory,[r],gold,own)[0] for r in work]
            else:scores=score_batch(reader,prompt,memory,work,gold,own)
            torch.cuda.synchronize();elapsed=time.monotonic()-start
        except torch.cuda.OutOfMemoryError:
            failed='oom'
        if failed:
            gc.collect();torch.cuda.empty_cache();trials.append(dict(batch=size,status=failed));break
        peak=torch.cuda.max_memory_allocated();peak_per_row=max(peak_per_row or 0,(peak-base)/size)
        if reference is None:reference=scores
        error=max(abs(s[k]-reference[i%2][k]) for i,s in enumerate(scores) for k in ['G','S'])
        seconds=elapsed/len(work)
        trials.append(dict(batch=size,status='passed' if error<=atol else 'parity_rejected',seconds_per_mask=seconds,max_score_difference=error,peak_allocated_bytes=peak))
        if error>atol:break
        if size==1 or seconds<best_time*.95:best,best_time=size,seconds
        else:break
    return best,trials
