"""Length-aware mask batching, bounded tuning and explicit numerical fallback."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]/"src"))
import time
import torch
from torch.nn import functional as F
from docprune.stage2.qwen import assemble_prefill


@torch.no_grad()
def padded_score(reader, prompt, memory, retained_sets, gold, own, pad_to=0):
    """Independent full-prefix scoring; left padding never changes real positions.

    All rows are masks of one question. DeepStack features follow boolean indexing
    order (batch row, then original retained-token order). No KV branch reuse.
    """
    model=reader.model
    prefixes=[assemble_prefill(model,prompt,memory,r) for r in retained_sets]
    answers=[]
    for target in [gold,own]:
        target=torch.tensor(target,dtype=torch.long,device=prompt.input_ids.device)
        tail=target[:-1]; width=max(pad_to,max(p[0]['inputs_embeds'].shape[1] for p in prefixes))+len(tail)
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



def main(a):
    import os,sys,json,statistics,subprocess
    from pathlib import Path
    from dataclasses import fields
    repo=Path(__file__).resolve().parents[2]
    sys.path.insert(0,str(repo/'experiments/m3doc600'))
    from common import read,sha,load_catalog,publish,fingerprint
    from baseline import PROMPT,admitted
    from transformers import AutoProcessor,Qwen3VLForConditionalGeneration
    from docprune.stage2.qwen import prepare_prompt
    from docprune.stage2.answerer import FrozenAnswerer
    from PIL import Image
    assert os.environ.get('SLURM_JOB_ID')
    torch.set_num_threads(2)
    root=Path(a.root); out=Path(a.out);out.mkdir(parents=True,exist_ok=False)
    qid='8c382451f4725b39b3872df86a082def'
    bankpath=root/'omp10-20260917-batch-v2/questions'/qid/'bank.json'
    bank=read(bankpath);c,digest=load_catalog(root)
    assert digest=='bf220ddb7c0b67b58a992292108e9ec386b8694d3b33f542c71dd5d35f0b4d08'
    q=next(q for q in c['questions'] if q['question_key']==qid)
    pages={p['key']:p for p in c['pages']};keep=admitted(q)
    assert [p['key'] for p in keep]==bank['page_keys']
    baseline=read(root/'baseline-qwen3-8b-admitted-v2/answers'/f'{fingerprint(qid)}.json')
    processor=AutoProcessor.from_pretrained(a.snapshot,local_files_only=True,min_pixels=256*32*32,max_pixels=2560*32*32)
    model=Qwen3VLForConditionalGeneration.from_pretrained(a.snapshot,local_files_only=True,dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa').eval().requires_grad_(False)
    images=[]
    for p in keep:
        page=pages[p['key']];path=root/page['image'];assert sha(path)==page['image_sha256']
        with Image.open(path) as im:images.append(im.convert('RGB'))
    prompt=prepare_prompt(processor,PROMPT+q['question'],images)
    for im in images:im.close()
    assert prompt.input_ids[0].tolist()==baseline['input_token_ids']
    assert prompt.image_grid_thw.tolist()==baseline['image_grid_thw']
    prompt=type(prompt)(**{f.name:getattr(prompt,f.name).to('cuda') for f in fields(prompt)})
    reader=FrozenAnswerer(model);memory=reader.vision(prompt,'frozen-catalog:'+digest)
    owner=torch.tensor(bank['owner'],device='cuda')
    masks=[torch.where(torch.tensor(row,device='cuda',dtype=torch.bool)[owner])[0] for row in bank['masks']]
    ids=sorted(range(len(masks)),key=lambda i:len(masks[i]),reverse=True)[:2]
    long,short=[masks[i] for i in ids];assert len(long)>len(short)
    gold,own=bank['gold_ids'],bank['self_ids']
    width=assemble_prefill(model,prompt,memory,long)[0]['inputs_embeds'].shape[1]
    def single(r):return reader.teacher_scores(prompt,memory,r,[gold],own,reuse_prefill=False)
    reference=[single(long),single(short)]
    cases={
      'sequential':(lambda:[single(long),single(short)],reference),
      'equal_batch':(lambda:padded_score(reader,prompt,memory,[long,long],gold,own),[reference[0],reference[0]]),
      'unequal_batch':(lambda:padded_score(reader,prompt,memory,[long,short],gold,own),reference),
      'padded_single':(lambda:padded_score(reader,prompt,memory,[short],gold,own,pad_to=width),[reference[1]])}
    publish(out/'design.json',dict(qid=qid,bank_sha256=sha(bankpath),mask_indices=ids,retained_lengths=[len(long),len(short)],prefix_width=width,commit=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),gpu=torch.cuda.get_device_name(),torch=torch.__version__,purpose='diagnostic only; never teacher bank'))
    records={name:[] for name in cases}
    for fn,_ in cases.values():fn();torch.cuda.synchronize()
    for repeat in range(3):
        names=list(cases)
        if repeat%2:names.reverse()
        for name in names:
            fn,ref=cases[name];torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
            begin=time.monotonic();values=fn();torch.cuda.synchronize();elapsed=time.monotonic()-begin
            row=dict(repeat=repeat,seconds=elapsed,seconds_per_mask=elapsed/len(values),peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved(),scores=values,max_error=max(abs(x[k]-y[k]) for x,y in zip(values,ref) for k in ['G','S']))
            records[name].append(row);publish(out/f'{name}-{repeat}.json',row)
    for name,(fn,_) in cases.items():
        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True) as prof:
            with torch.profiler.record_function(name):fn();torch.cuda.synchronize()
        prof.export_chrome_trace(str(out/f'{name}-trace.json'))
        (out/f'{name}-operators.txt').write_text(prof.key_averages().table(sort_by='self_cuda_time_total',row_limit=35))
    result={name:dict(median_seconds_per_mask=statistics.median(r['seconds_per_mask'] for r in rows),max_error=max(r['max_error'] for r in rows),peak_allocated=max(r['peak_allocated'] for r in rows),peak_reserved=max(r['peak_reserved'] for r in rows)) for name,rows in records.items()}
    publish(out/'complete.json',dict(status='complete',results=result));print(json.dumps(result),flush=True)

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--snapshot',required=True);p.add_argument('--out',required=True);main(p.parse_args())
