"""Bounded Qwen3 scoring audit. Frozen review required; no retrieval or training."""
import argparse
from dataclasses import fields, replace
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'src'),str(REPO/'experiments/m3doc600')]
from common import fingerprint, load_catalog, publish, read, sha, stats
from baseline import PROMPT, REVISION, admitted
from discovery import masks_for, fit_omp


def occurrence_assets(assets):
    """Pixel cache identity is not a unique occurrence identity in a prompt."""
    counts=Counter(v.page_id for v in assets)
    return [replace(v,page_id=f'{v.page_id}#occurrence:{i}') if counts[v.page_id]>1 else v for i,v in enumerate(assets)]


def main(a, loaded=None):
    a.smoke = a.smoke or a.smoke_then_run
    import torch
    import transformers
    from batching import score_batch, tune
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    from docprune.stage2.qwen import prepare_prompt, assemble_prefill
    from docprune.stage2.answerer import FrozenAnswerer
    from docprune.stage2.documents import PageAsset, region_layout
    assert transformers.__version__=='4.57.3'
    assert torch.cuda.is_available() and torch.cuda.get_device_capability()[0]>=8
    torch.set_num_threads(2)
    root=Path(a.root); out=root/'omp10-20260917-batch-v2'; review=read(a.review)
    assert review['status']=='frozen_before_mask_scoring' and len(review['selected'])==10
    selected=review['selected']; assert len({r['qid'] for r in selected})==10
    assert all(r['decision']=='confirmed_wrong' and r['evidence_pages'] and r['reason'] for r in selected)
    c,digest=load_catalog(root)
    assert digest=='bf220ddb7c0b67b58a992292108e9ec386b8694d3b33f542c71dd5d35f0b4d08'
    assert read(root/'combined/completion.json')['status']=='complete'
    qs={q['question_key']:q for q in c['questions']}; pages={p['key']:p for p in c['pages']}
    baseline=root/'baseline-qwen3-8b-admitted-v2'
    baseline_id='ad208c8f709e475507835f2f87054830c6d5f3dfaca962c5ffd805fbfd40f364'
    assert read(baseline/'contract.json')['sha256']==baseline_id
    pool={q['qid']:q for q in read(c['pool_path'])['questions']}
    # Bind every selected target to the original generated continuation before loading.
    originals={}
    for r in selected:
        path=baseline/'answers'/f"{fingerprint(r['qid'])}.json"
        b=read(path); assert sha(path)==r['baseline_file_sha256']
        assert b['contract_sha256']==baseline_id and b['answer']==r['baseline_answer']
        assert not b['hit_generation_limit']
        assert r['gold_items']==[v['answer'] for v in pool[r['qid']]['answers']]
        admitted_set={(p['doc_id'],p['page_index']) for p in pool[r['qid']]['admitted_pages']}
        assert all((p['doc_id'],p['page_index']) in admitted_set for p in r['evidence_pages'])
        originals[r['qid']]=b
    sources=[Path(__file__),Path(__file__).with_name('discovery.py'),Path(__file__).with_name('batching.py'),REPO/'experiments/m3doc600/common.py',REPO/'experiments/m3doc600/baseline.py']
    sources += [Path(m.__file__) for n,m in sys.modules.items() if n.startswith('docprune.stage2') and getattr(m,'__file__',None)]
    contract=dict(schema='omp10-fullprefix-batch-v1',review_sha256=sha(a.review),catalog_sha256=digest,
                  baseline_contract=baseline_id,revision=REVISION,precision='bfloat16',attention='sdpa',
                  source_hashes={str(p.relative_to(REPO)):sha(p) for p in sorted(set(sources))},
                  packages={k:importlib.metadata.version(k) for k in ['torch','transformers','numpy','Pillow']},
                  masks=22,mask_policy='independent Bernoulli(.5), no repair or resampling',
                  G='mean continuation log likelihood of complete gold answer; list items jointly serialized',
                  S='mean continuation log likelihood of original answer tokens, trailing EOS excluded',
                  position_policy='original multimodal positions; compact cache; original continuation origin',
                  partition='largest positive overlap; reading-order ties; page-local fallback tile8 cap128',
                  score_parity_atol=0.05,repeat_atol=1e-5,scorer='independent full-prefix, no KV branching',
                  batching='per-question longest-mask calibration 1/2/4; stop if slower, OOM or parity failure; largest first; no mask changes')
    parent_root=root/'omp10-20260917-batch-v1'
    parent=read(parent_root/'contract.json'); parent_id=parent.pop('sha256')
    assert fingerprint(parent)==parent_id
    for k,value in contract.items():
        if k=='source_hashes':
            assert {n:h for n,h in value.items() if n!='experiments/omp10/run.py'}=={n:h for n,h in parent[k].items() if n!='experiments/omp10/run.py'}
        else: assert value==parent[k],k
    contract.update(parent_contract=parent_id,layout_page_identity='qualify repeated pixel keys by frozen presentation occurrence; unique keys unchanged')
    cid=fingerprint(contract); publish(out/'contract.json',dict(contract,sha256=cid))
    parent_smoke=read(parent_root/'smoke.json')
    assert parent_smoke['status']=='passed' and parent_smoke['contract_sha256']==parent_id
    for qid in review['smoke_qids']:
        keys=[p['key'] for p in admitted(qs[qid])]
        assert len(keys)==len(set(keys)), 'Changed layout requires a fresh smoke'
    publish(out/'smoke.json',dict(parent_smoke,contract_sha256=cid,reused_from=str(parent_root/'smoke.json'),parent_contract=parent_id))
    assert Path(a.snapshot).name==REVISION
    smoke_ids=review['smoke_qids']; assert len(set(smoke_ids))==2 and set(smoke_ids)<=set(originals)
    if a.smoke:
        todo=[r for r in selected if r['qid'] in smoke_ids]
    else:
        assert 0<=a.shard<a.shards
        smoke=read(out/'smoke.json'); assert smoke['status']=='passed' and smoke['contract_sha256']==cid
        todo=selected[a.shard::a.shards]
    receipt_path=out/('smoke.json' if a.smoke else f'runs/shard-{a.shard}-of-{a.shards}.json')
    if receipt_path.exists():
        old=read(receipt_path); assert old['contract_sha256']==cid
        print(json.dumps(old))
        if a.smoke_then_run:
            a.smoke=False; a.smoke_then_run=False
            return main(a,loaded)
        return
    start=time.monotonic()
    if loaded is None:
        processor=AutoProcessor.from_pretrained(a.snapshot,local_files_only=True,min_pixels=256*32*32,max_pixels=2560*32*32)
        model=Qwen3VLForConditionalGeneration.from_pretrained(a.snapshot,local_files_only=True,dtype=torch.bfloat16,device_map='cuda:0',attn_implementation='sdpa').eval().requires_grad_(False)
    else: processor,model=loaded
    reader=FrozenAnswerer(model); results=[]
    eos=model.generation_config.eos_token_id; eos=[eos] if isinstance(eos,int) else eos
    for r in todo:
        qid=r['qid']; q=qs[qid]; b=originals[qid]; folder=out/'questions'/qid
        if a.smoke and (folder/'smoke.json').exists():
            prior=read(folder/'smoke.json'); assert prior['contract_sha256']==cid
            results.append(prior); continue
        previous=parent_root/'questions'/qid
        if not a.smoke and (previous/'omp.json').exists():
            bank=read(previous/'bank.json')
            assert bank['contract_sha256']==parent_id and len(bank['page_keys'])==len(set(bank['page_keys']))
            expected=masks_for(qid,bank['region_ids'])
            assert expected.tolist()==bank['masks'] and bank['ordered_pages']==b['ordered_pages']
            files=[previous/'bank.json',previous/'omp.json']
            assert read(files[1])['contract_sha256']==parent_id
            for i,row in enumerate(expected):
                path=previous/'measurements'/f'{i:02d}.json';v=read(path)
                assert v['contract_sha256']==parent_id and v['mask']==row.tolist() and np.isfinite([v['G'],v['S']]).all()
                files.append(path)
            item=dict(qid=qid,measurements=22,reused_from=str(previous),parent_contract=parent_id,files_sha256={str(p.relative_to(previous)):sha(p) for p in files})
            publish(folder/'reused.json',item);results.append(item);print(json.dumps(dict(qid=qid,reused_measurements=22)),flush=True)
            continue
        keep=admitted(q)
        assert b['ordered_pages']==[p['page_id'] for p in keep]
        assets=[]; images=[]; regions=[]
        for p in keep:
            page=pages[p['key']]; lp=root/'mineru/pages'/f"{p['key']}.json"
            asset=PageAsset(p['key'],str(root/page['image']),page['image_sha256'],str(lp),sha(lp),'frozen-catalog:'+digest)
            im,boxes=asset.read(); assets.append(asset); images.append(im); regions.append(boxes)
        prompt=prepare_prompt(processor,PROMPT+q['question'],images)
        for im in images:im.close()
        assert prompt.input_ids[0].tolist()==b['input_token_ids']
        assert prompt.image_grid_thw.tolist()==b['image_grid_thw']
        n=int(prompt.image_grid_thw.prod(1).sum()//4)
        layout,boxes,audit=region_layout(prompt.image_grid_thw,2,occurrence_assets(assets),regions,SimpleNamespace(fallback_tile_size=8,max_fallback_tokens=128,budget=max(1,n//2)))
        x=masks_for(qid,layout.region_ids)
        gold=r['gold_items']; gold_text=str(gold[0]) if len(gold)==1 else json.dumps(gold,ensure_ascii=False)
        gold_ids=processor.tokenizer.encode(gold_text,add_special_tokens=False)
        own=list(b['answer_token_ids'])
        while own and own[-1] in eos:own.pop()
        assert own and gold_ids
        bank=dict(contract_sha256=cid,qid=qid,region_ids=layout.region_ids,owner=layout.owner.tolist(),
                  page=layout.page.tolist(),footprints=boxes.tolist(),region_costs=layout.costs.tolist(),
                  ordered_pages=b['ordered_pages'],page_keys=[p['key'] for p in keep],audit=audit,
                  masks=x.tolist(),retained_token_counts=(x@layout.costs.numpy()).tolist(),
                  gold_text=gold_text,gold_ids=gold_ids,self_ids=own,self_text=b['answer'],
                  layout_hashes=[v.layout_sha256 for v in assets])
        publish(folder/'bank.json',bank) # Seal design and targets before any scoring.
        prompt=type(prompt)(**{f.name:getattr(prompt,f.name).to('cuda') for f in fields(prompt)})
        torch.cuda.reset_peak_memory_stats(); begin=time.monotonic()
        memory=reader.vision(prompt,'frozen-catalog:'+digest)
        allkeep=torch.arange(n,device='cuda'); targets=[gold_ids,own]
        def score(retained):
            torch.cuda.synchronize(); t=time.monotonic()
            v=reader.teacher_scores(prompt,memory,retained,[gold_ids],own,reuse_prefill=False)
            torch.cuda.synchronize()
            return dict(v,seconds=time.monotonic()-t)
        anchor_path=folder/'anchor.json'
        anchor=read(anchor_path) if anchor_path.exists() else score(allkeep)
        if anchor_path.exists(): assert anchor['contract_sha256']==cid
        retained_sets=[torch.where(torch.as_tensor(row,dtype=torch.bool)[layout.owner])[0].to('cuda') for row in x]
        if a.smoke:
            with torch.inference_mode():
                native=model(input_ids=prompt.input_ids,attention_mask=torch.ones_like(prompt.input_ids),pixel_values=prompt.pixel_values,image_grid_thw=prompt.image_grid_thw,use_cache=False,logits_to_keep=1).logits[:, -1].float()
                explicit=reader.prefill(prompt,memory,allkeep)[0].float()
            native_max=float((native-explicit).abs().max())
            assert native.argmax(-1).item()==explicit.argmax(-1).item()
            torch.testing.assert_close(native,explicit,atol=0.02,rtol=0.005)
            checks=[]
            for index in [0,1]:
                retained=torch.where(torch.as_tensor(x[index],dtype=torch.bool)[layout.owner])[0].to('cuda')
                packed,indices,_=assemble_prefill(model,prompt,memory,retained)
                assert len(indices)==prompt.input_ids.numel()-n+len(retained)
                assert int(packed['visual_pos_masks'].sum())==len(retained)
                for stream,original in zip(packed['deepstack_visual_embeds'],memory.deepstack):
                    assert torch.equal(stream,original[retained].to(stream))
                v=score(retained); repeat=score(retained)
                reference=reader.teacher_scores(prompt,memory,retained,[gold_ids],own,reuse_prefill=False)
                parity=max(abs(v[k]-reference[k]) for k in ['G','S'])
                repeat_error=max(abs(v[k]-repeat[k]) for k in ['G','S'])
                checks.append(dict(mask=index,retained=len(retained),scores=v,reference=reference,parity_max=parity,repeat_max=repeat_error))
                assert parity<=contract['score_parity_atol'] and repeat_error<=contract['repeat_atol'], checks[-1]
            batch_size,trials=tune(reader,prompt,memory,retained_sets,gold_ids,own,atol=contract['score_parity_atol'])
            item=dict(qid=qid,anchor=anchor,native_max_logit_difference=native_max,checks=checks,batch_size=batch_size,batch_trials=trials,stats=stats(begin,torch))
            publish(folder/'smoke.json',dict(item,contract_sha256=cid)); results.append(item)
        else:
            ap=folder/'anchor.json'
            if not ap.exists():publish(ap,dict(anchor,contract_sha256=cid))
            if loaded is not None and (folder/'smoke.json').exists():
                calibration=read(folder/'smoke.json');batch_size=calibration['batch_size'];trials=calibration['batch_trials']
            else:batch_size,trials=tune(reader,prompt,memory,retained_sets,gold_ids,own,atol=contract['score_parity_atol'])
            print(json.dumps(dict(qid=qid,batch_size=batch_size,batch_trials=trials)),flush=True)
            pending=[i for i in range(22) if not (folder/'measurements'/f'{i:02d}.json').exists()]
            pending.sort(key=lambda i:len(retained_sets[i]),reverse=True)
            batches=[]
            while pending:
                count=2**int(np.log2(min(batch_size,len(pending))))
                indices=pending[:count]
                values=None
                torch.cuda.synchronize();t=time.monotonic()
                try:values=score_batch(reader,prompt,memory,[retained_sets[i] for i in indices],gold_ids,own)
                except torch.cuda.OutOfMemoryError:
                    if count==1:raise
                if values is None:
                    import gc
                    gc.collect();torch.cuda.empty_cache();batch_size=count//2
                    batches.append(dict(status='oom_retry',batch=count,next_batch=batch_size))
                    continue
                torch.cuda.synchronize();seconds=time.monotonic()-t
                for index,v in zip(indices,values):
                    publish(folder/'measurements'/f'{index:02d}.json',dict(v,contract_sha256=cid,index=index,mask=x[index].tolist(),retained_tokens=len(retained_sets[index]),batch_size=count,batch_seconds=seconds))
                batches.append(dict(status='complete',indices=indices,seconds=seconds))
                pending=pending[count:]
            ys=[]
            for index,row in enumerate(x):
                v=read(folder/'measurements'/f'{index:02d}.json')
                assert v['contract_sha256']==cid and v['mask']==row.tolist()
                ys.append([v['G'],v['S']])
            fit=fit_omp(x,ys); publish(folder/'omp.json',dict(fit,contract_sha256=cid))
            results.append(dict(qid=qid,stats=stats(begin,torch),measurements=22,batch_size=batch_size,batch_trials=trials,batches=batches))
        print(json.dumps(results[-1]),flush=True)
        del memory,prompt
    receipt=dict(status='passed' if a.smoke else 'complete',contract_sha256=cid,results=results,stats=stats(start,torch))
    publish(out/('smoke.json' if a.smoke else f'runs/shard-{a.shard}-of-{a.shards}.json'),receipt)
    if a.smoke_then_run:
        a.smoke=False; a.smoke_then_run=False
        return main(a,(processor,model))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',required=True);p.add_argument('--snapshot',required=True);p.add_argument('--review',required=True)
    p.add_argument('--smoke',action='store_true');p.add_argument('--smoke-then-run',action='store_true');p.add_argument('--shard',type=int,default=0);p.add_argument('--shards',type=int,default=1)
    main(p.parse_args())
