"""Eight outcome-independent Bernoulli masks using sealed cached reader vision."""
import argparse
import gc
import json
from pathlib import Path
import sys
import time
REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'src'),str(REPO)]
from experiments.training_pilot.run import read,sha,publish,fingerprint,stats,load_model,REVISION


def random_masks(qid,regions):
    import numpy as np
    seed=int(fingerprint(['expansion59-extra-random8-v1',qid])[:16],16)
    # No outcome-dependent rejection or fixed-cost repair, including duplicates.
    return np.random.default_rng(seed).integers(0,2,(8,regions)).astype(bool).tolist()


def main(a):
    import torch
    from docprune.stage2.data import load_example
    from docprune.stage2.contracts import VisualMemory
    from docprune.stage2.qwen import PackedPrompt
    from docprune.stage2.answerer import FrozenAnswerer
    root,out=Path(a.root),Path(a.output);parent=root/'expansion59-masks-v1'
    manifest=read(root/'expansion59-mask-prep-20260919/manifest.json')
    assert fingerprint(manifest['record'])==manifest['sha256']
    complete=read(parent/'teacher-complete.json');assert complete['status']=='passed'
    qids=[r['question']['qid'] for r in manifest['record']['records']]
    assert len(qids)==59 and set(qids)=={r['qid'] for r in complete['questions']}
    publish(out/'contract.json',dict(schema='extra-random8-cached-vision-v1',parent_contract_sha256=sha(parent/'contract.json'),manifest_sha256=manifest['sha256'],reader=REVISION,source_sha256=sha(Path(__file__)),policy='8 independent Bernoulli(.5); no outcome filtering; old random indices0:24 plus these8 form32random; old targeted24:32 separate'))
    assert Path(a.reader).name==REVISION
    torch.set_num_threads(2)
    processor,model=load_model(a.reader);reader=FrozenAnswerer(model)
    # A call to the visual encoder is an implementation error in this job.
    def forbid_vision(*args,**kwargs):raise RuntimeError('Cached-only job must not encode pixels')
    model.model.visual.forward=forbid_vision
    start=time.monotonic();torch.cuda.reset_peak_memory_stats();receipts=[]
    records={r['question']['qid']:r['question'] for r in manifest['record']['records']}
    for qid in qids:
        folder=out/qid
        if (folder/'complete.json').exists():receipts.append(read(folder/'complete.json'));continue
        example,bank=load_example(parent/qid/'example')
        assert example.identity.question_id==qid and not bank.baseline_correct
        b=read(root/'baseline-qwen3-8b-admitted-v2/answers'/f'{fingerprint(qid)}.json')
        ids=b['input_token_ids'];question=example.question_ids.tolist()
        first_image=ids.index(model.config.image_token_id)
        matches=[i for i in range(first_image-len(question)+1) if ids[i:i+len(question)]==question]
        assert len(matches)==1,'Cached question token positions cannot be reconstructed'
        pos=torch.arange(matches[0],matches[0]+len(question),device='cuda')
        prompt=PackedPrompt(torch.tensor([ids],device='cuda'),pos,example.vision.grid_thw.to('cuda'))
        memory=VisualMemory(example.vision.merged.to('cuda'),tuple(t.to('cuda') for t in example.vision.deepstack),example.vision.grid_thw.to('cuda'),example.vision.provenance)
        own=list(read(parent/qid/'anchor.json')['continuation_ids']);eos=model.generation_config.eos_token_id
        eos=[eos] if isinstance(eos,int) else eos
        while own and own[-1] in eos:own.pop()
        gold=records[qid]['gold'];gold=processor.tokenizer.encode(str(gold[0]) if len(gold)==1 else json.dumps(gold,ensure_ascii=False),add_special_tokens=False)
        masks=random_masks(qid,len(example.layout.region_ids))
        publish(folder/'proposals.json',dict(masks=masks,policy='independent Bernoulli',parent_example_sha256=sha(parent/qid/'example/manifest.json')))
        for i,mask in enumerate(masks):
            path=folder/'measurements'/f'{i:02d}.json'
            if path.exists():assert read(path)['mask']==mask;continue
            retained=example.layout.retained_tokens(torch.tensor(mask,dtype=torch.bool)).to('cuda')
            g=reader.likelihood(prompt,memory,retained,gold)['mean']
            s=reader.likelihood(prompt,memory,retained,own)['mean']
            publish(path,dict(index=i,mask=mask,g=g,s=s,retained_tokens=len(retained)))
        receipt=dict(qid=qid,new_random_masks=8,visual_encoding_calls=0,parent_random_indices=list(range(24)))
        publish(folder/'complete.json',receipt);receipts.append(receipt);print(json.dumps(receipt),flush=True)
        del example,bank,prompt,memory;gc.collect();torch.cuda.empty_cache()
    publish(out/'complete.json',dict(status='passed',questions=receipts,resources=stats(start,torch)))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    for k in ('root','output','reader'):p.add_argument('--'+k,required=True)
    main(p.parse_args())
