"""Reader evaluation of shortlisted dev selections; no training or fresh retrieval."""
import argparse
import json
import math
from pathlib import Path
import sys
import time
sys.path[:0]=[str(Path(__file__).resolve().parents[2]/'src'),str(Path(__file__).resolve().parents[2])]
import torch
from experiments.training_pilot.run import read,sha,pages,load_catalog,load_model,to_prompt,PROMPT,REVISION,stats
from docprune.stage2.storage import publish,read_record,run_lock
from docprune.stage2.contracts import fingerprint
from docprune.stage2.data import load_example
from docprune.stage2.qwen import prepare_prompt
from docprune.stage2.answerer import FrozenAnswerer
from docprune.stage2.api_version import load_labels
from docprune.stage2.policy import allocate


def candidates(qid,ex,branches):
    result={f'{phase}/{r}':v['mask'] for phase,values in branches.items() for r,v in values[qid].items()}
    # Schema is [MaxSim, 128 query-vector dimensions], not the reverse.
    relevance=ex.retrieval.values[:,:,0].sum(1)
    for r in (.75,.5):
        z,_=allocate(relevance,ex.layout.costs,math.floor(ex.budget*r))
        result[f'colqwen/{r}']=z.tolist()
    seed=int(fingerprint(['fixed-dev-random-v1',qid])[:15],16)
    generator=torch.Generator().manual_seed(seed)
    for i in range(3):result[f'random/{i}']=(torch.rand(len(ex.layout.region_ids),generator=generator)<.5).tolist()
    return result


def run(a):
    root,out,training=Path(a.root),Path(a.output),Path(a.training)
    audit=read_record(a.audit);assert audit['status']=='passed'
    assert read_record(training/'training-complete.json')['status']=='passed'
    training_contract=read_record(training/'contract.json')
    assert training_contract['audit_sha256']==sha(a.audit)
    branches={'untrained':read_record(training/'untrained.json')['masks']}
    for phase in ('frozen','lora'):branches[phase]=read_record(training/(phase+'-complete.json'))['masks']
    dev=[r for r in audit['rows'] if r['split']=='dev']
    assert len(dev)==24 and all(set(b)=={r['qid'] for r in dev} for b in branches.values())
    contract=dict(schema='pilot64-reader-dev-v1',training_contract_sha256=sha(training/'contract.json'),audit_sha256=sha(a.audit),
        selection_sha256={p.name:sha(p) for p in [training/'untrained.json',training/'frozen-complete.json',training/'lora-complete.json']},
        reader=REVISION,decode=a.decode,retention=[.75,.5],random='three independent Bernoulli(0.5), not repaired',source_sha256=sha(__file__))
    publish(out/'contract.json',contract)
    assert Path(a.reader).name==REVISION
    catalog,_=load_catalog(root);labels=load_labels(root/'training-pilot-v1/labels-deepseek-v4.1-flash.json')
    processor,model=load_model(a.reader);reader=FrozenAnswerer(model)
    eos=model.generation_config.eos_token_id;eos=[eos] if isinstance(eos,int) else eos
    started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    for row in dev:
        qid=row['qid'];folder=Path(row['directory']);label=labels[qid]
        target=out/(qid+'.json')
        if target.exists():read_record(target);continue
        assert sha(folder/'example/manifest.json')==row['manifest_sha256']
        ex,bank=load_example(folder/'example');q,assets,images,regions=pages(root,catalog,qid)
        try:prompt=to_prompt(prepare_prompt(processor,PROMPT+q['question'],images),'cuda')
        finally:
            for im in images:im.close()
        memory=reader.vision(prompt,fingerprint(contract))
        own=list(read(folder/'anchor.json')['continuation_ids'])
        while own and own[-1] in eos:own.pop()
        gold=None if bank.baseline_correct else processor.tokenizer.encode(str(label['gold'][0]) if len(label['gold'])==1 else json.dumps(label['gold'],ensure_ascii=False),add_special_tokens=False)
        def measure(mask):
            tokens=ex.layout.retained_tokens(torch.tensor(mask,dtype=torch.bool)).cuda();start=time.monotonic()
            result=dict(retained_tokens=len(tokens),g=None if gold is None else reader.likelihood(prompt,memory,tokens,gold)['mean'],s=reader.likelihood(prompt,memory,tokens,own)['mean'])
            if a.decode:
                result['generation']=reader.generate(prompt,memory,tokens,max_new_tokens=256,eos_ids=eos,repetition_penalty=1.)
                result['answer']=processor.tokenizer.decode(result['generation']['token_ids'],skip_special_tokens=True).strip()
                result['grading_status']='pending_frozen_evaluator; original-answer label is not reused'
            torch.cuda.synchronize();result['reader_seconds']=time.monotonic()-start
            return result
        allkeep=[True]*len(ex.layout.region_ids);baseline=measure(allkeep)
        assert abs(baseline['s']-bank.reference.s)<1e-5
        if gold is not None:assert abs(baseline['g']-bank.reference.g)<1e-5
        if a.decode and baseline['answer']!=label['model_answer']:raise ValueError('All-keep decoder baseline mismatch')
        memo={tuple(allkeep):baseline};results={}
        for name,mask in candidates(qid,ex,branches).items():
            key=tuple(mask)
            if key not in memo:memo[key]=measure(mask)
            results[name]=dict(mask=mask,**memo[key])
        publish(target,dict(qid=qid,baseline_correct=bank.baseline_correct,baseline=baseline,selected=results))
        print(dict(qid=qid,distinct_masks=len(memo)),flush=True)
        del ex,bank,prompt,memory
    publish(out/'complete.json',dict(status='passed',questions=len(dev),resources=stats(started,torch),decoded=a.decode,
        accuracy_status='Requires grading new answers; likelihood changes alone are not correctness'))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','output','training','audit','reader'):p.add_argument('--'+n,required=True)
    p.add_argument('--decode',action='store_true');a=p.parse_args()
    with run_lock(a.output):run(a)
