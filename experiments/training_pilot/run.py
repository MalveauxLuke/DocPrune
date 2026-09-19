"""Approved two-correct-case pilot smoke; sealed inputs, cached retrieval, native models."""
import argparse
from dataclasses import asdict, replace
import gc
import json
from pathlib import Path
import sys
import time

REPO=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(REPO/'src'),str(REPO/'experiments/m3doc600'),str(REPO)]
from common import fingerprint, load_catalog, publish, read, sha, stats
from baseline import PROMPT, REVISION, admitted
from experiments.selector_smoke.run import pages, load_model, to_prompt, layout_to, SELECTOR_REV
from docprune.stage2.api_version import load_labels
from docprune.stage2.pilot import VERSION, RETRIEVAL_SCHEMA


def context(a):
    root=Path(a.root);out=Path(a.output);packet=root/'training-pilot-v1'
    smoke=read(packet/'smoke.json');labels_path=packet/smoke['labels_file']
    assert sha(labels_path)==smoke['labels_sha256']
    labels=load_labels(labels_path);cohort=root/'training463-evidence-filtered-v1'
    assert sha(cohort/'selection.json')==smoke['cohort_selection_sha256']
    selection=read(cohort/'selection.json');assert sha(cohort/'pool.json')==selection['pool_sha256']
    pool={q['qid']:q for q in read(cohort/'pool.json')['questions']}
    assert set(smoke['qids'])<=set(selection['qids']) and len(smoke['qids'])==2
    assert all(labels[q]['verdict']=='correct' for q in smoke['qids'])
    catalog,digest=load_catalog(root)
    code=[Path(__file__),REPO/'src/docprune/stage2/pilot.py',REPO/'src/docprune/stage2/supervision.py']
    contract=dict(schema='correct-preservation-pilot-smoke-v1',smoke=smoke,catalog=digest,
                  acquisition=VERSION,retrieval_schema=RETRIEVAL_SCHEMA,retrieval_dim=129,
                  reader=REVISION,selector=SELECTOR_REV,capacity=1.,pair_weighting='hamming_families_v1',
                  sources={str(p.relative_to(REPO)):sha(p) for p in code})
    publish(out/'contract.json',contract)
    return root,out,catalog,pool,labels,smoke,contract


def retrieval(root,q,layout,boxes,catalog):
    import torch
    from safetensors.torch import load_file
    from docprune.stage2.documents import map_retrieval_profiles
    base=root/'colqwen';contract=read(base/'contract.json');records=[]
    def get(kind,key):
        meta=read(base/kind/(key+'.json'));path=base/kind/(key+'.safetensors')
        assert meta['contract_sha256']==contract['sha256']
        assert sha(path)==meta['tensor_sha256']
        records.append(dict(kind=kind,key=key,sha256=meta['tensor_sha256']))
        return meta,load_file(path)
    meta,query=get('queries',fingerprint(q['question_key']))
    assert meta['question_key']==q['question_key'] and meta['question_sha256']==fingerprint(q['question'])
    vectors=query['embeddings'];assert vectors.shape[1]==128
    patch_pages=[];patch_boxes=[];scores=[]
    for i,p in enumerate(admitted(q)):
        meta,page=get('pages',p['key'])
        assert meta['page_key']==p['key']
        assert meta['image_sha256']==next(z['image_sha256'] for z in catalog['pages'] if z['key']==p['key'])
        pos=page['image_positions'];assert len(pos)==len(page['patch_boxes'])
        scores.append(vectors@page['embeddings'][pos].T)
        patch_boxes.append(page['patch_boxes']);patch_pages.append(torch.full((len(pos),),i,dtype=torch.long))
    features=map_retrieval_profiles(layout,boxes,torch.cat(patch_pages),torch.cat(patch_boxes),torch.cat(scores,1),vectors,schema=RETRIEVAL_SCHEMA)
    return features,dict(contract=contract['sha256'],records=records,mapping=fingerprint([layout.owner.tolist(),boxes.tolist(),records]),shape=list(features.values.shape))


def teacher(a, context_fn=context):
    import torch
    from transformers import GenerationConfig
    from types import SimpleNamespace
    from docprune.stage2.answerer import FrozenAnswerer
    from docprune.stage2.contracts import InstanceIdentity,SelectorInputs
    from docprune.stage2.data import save_example
    from docprune.stage2.documents import region_layout
    from docprune.stage2.qwen import prepare_prompt
    from docprune.stage2.supervision import Outcome,TeacherBank
    from docprune.stage2.pilot import next_probe
    root,out,catalog,pool,labels,smoke,contract=context_fn(a)
    assert Path(a.reader).name==REVISION
    processor,model=load_model(a.reader);reader=FrozenAnswerer(model)
    started=time.monotonic();torch.cuda.reset_peak_memory_stats();receipts=[]
    for qid in smoke['qids']:
        folder=out/qid;label=labels[qid]
        if (folder/'question-complete.json').exists():
            receipt=read(folder/'question-complete.json')
            from docprune.stage2.data import load_example
            loaded,loaded_bank=load_example(folder/'example')
            assert loaded.identity.question_id==qid and loaded_bank.adjudication_identity==label['manifest_identity']
            receipts.append(receipt);del loaded,loaded_bank
            continue
        correct=label['verdict']=='correct'
        if label['verdict'] not in ('correct','incorrect'):raise ValueError('Unresolved label')
        mask_count=smoke.get('mask_counts',{}).get(qid,smoke['mask_count'])
        dev=qid in smoke.get('dev_qids',[])
        flags=[]
        def check(ok,name,details=None):
            if ok:return
            if not smoke.get('diagnostic_flags_only',False):raise ValueError(name)
            flag=dict(check=name,details=details)
            flags.append(flag)
            publish(folder/'flags'/(name+'.json'),flag)
            print(json.dumps(dict(diagnostic_flag=name,question_id=qid,details=details)),flush=True)
        q,assets,images,regions=pages(root,catalog,qid)
        check(label['question']==q['question'],'question_text_mismatch')
        source_gold=[x['answer'] for x in pool[qid]['answers']]
        check(label['gold']==source_gold,'gold_representation_mismatch',
              dict(frozen=label['gold'],source=source_gold,
                   equivalent_as_text=label['gold']==[str(x) for x in source_gold]))
        b=read(root/'baseline-qwen3-8b-admitted-v2/answers'/f'{fingerprint(qid)}.json')
        check(b['answer']==label['model_answer'],'cached_answer_text_mismatch')
        assert sha(root/'baseline-qwen3-8b-admitted-v2/answers'/f'{fingerprint(qid)}.json')==label['source_answer_file_sha256']
        prompt=prepare_prompt(processor,PROMPT+q['question'],images)
        for im in images:im.close()
        assert prompt.input_ids[0].tolist()==b['input_token_ids'] and prompt.image_grid_thw.tolist()==b['image_grid_thw']
        n=int(prompt.image_grid_thw.prod(1).sum()//4)
        layout,boxes,audit=region_layout(prompt.image_grid_thw,2,assets,regions,SimpleNamespace(fallback_tile_size=8,max_fallback_tokens=128,budget=n))
        features,mapping=retrieval(root,q,layout,boxes,catalog)
        prompt=to_prompt(prompt,'cuda')
        anchor=folder/'anchor.json'
        if not anchor.exists() and smoke.get('use_saved_answer',False):
            publish(anchor,dict(answer=b['answer'],continuation_ids=b['answer_token_ids'],
                                label_source=label['label_source'],adjudication=label['manifest_identity'],
                                source='frozen_baseline_no_regeneration'))
        if not anchor.exists():
            generation=GenerationConfig(do_sample=False,max_new_tokens=256,repetition_penalty=1.,use_cache=True,eos_token_id=model.generation_config.eos_token_id,pad_token_id=processor.tokenizer.pad_token_id,bos_token_id=model.generation_config.bos_token_id)
            with torch.inference_mode():
                generated=model.generate(input_ids=prompt.input_ids,attention_mask=torch.ones_like(prompt.input_ids),pixel_values=prompt.pixel_values,image_grid_thw=prompt.image_grid_thw,generation_config=generation,use_model_defaults=False,do_sample=False)
            ids=generated[0,prompt.input_ids.shape[1]:].tolist()
            answer=processor.tokenizer.decode(ids,skip_special_tokens=True).strip()
            check(answer==label['model_answer'],'baseline_reproduction_mismatch',dict(generated=answer))
            check(ids==b['answer_token_ids'],'continuation_token_mismatch')
            publish(anchor,dict(answer=answer,continuation_ids=ids,label_source=label['label_source'],adjudication=label['manifest_identity']))
            del generated
        own=list(read(anchor)['continuation_ids']);eos=model.generation_config.eos_token_id
        eos=[eos] if isinstance(eos,int) else eos
        while own and own[-1] in eos:own.pop()
        assert own
        memory=reader.vision(prompt,fingerprint(contract))
        targets=label['gold']
        gold=None if correct else processor.tokenizer.encode(str(targets[0]) if len(targets)==1 else json.dumps(targets,ensure_ascii=False),add_special_tokens=False)
        if not correct and not gold:raise ValueError('Empty gold continuation')
        def score(mask):
            tokens=layout.retained_tokens(mask).cuda()
            return dict(g=None if correct else reader.likelihood(prompt,memory,tokens,gold)['mean'],s=reader.likelihood(prompt,memory,tokens,own)['mean'])
        refpath=folder/'reference.json'
        if not refpath.exists():publish(refpath,score(torch.ones(len(layout.region_ids),dtype=torch.bool)))
        reference=read(refpath)
        empty=folder/'empty-interface.json'
        if smoke.get('test_empty_interface',True) and not empty.exists():publish(empty,score(torch.zeros(len(layout.region_ids),dtype=torch.bool)))
        rows=[]
        while True:
            proposal=next_probe(rows,layout.costs.tolist(),qid+(':independent-dev-v1' if dev else ''),correct,reference,count=mask_count,random_only=dev)
            if proposal is None:break
            i=len(rows);publish(folder/'proposals'/f'{i:02d}.json',proposal)
            path=folder/'measurements'/f'{i:02d}.json'
            if not path.exists():
                mask=torch.tensor(proposal['mask'],dtype=torch.bool)
                vals=score(mask)
                publish(path,dict(mask=proposal['mask'],retained_tokens=int((mask*layout.costs).sum()),**vals))
            row=read(path);assert row['mask']==proposal['mask'];rows.append(row)
        assert len(rows)==min(mask_count,2**len(layout.region_ids))
        matrix=torch.tensor([r['mask'] for r in rows],dtype=torch.bool)
        repeat=score(matrix[0])
        repeat_deltas={key:repeat[key]-rows[0][key] for key in ('s',) if correct}
        if not correct:repeat_deltas={key:repeat[key]-rows[0][key] for key in ('g','s')}
        check(all(abs(v)<1e-5 for v in repeat_deltas.values()),'repeat_score_difference',repeat_deltas)
        publish(folder/'repeat-diagnostic.json',dict(deltas=repeat_deltas,within_1e_5=all(abs(v)<1e-5 for v in repeat_deltas.values())))
        identity=InstanceIdentity('m3docvqa',fingerprint(b['ordered_pages']),qid,fingerprint(q['question']),tuple(b['ordered_pages']),fingerprint(b['image_grid_thw']),fingerprint([layout.region_ids,layout.owner.tolist()]),REVISION,mapping['mapping'],fingerprint([PROMPT,own,gold,'S-only-full-prefix' if correct else 'GS-full-prefix',label['manifest_identity']]))
        example=SelectorInputs(identity,layout,memory,prompt.input_ids[0,prompt.question_positions].cpu(),n,features)
        bank=TeacherBank(identity.key,matrix,tuple(Outcome(r['g'],r['s']) for r in rows),Outcome(**reference),correct,not dev,identity.document_family,'s',label['manifest_identity'],'variable_pilot_v1','hamming_families_v1')
        bank.validate(example)
        publish(folder/'design.json',dict(boxes=boxes.tolist(),audit=audit,mapping=mapping,region_ids=layout.region_ids,costs=layout.costs.tolist()))
        if not (folder/'example').exists():save_example(folder/'example',example,bank)
        pairs=bank.pairs('gold_aware',.1,.05)
        if not pairs and smoke.get('require_pairs',True):raise ValueError('No strict pairs; retain case but smoke cannot establish training path')
        distances=[int((matrix[i]!=matrix[j]).sum()) for i,j in pairs]
        receipts.append(dict(qid=qid,label_source=label['label_source'],masks=len(rows),pairs=len(pairs),pair_families=[sum(d==1 for d in distances),sum(d==2 for d in distances),sum(d>2 for d in distances)],retained_min=min(r['retained_tokens'] for r in rows),retained_max=max(r['retained_tokens'] for r in rows),full_tokens=n,retrieval=mapping,repeat_delta=repeat['s']-rows[0]['s']))
        publish(folder/'question-complete.json',receipts[-1])
        print(json.dumps(receipts[-1]),flush=True)
        del memory,example,bank,prompt;gc.collect();torch.cuda.empty_cache()
    publish(out/'teacher-complete.json',dict(status='passed',questions=receipts,resources=stats(started,torch)))


def train(a):
    import torch
    from docprune.stage2.contracts import RetrievalFeatures
    from docprune.stage2.data import load_example
    from docprune.stage2.documents import native_action_layout
    from docprune.stage2.experiment import ExperimentConfig,Stage1Contract
    from docprune.stage2.policy import allocate,score_candidate_masks
    from docprune.stage2.qwen import prepare_prompt
    from docprune.stage2.training import build_selector
    from docprune.stage2.pilot import CachedPilotSelector,phase_optimizer,train_epoch,branch_scheduler
    root,out,catalog,pool,labels,smoke,contract=context(a)
    assert read(out/'teacher-complete.json')['status']=='passed' and Path(a.selector).name==SELECTOR_REV
    processor,backbone=load_model(a.selector)
    cfg=ExperimentConfig(name='api-correct-variable-pilot-smoke-v1',vision_mode='native',head2=True,retrieval_dim=129,retrieval_schema=RETRIEVAL_SCHEMA,stage1=Stage1Contract(answerer_revision=REVISION,selector_revision=SELECTOR_REV,epsilon=.1,margin=.05))
    model=CachedPilotSelector(build_selector(cfg,answerer_config=backbone.config,selector_model=backbone))
    model.base.backbone.model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    batches=[]
    for qid in smoke['qids']:
        example,bank=load_example(out/qid/'example');q,assets,images,regions=pages(root,catalog,qid)
        prompt=prepare_prompt(processor,q['question'],images)
        for im in images:im.close()
        native=native_action_layout(example.layout,torch.tensor(read(out/qid/'design.json')['boxes']),prompt.image_grid_thw,2)
        example=replace(example,layout=layout_to(example.layout,'cuda'),question_ids=example.question_ids.cuda(),retrieval=RetrievalFeatures(example.retrieval.values.cuda(),example.retrieval.schema))
        batches.append(dict(examples=[example],banks=[bank],proxy_prompts=[to_prompt(prompt,'cuda')],native_layouts=[layout_to(native,'cuda')]))
    started=time.monotonic();torch.cuda.reset_peak_memory_stats()
    names={n for n,p in model.named_parameters() if p.requires_grad or 'lora_' in n}
    def state():return {n:p.detach().cpu().clone() for n,p in model.named_parameters() if n in names}
    def restore(saved):
        model.load_state_dict(saved,strict=False);model.hidden_cache.clear()
    def predictions(batch,bypass=False):
        with torch.no_grad():
            enc=model(batch['examples'][0],batch['proxy_prompts'][0],native_layout=batch['native_layouts'][0],return_encoding=True,bypass_cache=bypass)
            return enc,score_candidate_masks(enc,batch['banks'][0].masks,model.correction)
    optimizer=phase_optimizer(model,'warmup');initial=state();model.eval()
    parity=[]
    for batch in batches:
        direct=predictions(batch,True)[1];cached=predictions(batch)[1]
        torch.testing.assert_close(direct.total,cached.total,rtol=0,atol=0)
        parity.append(float((direct.total-cached.total).abs().max()))
    history=[]
    for epoch in range(2):
        record=train_epoch(model,batches if epoch%2==0 else reversed(batches),cfg,optimizer)
        history.append(dict(phase='warmup',epoch=epoch,**record))
    warmup=state()
    assert all(torch.equal(warmup[n],initial[n]) for n in names if 'lora_' in n)
    assert any(not torch.equal(warmup[n],initial[n]) for n in names if 'reader' in n)
    checkpoint=out/'warmup.pt'
    torch.save(dict(schema='pilot-phase-checkpoint-v1',phase='warmup',config=asdict(cfg),contract=fingerprint(contract),parameters=warmup,optimizer=optimizer.state_dict(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all()),checkpoint)
    branches={};chosen={}
    for phase in ('frozen','lora'):
        loaded=torch.load(checkpoint,map_location='cpu',weights_only=True);restore(loaded['parameters'])
        torch.set_rng_state(loaded['torch_rng']);torch.cuda.set_rng_state_all(loaded['cuda_rng'])
        optimizer=phase_optimizer(model,phase);assert not optimizer.state
        scheduler=branch_scheduler(optimizer,4)
        for epoch in range(4):
            record=train_epoch(model,batches if epoch%2==0 else reversed(batches),cfg,optimizer,scheduler=scheduler)
            history.append(dict(phase=phase,epoch=epoch,**record))
        after=state();changes={}
        for name in names:
            group='lora' if 'lora_' in name else name.split('.')[1]
            changes[group]=changes.get(group,0.)+float((after[name]-warmup[name]).float().square().sum())
        assert changes['lora']>0 if phase=='lora' else changes['lora']==0
        assert all(changes[k]>0 for k in ('reader','head','correction'))
        model.eval();prior=[predictions(b)[1] for b in batches]
        path=out/(phase+'.pt');torch.save(dict(parameters=after,phase=phase,config=asdict(cfg),parent_sha256=sha(checkpoint),optimizer=optimizer.state_dict(),scheduler=scheduler.state_dict()),path)
        restore(torch.load(path,map_location='cpu',weights_only=True)['parameters'])
        diagnostics=[]
        for i,batch in enumerate(batches):
            enc,pred=predictions(batch);torch.testing.assert_close(pred.total,prior[i].total,rtol=0,atol=0)
            bank=batch['banks'][0];example=batch['examples'][0];pairs=torch.tensor(bank.pairs('gold_aware',.1,.05),device='cuda')
            mask,budget=allocate(enc.scores,example.layout.costs,example.budget//2)
            chosen.setdefault(smoke['qids'][i],{})[phase]=mask.cpu().tolist()
            diagnostics.append(dict(qid=smoke['qids'][i],head1_accuracy=float((pred.direct[pairs[:,0]]>pred.direct[pairs[:,1]]).float().mean()),combined_accuracy=float((pred.total[pairs[:,0]]>pred.total[pairs[:,1]]).float().mean()),selected_tokens=budget))
        branches[phase]=dict(update_squared_norms=changes,diagnostics=diagnostics,checkpoint_sha256=sha(path),reload_exact=True)
    publish(out/'trained-masks.json',chosen)
    publish(out/'training-complete.json',dict(status='passed',config=asdict(cfg),history=history,branches=branches,warmup_lora_unchanged=True,cache_parity_max_abs=parity,cache_calls=model.calls,resident_cache_bytes=model.cache_bytes(),resources=stats(started,torch)))
    print(json.dumps(read(out/'training-complete.json')),flush=True)


def evaluate(a):
    import torch
    from docprune.stage2.answerer import FrozenAnswerer
    from docprune.stage2.data import load_example
    from docprune.stage2.qwen import prepare_prompt
    root,out,catalog,pool,labels,smoke,contract=context(a)
    assert read(out/'training-complete.json')['status']=='passed'
    chosen=read(out/'trained-masks.json');processor,model=load_model(a.reader)
    reader=FrozenAnswerer(model);started=time.monotonic();torch.cuda.reset_peak_memory_stats();results=[]
    for qid in smoke['qids']:
        example,bank=load_example(out/qid/'example')
        q,assets,images,regions=pages(root,catalog,qid)
        prompt=to_prompt(prepare_prompt(processor,PROMPT+q['question'],images),'cuda')
        for im in images:im.close()
        own=read(out/qid/'anchor.json')['continuation_ids'];eos=model.generation_config.eos_token_id
        eos=[eos] if isinstance(eos,int) else eos
        while own and own[-1] in eos:own.pop()
        memory=reader.vision(prompt,fingerprint(contract));values={}
        for phase,mask in chosen[qid].items():
            indices=example.layout.retained_tokens(torch.tensor(mask,dtype=torch.bool)).cuda()
            values[phase]=dict(g=None,s=reader.likelihood(prompt,memory,indices,own)['mean'],retained_tokens=len(indices))
        results.append(dict(qid=qid,reference=asdict(bank.reference),selected=values))
        del memory,example,bank,prompt;gc.collect();torch.cuda.empty_cache()
    publish(out/'complete.json',dict(status='passed',results=results,resources=stats(started,torch),scope='wiring only; two exposed correct cases, no accuracy or generalization claim'))
    print(json.dumps(read(out/'complete.json')),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['teacher','train','evaluate'],required=True)
    for name in ('root','output','reader','selector'):p.add_argument('--'+name,required=True)
    a=p.parse_args()
    import torch,transformers
    assert transformers.__version__=='4.57.3' and torch.cuda.is_available()
    torch.set_num_threads(2)
    {'teacher':teacher,'train':train,'evaluate':evaluate}[a.phase](a)
