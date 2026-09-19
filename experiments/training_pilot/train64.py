"""Streaming pilot training, fixed dev ranking and epoch-boundary recovery."""
import argparse
from dataclasses import asdict,replace
import math
from pathlib import Path
import random
import sys
import time
sys.path[:0]=[str(Path(__file__).resolve().parents[2]/'src'),str(Path(__file__).resolve().parents[2])]
import torch
from experiments.training_pilot.run import (read,sha,load_catalog,pages,load_model,to_prompt,layout_to,SELECTOR_REV,REVISION,RETRIEVAL_SCHEMA,stats)
from docprune.stage2.storage import publish,read_record,run_lock,runtime_identity
from docprune.stage2.contracts import RetrievalFeatures,fingerprint
from docprune.stage2.data import load_example
from docprune.stage2.documents import native_action_layout
from docprune.stage2.qwen import prepare_prompt,PackedPrompt
from docprune.stage2.experiment import ExperimentConfig,Stage1Contract
from docprune.stage2.training import build_selector
from docprune.stage2.pilot import CachedPilotSelector,phase_optimizer,train_epoch,branch_scheduler,encoding_cache_key
from docprune.stage2.pilot_runtime import TensorCache,checkpoint,restore_parameters,adaptable_state
from docprune.stage2.policy import score_candidate_masks,allocate


TASK_PROMPT_SYSTEM = (
    'Judge whether the Document meets the requirements based on the Query and the '
    'Instruct provided. Note that the answer can only be "yes" or "no".'
)
TASK_PROMPT_INSTRUCTION = (
    'Assess the document for evidence needed to answer the query accurately. '
    'Distinguish evidence matching the requested entity, role, attribute, conditions, '
    'and time period from content that is only superficially related. Consider headers, '
    'captions, identity cues, and combinations needed to interpret the evidence. A '
    'region may contain both useful and confusing information. Do not assume a '
    'distractor exists.'
)


def prompt_spec(condition):
    if condition == 'current':
        return dict(condition='current', schema='raw-question-before-document-v1',
                    system=None, instruction=None)
    if condition == 'evidence-v1':
        return dict(condition='evidence-v1', schema='qwen-reranker-instruction-v1',
                    system=TASK_PROMPT_SYSTEM, instruction=TASK_PROMPT_INSTRUCTION)
    raise ValueError(f'Unknown prompt condition: {condition}')


def phase_plan(branches):
    if branches == 'all':return (('warmup',2),('frozen',4),('lora',4))
    if branches == 'frozen-only':return (('warmup',2),('frozen',4))
    raise ValueError(f'Unknown branch selection: {branches}')


def ranking_metrics(values,bank):
    pairs=bank.pairs('gold_aware',.1,.05)
    if not pairs:return None
    plus,minus=torch.tensor(pairs,device=values.device).T
    delta=values[plus].float()-values[minus].float()
    if not torch.isfinite(delta).all():raise ValueError('Nonfinite development prediction')
    distance=(bank.masks[plus.cpu()]!=bank.masks[minus.cpu()]).sum(1).to(values.device)
    groups=[f for f in (distance==1,distance==2,distance>2) if f.any()]
    return dict(loss=float(torch.stack([torch.nn.functional.softplus(-delta[f]).mean() for f in groups]).mean()),
        accuracy=float(torch.stack([(delta[f]>0).float().mean() for f in groups]).mean()),pairs=len(pairs))


def regional_metadata_ablation(layout, zero_token_count=False):
    if not zero_token_count:
        return layout
    # Frozen region schema: six coordinate extrema, page index, raw token count.
    if layout.metadata.shape[1] != 8 or not torch.equal(
        layout.metadata[:, -1].cpu(), layout.costs.to(layout.metadata).cpu()
    ):
        raise ValueError('Regional token-count metadata schema mismatch')
    metadata = layout.metadata.clone()
    metadata[:, -1] = 0
    return replace(layout, metadata=metadata)


class Stream:
    def __init__(self,root,rows,processor,cache,device='cuda',prompt_condition='current',zero_token_count=False):
        self.root=Path(root);self.rows={r['qid']:r for r in rows};self.processor=processor;self.cache=cache;self.device=device
        self.prompt_spec=prompt_spec(prompt_condition)
        self.zero_token_count=zero_token_count
        self.catalog,_=load_catalog(self.root)
    def _prepare(self,question,images,spec=None):
        spec=self.prompt_spec if spec is None else spec
        return prepare_prompt(self.processor,question,images,
            instruction=spec['instruction'],system=spec['system'])
    def prompt_audit(self,qid,model_config):
        q,assets,images,regions=pages(self.root,self.catalog,qid)
        try:
            current=self._prepare(q['question'],images,prompt_spec('current'))
            proposed=self._prepare(q['question'],images)
        finally:
            for im in images:im.close()
        if not torch.equal(current.image_grid_thw,proposed.image_grid_thw):
            raise ValueError('Prompt condition changed admitted image grids')
        if not torch.equal(current.pixel_values,proposed.pixel_values):
            raise ValueError('Prompt condition changed admitted page pixels')
        text=getattr(model_config,'text_config',model_config)
        limit=getattr(text,'max_position_embeddings',None)
        length=int(proposed.input_ids.shape[1])
        if limit is not None and length>limit:
            raise ValueError(f'Task prompt exceeds model context: {length}>{limit}')
        return dict(qid=qid,pages=len(images),current_input_tokens=int(current.input_ids.shape[1]),
            proposed_input_tokens=length,context_limit=limit,
            image_grid_thw=proposed.image_grid_thw.tolist(),image_grid_equal=True,
            pixel_values_equal=True,question_tokens=int(proposed.question_positions.numel()),
            question_positions=proposed.question_positions.tolist())
    def batch(self,qid):
        row=self.rows[qid];folder=Path(row['directory'])
        assert sha(folder/'example/manifest.json')==row['manifest_sha256']
        if 'design_sha256' in row:assert sha(folder/'design.json')==row['design_sha256']
        ex,bank=load_example(folder/'example')
        key=fingerprint([qid,row['manifest_sha256'],self.prompt_spec]);saved=self.cache.get('prepared-prompt',key)
        if saved is None:
            q,assets,images,regions=pages(self.root,self.catalog,qid)
            try:prompt=self._prepare(q['question'],images)
            finally:
                for im in images:im.close()
            saved={k:getattr(prompt,k) for k in ('input_ids','question_positions','image_grid_thw')}
            self.cache.put('prepared-pixels',key,{'pixel_values':prompt.pixel_values})
            self.cache.put('prepared-prompt',key,saved)
        prompt=PackedPrompt(**saved)
        native=native_action_layout(ex.layout,torch.tensor(read(folder/'design.json')['boxes']),prompt.image_grid_thw,2)
        if not self.cache.contains('native-vision',encoding_cache_key(ex,prompt,native)):
            prompt.pixel_values=self.cache.get('prepared-pixels',key)['pixel_values']
        ex=replace(ex,layout=layout_to(regional_metadata_ablation(ex.layout,self.zero_token_count),self.device),question_ids=ex.question_ids.to(self.device),
            retrieval=RetrievalFeatures(ex.retrieval.values.to(self.device),ex.retrieval.schema))
        return dict(examples=[ex],banks=[bank],proxy_prompts=[PackedPrompt(**{k:None if v is None else v.to(self.device) for k,v in vars(prompt).items()})],native_layouts=[layout_to(native,self.device)])
    def batches(self,qids):
        for qid in qids:
            batch=self.batch(qid)
            yield batch
            del batch


@torch.no_grad()
def evaluate(model,stream,qids,*,select=True):
    model.eval();rows=[];selected={}
    for qid,batch in zip(qids,stream.batches(qids)):
        ex,bank=batch['examples'][0],batch['banks'][0]
        enc=model(ex,batch['proxy_prompts'][0],native_layout=batch['native_layouts'][0],return_encoding=True)
        pred=score_candidate_masks(enc,bank.masks,model.correction)
        if not torch.isfinite(pred.total).all() or not torch.isfinite(pred.direct).all():raise ValueError('Nonfinite dev output')
        costs=ex.layout.costs
        rows.append(dict(qid=qid,correct=bank.baseline_correct,head1=ranking_metrics(pred.direct,bank),combined=ranking_metrics(pred.total,bank),
            retention=ranking_metrics((bank.masks.to(costs)*costs).sum(1),bank),score_std=float(enc.scores.float().std(unbiased=False))))
        selected[qid]={}
        for retention in ((.75,.5) if select else ()):
            mask,tokens=allocate(enc.scores,costs,math.floor(ex.budget*retention))
            selected[qid][str(retention)]=dict(mask=mask.cpu().tolist(),retained_tokens=int(tokens),full_tokens=ex.budget)
    active=[r['head1']['loss'] for r in rows if r['head1'] is not None]
    if not active:raise ValueError('No fixed development preferences; cannot select checkpoint')
    return dict(head1_loss=sum(active)/len(active),rows=rows),selected


def run(a):
    out=Path(a.output);audit=read_record(a.audit)
    prompt_condition=getattr(a,'prompt_condition','current')
    branches=getattr(a,'branches','all')
    zero_token_count=getattr(a,'zero_regional_token_count',False)
    preflight_only=getattr(a,'preflight_only',False)
    prompt=prompt_spec(prompt_condition);phases=phase_plan(branches)
    if (out/'training-complete.json').exists():
        assert read_record(out/'contract.json')['audit_sha256']==sha(a.audit)
        return
    if audit['status']!='passed' or not audit['train_active'] or not audit['dev_active']:raise ValueError('Complete usable audited banks required')
    train=[r['qid'] for r in audit['rows'] if r['split']=='train' and r['pairs']]
    dev=[r['qid'] for r in audit['rows'] if r['split']=='dev']
    assert len(dev)==24 and not set(train)&set(dev) and Path(a.selector).name==SELECTOR_REV
    torch.manual_seed(a.seed);random.seed(a.seed);torch.set_num_threads(2)
    if torch.cuda.is_available():
        import transformers
        if torch.__version__!='2.8.0+cu128' or transformers.__version__!='4.57.3':raise ValueError('Pinned SOL runtime required')
        if not torch.cuda.is_bf16_supported():raise ValueError('Native BF16 GPU required')
    cfg=ExperimentConfig(name='quality-first-pilot64-v1',seed=a.seed,vision_mode='native',head2=True,retrieval_dim=129,retrieval_schema=RETRIEVAL_SCHEMA,
        stage1=Stage1Contract(answerer_revision=REVISION,selector_revision=SELECTOR_REV,epsilon=.1,margin=.05))
    processor,backbone=load_model(a.selector)
    code=[Path(__file__),Path(__file__).with_name('run.py'),Path(__file__).parents[1]/'selector_smoke/run.py',*sorted((Path(__file__).parents[2]/'src/docprune/stage2').glob('*.py'))]
    contract=dict(schema='pilot64-trainer-v2',audit_sha256=sha(a.audit),config=asdict(cfg),seed=a.seed,train=train,dev=dev,
        zero_regional_token_count=zero_token_count,prompt=prompt,branches=branches,phases=list(phases),preflight_only=preflight_only,
        retention=[.75,.5],sources={str(p.relative_to(Path(__file__).parents[2])):sha(p) for p in code},runtime=runtime_identity(backbone))
    publish(out/'contract.json',contract)
    cache=TensorCache(out/'cache',dict(selector=SELECTOR_REV,runtime=contract['runtime'],processor=processor.to_dict() if hasattr(processor,'to_dict') else str(type(processor)),sources=contract['sources'],audit=contract['audit_sha256'],gpu=torch.cuda.get_device_name() if torch.cuda.is_available() else 'cpu'))
    model=CachedPilotSelector(build_selector(cfg,answerer_config=backbone.config,selector_model=backbone),disk_cache=cache)
    model.base.backbone.model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    stream=Stream(a.root,audit['rows'],processor,cache,prompt_condition=prompt_condition,zero_token_count=zero_token_count)
    started=time.monotonic();torch.cuda.reset_peak_memory_stats();identity=fingerprint(contract)
    warmup=out/'warmup-latest.pt';updates=math.ceil(len(train)/4)
    if not (out/'cache-preflight.json').exists():
        # One largest training context: new disk path is checked before any update.
        largest=max((r for r in audit['rows'] if r['qid'] in train),key=lambda r:r.get('full_tokens',0))
        if prompt_condition!='current':
            publish(out/'prompt-audit.json',stream.prompt_audit(largest['qid'],backbone.config))
        model.set_phase('warmup');model.eval();batch=stream.batch(largest['qid'])
        with torch.no_grad():
            args=(batch['examples'][0],batch['proxy_prompts'][0])
            direct=model(*args,native_layout=batch['native_layouts'][0]).detach()
            model.hidden_cache.clear();model.vision_cache.clear()
            cached_prompt=replace(args[1],pixel_values=None)
            cached=model(args[0],cached_prompt,native_layout=batch['native_layouts'][0]).detach()
            torch.testing.assert_close(direct,cached,rtol=0,atol=0)
        publish(out/'cache-preflight.json',dict(qid=largest['qid'],exact=True,resources=stats(started,torch),disk_bytes=cache.written_bytes))
        del args,batch,direct,cached,cached_prompt
    if preflight_only:
        publish(out/'preflight-complete.json',dict(status='passed',contract=identity,
            prompt_audit_sha256=sha(out/'prompt-audit.json'),cache_preflight_sha256=sha(out/'cache-preflight.json'),
            resources=stats(started,torch)))
        return

    for phase,epochs in phases:
        latest=out/(phase+'-latest.pt');done=out/(phase+'-complete.json')
        if done.exists():continue
        # Restore weights BEFORE changing cache phase (LoRA -> frozen requires identity).
        resume=restore_parameters(latest,model,identity) if latest.exists() else None
        if resume is None and phase!='warmup':restore_parameters(warmup,model,identity)
        optimizer=phase_optimizer(model,phase)
        publish(out/(phase+'-parameters.json'),{n:list(p.shape) for n,p in model.named_parameters() if p.requires_grad})
        scheduler=None if phase=='warmup' else branch_scheduler(optimizer,epochs*updates)
        history=[];start=0;best=None;stale=0
        if resume:
            assert resume['phase']==phase
            optimizer.load_state_dict(resume['optimizer'])
            if scheduler:scheduler.load_state_dict(resume['scheduler'])
            history=resume['history'];start=resume['epoch'];best=resume['best'];stale=resume['stale']
        if phase=='warmup' and start==0 and not (out/'untrained.json').exists():
            metrics,masks=evaluate(model,stream,dev);publish(out/'untrained.json',dict(metrics=metrics,masks=masks))
        for epoch in range(start,epochs):
            if phase!='warmup' and stale>=2:break
            order=list(train);random.Random(a.seed+epoch).shuffle(order)
            result=train_epoch(model,stream.batches(order),cfg,optimizer,scheduler=scheduler)
            metrics,masks=evaluate(model,stream,dev,select=False)
            history.append(dict(epoch=epoch+1,training=result,development=metrics,resources=stats(started,torch)))
            improved=best is None or metrics['head1_loss']<best
            if improved:best=metrics['head1_loss'];stale=0
            else:stale+=1
            # Best is written first; latest is the epoch commit marker.
            if improved:
                checkpoint(out/(phase+'-best.pt'),model,identity,phase,epoch+1,optimizer,scheduler,history,best,stale)
            checkpoint(latest,model,identity,phase,epoch+1,optimizer,scheduler,history,best,stale)
            print(dict(phase=phase,epoch=epoch+1,head1_dev_loss=metrics['head1_loss'],updates=result['updates']),flush=True)
        if phase!='warmup':restore_parameters(out/(phase+'-best.pt'),model,identity)
        metrics,masks=evaluate(model,stream,dev)
        lora_nonzero=any(torch.count_nonzero(p).item() for n,p in model.named_parameters() if 'lora_B' in n)
        if lora_nonzero != (phase=='lora'):raise ValueError('LoRA phase failed identity/update check')
        publish(done,dict(phase=phase,lora_nonzero=lora_nonzero,history=history,metrics=metrics,masks=masks,epochs=len(history),latest_sha256=sha(latest)))
    publish(out/'training-complete.json',dict(status='passed',contract=identity,resources=stats(started,torch),cache_calls=model.calls,
        disk_written_bytes=cache.written_bytes,disk_read_bytes=cache.read_bytes,resident_cache_bytes=model.cache_bytes(),
        bank_shapes=[{k:r[k] for k in ('qid','pages','full_tokens') if k in r} for r in audit['rows']],
        evaluation='Fixed bank ranking and selected masks only; reader evaluation is a separate process'))

if __name__=='__main__':
    p=argparse.ArgumentParser()
    for n in ('root','output','audit','selector'):p.add_argument('--'+n,required=True)
    p.add_argument('--seed',type=int,default=0)
    p.add_argument('--prompt-condition',choices=('current','evidence-v1'),default='current')
    p.add_argument('--branches',choices=('all','frozen-only'),default='all')
    p.add_argument('--preflight-only',action='store_true')
    p.add_argument('--zero-regional-token-count',action='store_true')
    a=p.parse_args()
    with run_lock(a.output):run(a)
