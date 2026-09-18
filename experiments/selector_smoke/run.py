"""Two exposed questions: real teacher -> native rich/LoRA/Head2 -> reader smoke."""
import argparse
from dataclasses import asdict, fields, replace
import gc
import json
from pathlib import Path
import sys
import time
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(REPO / 'src'), str(REPO / 'experiments/m3doc600')]
from common import fingerprint, load_catalog, publish, read, sha, stats
from baseline import PROMPT, REVISION, admitted

QIDS = ('d579841142ec52ca5ed6b765eaad8f7c', '1cf687db5007935b41eaacbf2332e27f')
SELECTOR_REV = '4bd860ac4f15ad1897a214615cccc700f8f71818'


def to_prompt(prompt, device):
    return type(prompt)(**{f.name: getattr(prompt, f.name).to(device) for f in fields(prompt)})


def layout_to(layout, device):
    return replace(layout, **{k: getattr(layout, k).to(device) for k in ('owner', 'page', 'coordinates', 'metadata')})


def random_masks(costs, budget, seed=1732, count=32):
    import torch
    from docprune.stage2.policy import allocate
    generator = torch.Generator().manual_seed(seed)
    result, seen = [], set()
    for _ in range(4096):
        mask, _ = allocate(torch.randn(len(costs), generator=generator), costs, budget)
        key = tuple(mask.tolist())
        if key not in seen:
            seen.add(key)
            result.append(mask)
        if len(result) == count:
            return torch.stack(result)
    raise ValueError('Not enough distinct feasible masks; do not invent teacher pairs')


def component(name):
    if 'lora_' in name:
        return 'lora'
    return name.split('.')[0]


def grad_report(model):
    import torch
    result = {}
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            if parameter.grad is not None:
                raise ValueError('Frozen parameter received a gradient: ' + name)
            continue
        key = component(name)
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise ValueError('Nonfinite gradient: ' + name)
            result[key] = result.get(key, 0.) + float(parameter.grad.float().square().sum())
    if set(result) != {'lora', 'reader', 'head', 'correction'} or not all(v > 0 for v in result.values()):
        raise ValueError('Missing trainable path gradient: ' + repr(result))
    return {k: v ** .5 for k, v in result.items()}


def sources():
    paths = [Path(__file__), REPO/'experiments/m3doc600/common.py', REPO/'experiments/m3doc600/baseline.py']
    paths += sorted((REPO/'src/docprune/stage2').glob('*.py'))
    return {str(p.relative_to(REPO)): sha(p) for p in paths}


def context(a):
    root = Path(a.root)
    cohort = root/'training463-evidence-filtered-v1'
    selection = read(cohort/'selection.json')
    assert selection['question_count'] == 463 and len(selection['qids']) == 463
    assert sha(cohort/'pool.json') == selection['pool_sha256']
    assert set(QIDS) <= set(selection['qids'])
    review = read(root/'omp10-20260917-v1/review/frozen.json')
    reviewed = {r['qid']: r for r in review['selected']}
    assert all(reviewed[q]['decision'] == 'confirmed_wrong' for q in QIDS)
    pool = {q['qid']: q for q in read(cohort/'pool.json')['questions']}
    catalog, digest = load_catalog(root)
    assert digest == 'bf220ddb7c0b67b58a992292108e9ec386b8694d3b33f542c71dd5d35f0b4d08'
    original = {q['qid']: q for q in read(catalog['pool_path'])['questions']}
    assert all(pool[q] == original[q] for q in QIDS), 'Filtered cohort changed retained example'
    spec = dict(schema='selector-two-question-smoke-v1', qids=QIDS, masks=32,
                cohort_selection_sha256=sha(cohort/'selection.json'), pool_sha256=selection['pool_sha256'],
                review_sha256=sha(root/'omp10-20260917-v1/review/frozen.json'),
                answerer_revision=REVISION, selector_revision=SELECTOR_REV, catalog=digest,
                mask_policy='32 distinct seeded random utility allocations at achievable half-token budget',
                exposure='previously inspected diagnostic questions, not held-out evaluation',
                teacher='full-prefix mean log likelihood; native question-first baseline prompt',
                sources=sources())
    out = Path(a.output)
    publish(out/'contract.json', spec)
    return root, out, catalog, pool, reviewed, spec


def pages(root, catalog, qid):
    from docprune.stage2.documents import PageAsset
    qs = {q['question_key']: q for q in catalog['questions']}
    ps = {p['key']: p for p in catalog['pages']}
    assets, images, regions = [], [], []
    for i, p in enumerate(admitted(qs[qid])):
        page = ps[p['key']]
        lp = root/'mineru/pages'/f"{p['key']}.json"
        asset = PageAsset(p['key'], str(root/page['image']), page['image_sha256'], str(lp), sha(lp), 'frozen-catalog')
        im, boxes = asset.read()
        # Validate the cached pixel identity before assigning prompt-occurrence identity.
        asset = replace(asset, page_id=asset.page_id+f'#occurrence:{i}')
        assets.append(asset); images.append(im); regions.append(boxes)
    return qs[qid], assets, images, regions


def load_model(snapshot):
    import torch
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    processor = AutoProcessor.from_pretrained(snapshot, local_files_only=True, min_pixels=256*32*32, max_pixels=2560*32*32)
    model = Qwen3VLForConditionalGeneration.from_pretrained(snapshot, local_files_only=True, dtype=torch.bfloat16, device_map='cuda:0', attn_implementation='sdpa').eval().requires_grad_(False)
    return processor, model


def teacher(a, final=False):
    import torch
    from docprune.stage2.answerer import FrozenAnswerer
    from docprune.stage2.contracts import InstanceIdentity, SelectorInputs
    from docprune.stage2.data import save_example, load_example
    from docprune.stage2.documents import region_layout
    from docprune.stage2.qwen import prepare_prompt
    from docprune.stage2.supervision import Outcome, TeacherBank
    root, out, catalog, pool, reviewed, spec = context(a)
    assert Path(a.reader).name == REVISION
    processor, model = load_model(a.reader)
    reader = FrozenAnswerer(model)
    torch.cuda.reset_peak_memory_stats(); started = time.monotonic()
    receipts = []
    for qid in QIDS:
        folder = out/qid
        q, assets, images, regions = pages(root, catalog, qid)
        bpath = root/'baseline-qwen3-8b-admitted-v2/answers'/f'{fingerprint(qid)}.json'
        b = read(bpath); review = reviewed[qid]
        assert sha(bpath) == review['baseline_file_sha256'] and b['answer'] == review['baseline_answer']
        assert [x['answer'] for x in pool[qid]['answers']] == review['gold_items']
        prompt = prepare_prompt(processor, PROMPT+q['question'], images)
        for im in images: im.close()
        assert prompt.input_ids[0].tolist() == b['input_token_ids']
        assert prompt.image_grid_thw.tolist() == b['image_grid_thw']
        n = int(prompt.image_grid_thw.prod(1).sum()//4)
        layout, boxes, audit = region_layout(prompt.image_grid_thw, 2, assets, regions, SimpleNamespace(fallback_tile_size=8, max_fallback_tokens=128, budget=n//2))
        matrix = random_masks(layout.costs, n//2)
        targets = review['gold_items']
        gold_text = str(targets[0]) if len(targets)==1 else json.dumps(targets, ensure_ascii=False)
        gold = processor.tokenizer.encode(gold_text, add_special_tokens=False)
        own = list(b['answer_token_ids']); eos = model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos,int) else eos
        while own and own[-1] in eos: own.pop()
        assert own and gold
        identity = InstanceIdentity('m3docvqa', fingerprint(review['support_families']), qid, fingerprint(q['question']), tuple(b['ordered_pages']), fingerprint(b['image_grid_thw']), fingerprint([layout.region_ids,layout.owner.tolist()]), REVISION, spec['catalog'], fingerprint([PROMPT,gold,own,'full-prefix']))
        design = dict(instance=asdict(identity),question=q['question'],region_ids=layout.region_ids,owner=layout.owner.tolist(),boxes=boxes.tolist(),masks=matrix.tolist(),gold=gold,self_target=own,audit=audit)
        publish(folder/'design.json',design)
        prompt = to_prompt(prompt,'cuda')
        memory = reader.vision(prompt, fingerprint(spec))
        def score(mask):
            tokens = layout.retained_tokens(mask).cuda()
            return {'g':reader.likelihood(prompt,memory,tokens,gold)['mean'], 's':reader.likelihood(prompt,memory,tokens,own)['mean']}
        if final:
            masks=read(out/'trained-masks.json')[qid]
            vals={key:score(torch.tensor(value,dtype=torch.bool)) for key,value in masks.items()}
            publish(folder/'selected-reader-scores.json',vals); receipts.append(dict(qid=qid,**vals))
        else:
            refpath=folder/'reference.json'
            if not refpath.exists(): publish(refpath,score(torch.ones(len(layout.region_ids),dtype=torch.bool)))
            outcomes=[]
            for i,mask in enumerate(matrix):
                path=folder/'measurements'/f'{i:02d}.json'
                if not path.exists(): publish(path,dict(mask=mask.tolist(),**score(mask)))
                row=read(path); assert row['mask']==mask.tolist()
                outcomes.append(Outcome(row['g'],row['s']))
            repeat=score(matrix[0]); first=outcomes[0]
            assert max(abs(repeat['g']-first.g),abs(repeat['s']-first.s)) < 1e-5
            example=SelectorInputs(identity,layout,memory,prompt.input_ids[0,prompt.question_positions].cpu(),n//2)
            bank=TeacherBank(identity.key,matrix,tuple(outcomes),Outcome(**read(refpath)),False,False,identity.document_family)
            bank.validate(example)
            if not (folder/'example').exists(): save_example(folder/'example',example,bank)
            else: load_example(folder/'example')
            pairs=len(bank.pairs('gold_aware',.1,.05))
            if pairs==0: raise ValueError('No meaningful teacher preferences; smoke cannot test training on this question')
            receipts.append(dict(qid=qid,pairs=pairs,regions=len(layout.region_ids),visual_tokens=n,repeat=repeat))
            print(json.dumps(receipts[-1]),flush=True)
        del memory, prompt
        if not final: del example, bank
        gc.collect(); torch.cuda.empty_cache()
    publish(out/('complete.json' if final else 'teacher-complete.json'),dict(status='passed',phase='reader-evaluation' if final else 'teacher',questions=receipts,resources=stats(started,torch)))


def train(a):
    import torch
    from docprune.stage2.data import load_example
    from docprune.stage2.documents import native_action_layout
    from docprune.stage2.experiment import ExperimentConfig, Stage1Contract
    from docprune.stage2.policy import allocate, score_candidate_masks
    from docprune.stage2.qwen import prepare_prompt
    from docprune.stage2.training import build_selector, objective, save_checkpoint, restore_checkpoint
    root,out,catalog,pool,reviewed,spec=context(a)
    teacher_out=Path(a.teacher_source) if a.teacher_source else out
    prior=read(teacher_out/'contract.json')
    for key in ('qids','cohort_selection_sha256','pool_sha256','answerer_revision','selector_revision','catalog','mask_policy','teacher'):
        assert prior[key] == json.loads(json.dumps(spec))[key], key
    assert read(teacher_out/'teacher-complete.json')['status']=='passed'
    assert Path(a.selector).name==SELECTOR_REV
    processor,backbone=load_model(a.selector)
    assert backbone.config.text_config.hidden_size==2048 and backbone.config.vision_config.depth==24
    cfg=ExperimentConfig(name='two-question-native-rich-head2-smoke',vision_mode='native',head2=True,stage1=Stage1Contract(answerer_revision=REVISION,selector_revision=SELECTOR_REV,epsilon=.1,margin=.05))
    # Diagnostic-only configuration: no fabricated production interface/split receipt.
    model=build_selector(cfg,answerer_config=backbone.config,selector_model=backbone)
    model.backbone.model.language_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant':False})
    params={n:p for n,p in model.named_parameters() if p.requires_grad}
    assert all('lora_' in n or n.startswith(('reader.','head.','correction.')) for n in params)
    assert not any(p.requires_grad for p in model.backbone.model.visual.parameters())
    initial={n:p.detach().cpu().clone() for n,p in params.items()}
    optimizer=torch.optim.AdamW(list(params.values()),lr=cfg.learning_rate,weight_decay=cfg.weight_decay)
    batches=[]
    for qid in QIDS:
        example,bank=load_example(teacher_out/qid/'example')
        q,assets,images,regions=pages(root,catalog,qid)
        prompt=prepare_prompt(processor,q['question'],images)
        for im in images: im.close()
        boxes=torch.tensor(read(teacher_out/qid/'design.json')['boxes'])
        native=native_action_layout(example.layout,boxes,prompt.image_grid_thw,2)
        # Reader vision cache stays on CPU; native selector ignores its feature values.
        example=replace(example,layout=layout_to(example.layout,'cuda'),question_ids=example.question_ids.cuda())
        batches.append(dict(examples=[example],banks=[bank],proxy_prompts=[to_prompt(prompt,'cuda')],native_layouts=[layout_to(native,'cuda')]))
    started=time.monotonic(); torch.cuda.reset_peak_memory_stats()
    history=[]
    for step in range(4):
        model.train(); optimizer.zero_grad(set_to_none=True)
        loss=objective(model,config=cfg,**batches[step%2])
        if loss is None or not torch.isfinite(loss): raise ValueError('Invalid training loss')
        loss.backward(); gradients=grad_report(model)
        torch.nn.utils.clip_grad_norm_(list(params.values()),1.)
        optimizer.step(); torch.cuda.synchronize()
        record=dict(step=step+1,qid=QIDS[step%2],loss=float(loss.detach()),gradient_norms=gradients)
        history.append(record);print(json.dumps(record),flush=True)
    updates={}
    for n,p in params.items(): updates[component(n)]=updates.get(component(n),0.)+float((p.detach().cpu()-initial[n]).float().square().sum())
    assert set(updates)=={'lora','reader','head','correction'} and all(v>0 for v in updates.values())
    model.eval()
    def predictions(batch):
        with torch.no_grad():
            enc=model(batch['examples'][0],batch['proxy_prompts'][0],native_layout=batch['native_layouts'][0],return_encoding=True)
            pred=score_candidate_masks(enc,batch['banks'][0].masks,model.correction)
            return enc,pred
    before=[predictions(b) for b in batches]
    checkpoint=out/'checkpoint.pt'
    save_checkpoint(checkpoint,model,cfg,step=4,optimizer=optimizer)
    with torch.no_grad():
        for p in params.values():p.add_(.01)
    assert restore_checkpoint(checkpoint,model,cfg,optimizer=optimizer)==4
    chosen={};reports=[]
    for i,batch in enumerate(batches):
        enc,pred=predictions(batch)
        torch.testing.assert_close(pred.direct,before[i][1].direct,rtol=0,atol=0)
        torch.testing.assert_close(pred.total,before[i][1].total,rtol=0,atol=0)
        example,bank=batch['examples'][0],batch['banks'][0]
        mask,target=allocate(enc.scores,example.layout.costs,example.budget)
        chosen[QIDS[i]]={'direct':mask.cpu().tolist(),'head2_bank_best':bank.masks[int(pred.total.argmax())].tolist()}
        pairs=torch.tensor(bank.pairs(cfg.supervision,.1,.05),device=pred.direct.device)
        reports.append(dict(qid=QIDS[i],pairs=len(pairs),direct_pair_accuracy=float((pred.direct[pairs[:,0]]>pred.direct[pairs[:,1]]).float().mean()),total_pair_accuracy=float((pred.total[pairs[:,0]]>pred.total[pairs[:,1]]).float().mean()),retained_tokens=target))
    publish(out/'trained-masks.json',chosen)
    publish(out/'training-complete.json',dict(status='passed',config=asdict(cfg),history=history,update_squared_norms=updates,checkpoint_sha256=sha(checkpoint),checkpoint_reload_exact=True,trainable_parameters=sum(p.numel() for p in params.values()),diagnostics=reports,resources=stats(started,torch)))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['teacher','train','evaluate'],required=True)
    p.add_argument('--root',required=True);p.add_argument('--output',required=True);p.add_argument('--reader',required=True);p.add_argument('--selector',required=True)
    p.add_argument('--teacher-source',help='Verified completed teacher bank from an earlier wiring attempt')
    a=p.parse_args()
    import torch,transformers
    assert transformers.__version__=='4.57.3' and torch.cuda.is_available()
    torch.set_num_threads(2)
    if a.phase=='train':train(a)
    else:teacher(a,final=a.phase=='evaluate')
