"""Native unpruned Qwen3 baseline; immutable cached page admission, no masking."""
import argparse
import importlib.metadata as metadata
import json
import os
import time
from pathlib import Path
from common import fingerprint, load_catalog, open_image, publish, read, sha, stats

REVISION = '0c351dd01ed87e9c1b53cbc748cba10e6187ff3b'
PROMPT = ('Answer the question using the supplied document pages. Give only the complete '
          'answer, without explanation. For multiple answer items, output a JSON array of '
          'strings. Include units when needed. If the pages do not provide the answer, '
          'respond with "Not answerable".\n\nQuestion: ')


def admitted(q, rankings=None):
    order=q['presentation_order']
    assert sorted(order)==list(range(len(q['pages'])))
    return [q['pages'][i] for i in order]


def smoke_questions(questions, pool):
    # Longest admitted context within each modality/coverage stratum; no answers.
    groups={}
    for q in questions:
        meta=pool[q['question_key']]
        key=(meta['metadata']['type'],meta['operational_coverage']['within_original_top4'])
        groups.setdefault(key,[]).append(q)
    return [max(groups[k],key=lambda q:(len(q['pages']),len(q['question']),q['question_key'])) for k in sorted(groups)]


def main(args):
    import torch
    import transformers
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration, GenerationConfig
    assert torch.cuda.is_available() and torch.cuda.get_device_capability(0)[0]>=8
    assert Path(args.snapshot).name == REVISION
    assert transformers.__version__ == '4.57.3'
    torch.set_num_threads(2)
    root = Path(args.root)
    c, digest = load_catalog(root)
    completion = read(root/'combined/completion.json')
    assert completion['status'] == 'complete' and completion['catalog_sha256'] == digest
    assert c['stage']=='owner_approved_admitted471'
    assert digest=='bf220ddb7c0b67b58a992292108e9ec386b8694d3b33f542c71dd5d35f0b4d08'
    rankings = read(root/'combined/rankings.json')
    pages = {p['key']: p for p in c['pages']}
    all_q = sorted(c['questions'], key=lambda q: q['question_key'])
    assert len(all_q) == 471 and set(rankings) == {q['question_key'] for q in all_q}
    for q in all_q:
        admitted(q, rankings)
    pool={q['qid']:q for q in read(c['pool_path'])['questions']}
    assert set(pool)=={q['question_key'] for q in all_q}
    if args.smoke:
        selected=smoke_questions(all_q,pool)
    else:
        assert 0 <= args.shard < args.shards
        selected = all_q[args.shard::args.shards]
    out = root/'baseline-qwen3-8b-admitted-v1'
    contract = dict(model='Qwen/Qwen3-VL-8B-Instruct', revision=REVISION,
                    catalog_sha256=digest, rankings_sha256=sha(root/'combined/rankings.json'),
                    code_sha256=sha(__file__), common_sha256=sha(Path(__file__).with_name('common.py')),
                    top_k=None, page_order='frozen gold-independent presentation_order', prompt=PROMPT,
                    max_pixels=2560*32*32, min_pixels=256*32*32,
                    dtype='bfloat16', attention='sdpa', max_new_tokens=256,
                    do_sample=False, repetition_penalty=1.0, retention='all',
                    packages={k: metadata.version(k) for k in ['torch','transformers','torchvision','Pillow','accelerate']})
    identity = fingerprint(contract)
    publish(out/'contract.json', dict(contract, sha256=identity))
    if not args.smoke:
        smoke = read(out/'smoke.json')
        assert smoke['contract_sha256'] == identity and smoke['status'] == 'complete'
    receipt_path = out/('smoke.json' if args.smoke else f'runs/shard-{args.shard}-of-{args.shards}.json')
    if receipt_path.exists():
        old = read(receipt_path)
        assert old['contract_sha256'] == identity
        assert old['questions'] == [q['question_key'] for q in selected]
        for q in selected:
            record = read(out/'answers'/f"{fingerprint(q['question_key'])}.json")
            assert record['contract_sha256'] == identity
        print(json.dumps(old), flush=True)
        return
    start = time.monotonic()
    processor = AutoProcessor.from_pretrained(args.snapshot, local_files_only=True,
                    min_pixels=contract['min_pixels'], max_pixels=contract['max_pixels'])
    model = Qwen3VLForConditionalGeneration.from_pretrained(args.snapshot,
                    local_files_only=True, dtype=torch.bfloat16, device_map='cuda:0',
                    attn_implementation='sdpa').eval().requires_grad_(False)
    generation = GenerationConfig(do_sample=False, max_new_tokens=256,
                    repetition_penalty=1.0, use_cache=True,
                    eos_token_id=model.generation_config.eos_token_id,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    bos_token_id=model.generation_config.bos_token_id)
    done = []
    for q in selected:
        target = out/'answers'/f"{fingerprint(q['question_key'])}.json"
        keep = admitted(q, rankings)
        if target.exists():
            record = read(target)
            assert record['contract_sha256'] == identity and record['question_key'] == q['question_key']
            assert record['ordered_pages'] == [p['page_id'] for p in keep]
            done.append(q['question_key'])
            continue
        images = [open_image(root, pages[p['key']]) for p in keep]
        content = [{'type':'text', 'text':PROMPT+q['question']}] + [{'type':'image'} for _ in images]
        text = processor.apply_chat_template([{'role':'user','content':content}], tokenize=False, add_generation_prompt=True)
        inputs = processor(text=[text], images=images, return_tensors='pt', padding=False).to('cuda')
        for im in images:
            im.close()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        begin = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(**inputs, generation_config=generation)
        torch.cuda.synchronize()
        answer_ids = generated[0, inputs['input_ids'].shape[1]:].tolist()
        eos = generation.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos
        record = dict(question_key=q['question_key'], source=q['source'], question=q['question'],
                      contract_sha256=identity, ordered_pages=[p['page_id'] for p in keep],
                      image_sha256=[pages[p['key']]['image_sha256'] for p in keep],
                      answer=processor.tokenizer.decode(answer_ids, skip_special_tokens=True).strip(),
                      answer_token_ids=answer_ids, input_token_ids=inputs['input_ids'][0].tolist(),
                      image_grid_thw=inputs['image_grid_thw'].tolist(),
                      visual_tokens=int(inputs['image_grid_thw'].prod(dim=1).sum().item()//4),
                      hit_generation_limit=bool(len(answer_ids)>=256 and answer_ids[-1] not in eos),
                      correctness='pending official EM/F1 evaluation', context_stratum=('original_four' if pool[q['question_key']]['operational_coverage']['within_original_top4'] else 'supplemented'), modality=pool[q['question_key']]['metadata']['type'], stats=stats(begin, torch))
        publish(target, record)
        done.append(q['question_key'])
        print(json.dumps({k:record[k] for k in ['question_key','answer','visual_tokens','hit_generation_limit','stats']}), flush=True)
        del inputs, generated
        torch.cuda.empty_cache()
    receipt = dict(status='complete', contract_sha256=identity, questions=done, stats=stats(start, torch))
    if args.smoke:
        for q in selected:
            r = read(out/'answers'/f"{fingerprint(q['question_key'])}.json")
            assert r['answer'] and not r['hit_generation_limit'], 'Inspect incomplete smoke answer'
    publish(receipt_path, receipt)
    print(json.dumps(receipt), flush=True)


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    p.add_argument('--snapshot', required=True)
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--shard', type=int, default=0)
    p.add_argument('--shards', type=int, default=1)
    main(p.parse_args())
