#!/usr/bin/env python3
"""Owner-approved Q12 diagnostic; cannot produce an admissible smoke receipt."""
import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from docprune.acquisition_io import read_record, validate_package, write_new, token_identity, sha
from docprune.acquisition_sol import sol_preflight
from docprune.adaptive_acquisition import scores

# Fixed before any diagnostic observations; each bank entry retains 5,016 tokens.
MASK_SLOTS = (('A', 0), ('A', 16), ('R', 8), ('R', 24))


def main():
    parser = argparse.ArgumentParser()
    for name in ('package', 'snapshot', 'output'):
        parser.add_argument('--'+name, type=Path, required=True)
    parser.add_argument('--execute-gpu', action='store_true')
    args = parser.parse_args()
    if not args.execute_gpu:
        raise ValueError('Explicit --execute-gpu is required')
    package_identity = validate_package(args.package)
    preflight = sol_preflight(SimpleNamespace(command='smoke', physical_gpu=None,
                              package=args.package, output=args.output))
    if float(preflight['memory_total_mb']) < 39000:
        raise ValueError('The dense per-page comparison requires a 40GB-class GPU')
    import torch
    import torch.nn.functional as F
    from torch.nn.attention import sdpa_kernel, SDPBackend
    import docprune.acquisition_attention as attention
    from docprune.acquisition_reader import QwenTeacher, QuestionCheckpoint
    runtime = read_record(args.package/'runtime.json')
    case = read_record(args.package/'cases/Q12.json')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    def save(name, value):
        record = {'passed': False, 'diagnostic_only': True, 'case': 'Q12',
                  'elapsed_seconds': time.monotonic()-started, **value}
        write_new(args.output/(name+'.json'), record)
        print(json.dumps({'saved': name, **record}), flush=True)
    def delta(a,b):
        return [float(x-y) for x,y in zip(a,b)]
    def gsc(values):
        g,s=scores(values)
        return [g,s,g-s]
    save('identity', {'preflight': preflight, 'package':package_identity,
         'diagnostic_script_sha256':sha(Path(__file__)), 'mask_slots':MASK_SLOTS})
    teacher = QwenTeacher(runtime,args.snapshot)
    save('environment',{'environment':teacher.identity})

    class DiagnosticCheckpoint(QuestionCheckpoint):
        def _guard(self):
            # Only this diagnostic subclass observes failed parity. It never
            # marks itself guarded and cannot call the production score method.
            return

    def segmented_math(q,k,v,cu):
        bounds=cu.cpu().tolist()
        result=[]
        with sdpa_kernel([SDPBackend.MATH]):
            for a,b in zip(bounds,bounds[1:]):
                result.append(F.scaled_dot_product_attention(q[a:b].transpose(0,1),
                    k[a:b].transpose(0,1),v[a:b].transpose(0,1),dropout_p=0.0).transpose(0,1))
        return torch.cat(result)

    efficient = attention.segmented_sdpa
    results={}
    checkpoint=None
    try:
        for variant in ('efficient','segmented_math'):
            attention.segmented_sdpa=efficient if variant=='efficient' else segmented_math
            torch.cuda.empty_cache()
            checkpoint=DiagnosticCheckpoint(teacher,case,args.package)
            save(variant+'-prepared', {'variant':variant,'preparation_seconds':checkpoint.preparation_seconds,
                                     'peak_gpu_bytes':torch.cuda.max_memory_allocated()})
            all_ids=list(range(checkpoint.population))
            baseline=checkpoint._measure(all_ids)['likelihoods']
            results[variant]={'baseline':gsc(baseline),'masks':{}}
            save(variant+'-baseline', {'likelihoods':baseline, 'G_S_C':gsc(baseline),
                 'reference':case['reader']['reference_likelihoods'],
                 'reference_delta':delta(baseline,case['reader']['reference_likelihoods'])})
            for arm,slot in MASK_SLOTS:
                name=arm+str(slot)
                mask_hash,ids=token_identity(case['public'],case['public']['banks'][arm][slot])
                measured=checkpoint._measure(ids)['likelihoods']
                change=delta(gsc(measured),gsc(baseline))
                results[variant]['masks'][name]={'G_S_C':gsc(measured),'delta_G_S_C':change}
                save(variant+'-'+name, {'mask_hash':mask_hash,'retained_tokens':len(ids),
                     'likelihoods':measured,'G_S_C':gsc(measured),'delta_G_S_C':change,
                     'peak_gpu_bytes':torch.cuda.max_memory_allocated()})
                if (arm,slot)==MASK_SLOTS[0]:
                    legacy=checkpoint._legacy_measure(ids)
                    save(variant+'-legacy', {'legacy_likelihoods':legacy,
                         'persistent_minus_legacy':delta(measured,legacy)})
            with torch.inference_mode():
                generated=teacher.model.generate(**checkpoint.batch,max_new_tokens=runtime['max_new_tokens'],
                    do_sample=False,num_beams=1,eos_token_id=runtime['eos_ids'],repetition_penalty=1.05)
            response=generated[0,checkpoint.batch['input_ids'].shape[1]:].tolist()
            eos=response.pop() if response and response[-1] in runtime['eos_ids'] else None
            save(variant+'-generation', {'generated_ids':response,'terminal_eos':eos,
                 'text':teacher.processor.decode(response,skip_special_tokens=True),
                 'matches_historical':response==case['reader']['expected_generated_ids'] and eos==case['reader']['expected_terminal_eos'],
                 'peak_gpu_bytes':torch.cuda.max_memory_allocated()})
            checkpoint.close()
            checkpoint=None
        comparison={name:{'efficient_minus_math_scores':delta(results['efficient']['masks'][name]['G_S_C'],results['segmented_math']['masks'][name]['G_S_C']),
                    'efficient_minus_math_effects':delta(results['efficient']['masks'][name]['delta_G_S_C'],results['segmented_math']['masks'][name]['delta_G_S_C'])}
                    for name in results['efficient']['masks']}
        ranks={variant:{metric:sorted(values['masks'],key=lambda name:values['masks'][name]['G_S_C'][i],reverse=True)
                        for i,metric in enumerate(('G','S','C'))} for variant,values in results.items()}
        save('comparison',{'comparisons':comparison,'rankings':ranks,'complete':True,
             'limitation':'One question, four fixed masks; segmented math is not a bitwise replay of the historical concatenated kernel.'})
    except Exception as exc:
        save('error',{'error_type':type(exc).__name__,'message':str(exc)})
        raise
    finally:
        attention.segmented_sdpa=efficient
        if checkpoint is not None:
            checkpoint.close()

if __name__=='__main__':
    main()
