"""CPU contract tests for correction depth orchestration."""
import tempfile
from pathlib import Path
from docprune.correction_depth import (validate_case, boundaries, likelihood_effect,
    baseline_stratum, publish, resume, digest, resource)


def raises(fn):
    try:
        fn()
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_complete_answer_target_and_aliases():
    case = {"question": "Which two?", "pages": [{}]*4, "gold_answers": ['["A", "B"]'],
            "answer_contract": {"kind": "set", "items": [{"canonical": "A"}, {"canonical": "B"}]}}
    validate_case(case)
    case["gold_answers"] = ["A", "B"]
    raises(lambda: validate_case(case))


def test_input_is_distinct_from_after_first_block():
    assert boundaries(["input", "0", "dynamic", "14"], 14) == ["input", 0, 14]
    raises(lambda: boundaries(["dynamic"], None))
    raises(lambda: boundaries(["-1"], 14))


def test_margin_gain_does_not_imply_gold_support_gain():
    effect = likelihood_effect([-4, -8], [-3, -5])
    assert effect["delta_G"] == -1
    assert effect["delta_S"] == -3
    assert effect["delta_G_minus_S"] == 2


def test_baseline_adjudication_binds_generation():
    case = {"case_id": "q", "answer_contract": {"kind": "scalar", "items": [{"canonical": "A"}], "rejected_complete": ["B"]}}
    baseline = {"artifact_sha256": "hash", "unpruned": {"answer": "C"}}
    assert baseline_stratum(case, baseline)["status"] == "review"
    review = {"q": {"baseline_sha256": "hash", "status": "incorrect", "reviewer": "reviewer", "rationale": "different entity on supplied page"}}
    assert baseline_stratum(case, baseline, review)["status"] == "incorrect"
    review["q"]["baseline_sha256"] = "other-generation"
    raises(lambda: baseline_stratum(case, baseline, review))


def test_resume_rejects_input_drift_and_modified_output():
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/"result.json"
        publish(path, {"identity": {"input": "one"}, "value": 3})
        assert resume(path, {"input": "one"})["value"] == 3
        raises(lambda: resume(path, {"input": "two"}))
        path.write_text(path.read_text().replace('"value": 3', '"value": 4'))
        raises(lambda: resume(path, {"input": "one"}))


def test_depth_masks_and_achieved_token_budgets_match():
    from docprune.task9_attribution import (build_region_mask_design,
        build_task9_preliminary_arm_selections)
    regions = [{"source_id": "r0", "token_cost": 3},
               {"source_id": "r1", "token_cost": 5},
               {"source_id": "r2", "token_cost": 7}]
    designs = [build_region_mask_design(regions, question_id="q", forced_boundary=boundary,
        mapping_artifact_sha256="a"*64, prompt_input_sha256="b"*64,
        target_kind="max-accepted-reference-mean-loglikelihood",
        reference_set_token_ids_sha256="c"*64, generated_response_token_ids_sha256=None,
        fit_mask_count=256, holdout_mask_count=0) for boundary in ["B_input", "B_0", "B_14"]]
    assert len({digest(x["fit_masks"]) for x in designs}) == 1
    assert len({x["attribution_identity_sha256"] for x in designs}) == 3
    selected = build_task9_preliminary_arm_selections(regions, question_id="q",
        requested_token_count=9, gold_support_scores={"r0": 2, "r1": -1, "r2": 5},
        gold_margin_scores={"r0": -1, "r1": 5, "r2": 2})
    assert [x["achieved_token_count"] for x in selected["budgets"][0]["arms"]] == [8,8,8]


def test_mock_model_baseline_and_two_depths_publish_and_resume():
    """Exercise orchestration with real output dataclasses/masks/fit adapter/knapsack.

    Model execution, fixture loading, legacy scoring and the unavailable
    sklearn numerical kernel are replaced. This is an API/serialization smoke, not evidence of GPU parity.
    """
    import json
    import hashlib
    from types import SimpleNamespace, ModuleType
    from unittest.mock import patch
    import docprune.correction_depth as runtime
    from docprune.answerers import RegionalLikelihoodOutput
    from docprune.m3docrag import AnswerOutput
    from docprune.qwen2vl.model import (PruningTrace, SharedBoundaryLikelihoodResult,
                                     ForcedInterventionLikelihoodBranch)
    from docprune.segmentation import RegionTokenMapping
    from docprune.correction_scoring import contract_digest

    calls = []
    population = 15
    trace = lambda retained, boundary=None: PruningTrace(20, 18, population, retained, boundary)
    values = lambda count: (-3.0 + .1*(population-count), -2.0-.05*(population-count))
    policy = SimpleNamespace(to_dict=lambda: {'achieved_budget': 9, 'native_layer': 14,
        'boundary': 'B_14', 'visual_population': population})

    class FakeAnswerer:
        processor = SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda *a, **kw: 'wrong'))
        def __init__(self, targets=None, forced=None, native=False):
            self.targets, self.forced, self.native = targets, forced, native
        def answer(self, images, question, *, retrieval_output):
            count = 9 if self.native else population if self.forced is None else len(self.forced.retained_visual_ids)
            boundary = 14 if self.native else None if self.forced is None or self.forced.boundary == 'input' else self.forced.boundary
            forced_record = None if self.forced is None else SimpleNamespace(to_dict=lambda: {
                'boundary': self.forced.boundary, 'retained_visual_ids': list(self.forced.retained_visual_ids)})
            calls.append(('generation', count, boundary))
            return AnswerOutput(answer='right' if count < population and not self.native else 'wrong',
                trace=trace(count, boundary), qa_seconds=.1, encoder_seconds=.02, decoder_seconds=.08,
                forced_intervention=forced_record, policy_selection=policy if self.native else None,
                teacher_forced_loglikelihoods=values(count) if self.targets else None)
        def score_forced_intervention_likelihoods(self, images, question, *, retrieval_output,
                forced_interventions, teacher_forced_target_token_ids, include_unpruned_generated_response):
            assert len(teacher_forced_target_token_ids) == (1 if include_unpruned_generated_response else 2)
            calls.append(('masks', len(forced_interventions), forced_interventions[0].boundary))
            branches = tuple(ForcedInterventionLikelihoodBranch(None, values(len(x.retained_visual_ids)))
                             for x in forced_interventions)
            result = SharedBoundaryLikelihoodResult(boundary=str(forced_interventions[0].boundary),
                original_visual_tokens=20, post_btp_visual_tokens=18, post_qtp_visual_tokens=population,
                checkpoint_cache_lengths=(30,), query_aggregate_attention_scores=(0.,)*population,
                branches=branches, encoder_seconds=.2, prefix_decoder_seconds=.3,
                branch_decoder_seconds=(.1,)*len(branches))
            return RegionalLikelihoodOutput(result=result, assistant_prompt_sha256='a'*64,
                prefill_input_ids_shape=(1,30), prefill_input_ids_sha256='b'*64, peak_allocated_gpu_bytes=100,
                unpruned_generated_response_token_ids=(7,) if include_unpruned_generated_response else None,
                unpruned_terminal_eos_token_id=151645 if include_unpruned_generated_response else None)

    sources = (SimpleNamespace(source_id='r0',token_ids=tuple(range(3))),
               SimpleNamespace(source_id='r1',token_ids=tuple(range(3,8))),
               SimpleNamespace(source_id='r2',token_ids=tuple(range(8,15))))
    mapping = RegionTokenMapping(artifacts=(SimpleNamespace(pages=(SimpleNamespace(fixture_question_id='q'),)),),
        geometry=(None,)*population, geometry_count=population, geometry_sha256='c'*64,
        residual_grid_size=2, assignment_contract='test', audited_regions=(), empty_region_source_ids=(),
        sources=sources, token_to_source=('r0',)*3+('r1',)*5+('r2',)*7, sha256='d'*64)
    contract = {'kind':'scalar','items':[{'canonical':'right'}], 'rejected_complete':['wrong']}
    case = {'qid':'q','case_id':'q-v1','question':'Which?', 'pages':[
        {'doc_id':'doc','page_index':i,'score':1.0} for i in range(4)],
        'gold_answers':['right'], 'original_answers':['right'], 'answer_contract':contract,
        'contract_sha256':contract_digest(contract), 'protected_overlap_clear':True,
        'evidence_review':{'rationale':'fixture visual review'}}
    evaluation = ModuleType('docprune.evaluation')
    evaluation.list_f1 = lambda answer, references: float(answer in references)
    evaluation.list_em = lambda answer, references: answer in references
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        artifact = root/'mapping.json'; artifact.write_text('{}')
        resources = {'fixture':{'path':'fixture','sha256':'e'*64},
                     'mappings':{'q':{'path':str(artifact),'sha256':hashlib.sha256(artifact.read_bytes()).hexdigest()}}}
        with patch.dict('sys.modules', {'docprune.evaluation':evaluation}), \
             patch.object(runtime, '_workload', return_value=(FakeAnswerer(), [], object())), \
             patch.object(runtime, '_configured', side_effect=lambda base, **kw: FakeAnswerer(
                 targets=kw.get('targets'),forced=kw.get('forced'),native=kw.get('native',False))), \
             patch('docprune.answerers.prepare_task7_likelihood_target_for_question', return_value={'target_token_ids':[[9]]}), \
             patch('docprune.segmentation.load_region_mapping', return_value=mapping), \
             patch('docprune.task9_attribution._fit_contextcite_solver', side_effect=lambda matrix, targets:
                 (__import__('numpy').ones(matrix.shape[1]), float(__import__('numpy').mean(targets)))) as solver:
            baseline = runtime.run_baseline(case,resources,root,root/'results','f'*40)
            assert baseline['fixed_self_token_ids'] == [7]
            assert baseline['unpruned']['contract_score']['status'] == 'incorrect'
            assert json.loads((root/'results/q-v1/reference.jsonl').read_text())['trace']['post_qtp_visual_tokens'] == 15
            results = runtime.run_comparison(case,resources,root,root/'results','f'*40)
            assert [r['boundary'] for r in results] == ['B_input','B_14']
            assert all(len(r['raw_likelihoods']) == 256 for r in results)
            assert all(len(row)==2 for r in results for row in r['raw_likelihoods'])
            assert all(x['genuine_correction'] for r in results for x in r['selected_results'])
            assert all(x['trace']['post_ctp_visual_tokens']==8 for r in results for x in r['selected_results'])
            assert all(len(r['scoring_cost']['branch_decoder_seconds'])==257 for r in results)
            assert solver.call_count == 6  # G, S, and direct G-S at each depth.
            assert sum(call[0] == 'masks' for call in calls) == 3  # baseline parity + two mask banks.
            for result in results:
                fits = result['surrogates']
                assert len({fits[key]['fit']['fit_masks_sha256'] for key in
                            ['gold_support','fixed_self_support','gold_margin']}) == 1
                assert fits['gold_support']['fit']['attribution_identity_sha256'] != fits['fixed_self_support']['fit']['attribution_identity_sha256']
                assert fits['coefficient_residual']['coefficients'] == {'r0':0.,'r1':0.,'r2':0.}
                assert fits['gold_margin']['fit']['coefficients'] == {'r0':1.,'r1':1.,'r2':1.}
                assert len(result['selected_results']) == 5
                context = result['selected_context_comparison']
                assert set(context['tokens']['both']) | set(context['tokens']['gold_only']) == set(next(
                    x['retained_visual_ids'] for x in result['selected_results'] if x['arm']=='contextcite_gold_support'))
            summary = runtime.summarize([case], root/'results')
            assert len(summary['counts']) == 11  # five arms at two depths + native comparator.
            assert all(row['n_verified_incorrect'] == 1 for row in summary['counts'])
            native_count = next(row for row in summary['counts'] if row['arm']=='native_docprune')
            assert native_count['n_genuine_corrections'] == 0
            assert native_count['verified_correction_fraction'] == 0.
            before = len(calls)
            runtime.run_baseline(case,resources,root,root/'results','f'*40)
            runtime.run_comparison(case,resources,root,root/'results','f'*40)
            assert len(calls) == before
            # Missing final summary does not force another completed mask sweep.
            (root/'results/q-v1/B_input/comparison.json').unlink()
            runtime.run_comparison(case,resources,root,root/'results','f'*40)
            assert len(calls) == before
