# Implementation and execution readiness

These are unresolved choices in the supplied design, not permission requests for
the completed cleanup. The bounded Colfeatures17 task now freezes its own retriever, inputs and runtime;
the choices below still apply to later learning experiments.

1. **Models and feature bridge:** choose exact reader, selector, retriever and revisions;
   validate the shared-vision shape/layer/DeepStack contract and adapter versus native
   vision diagnostic. Candidate Qwen3 models are not supported merely because the
   retained Qwen2 reproduction imports successfully.
2. **Runtime:** establish a separate compatible environment before changing baseline
   Python/torch/transformers pins. Current `pyproject.toml` model extras pin 4.46.3.
3. **Data:** source versions, exposure/near-duplicate audit, document-family splits,
   train-only retrieval index, held-out transfer allocation, and immutable manifests.
4. **Intervention:** all-kept parity, exact insertion boundary, atomic region coverage,
   fallback size and cost accounting, shared-feature provenance and sparse decoding.
5. **Teacher:** accepted answers/tokenization, G/S scoring, epsilon/ties, bank composition,
   seed policy, proposal budgets and exact attainable token-budget allocation.
6. **Evaluation:** retrieval pool/admitted pages/budget grid, admissibility population,
   rescue/harm criteria, repeated random controls, document-level uncertainty and cost.
7. **Operations:** hardware, checkout/commit, environment, input hashes, scratch/output
   roots, resources, exact commands and recovery plan. The Colfeatures17 handoff is active; subsequent stage handoffs remain unset.
8. **Plan wording:** section 2.2's generic conditional language for richer readouts
   should be reconciled with explicit sections 9.3 and 10.4. Preparation follows
   those explicit sections: 2B-Rich is a primary early arm. Original text is unchanged.

Stage 0 should resolve evidence and interface questions before broad implementation.
Model-specific labels cannot be inherited from older reader checkpoints unchanged.
