# DocPrune

DocPrune is a training-free document-token pruning system for multimodal question answering. This repository contains the implementation, controlled reproduction studies, regional ContextCite experiments, and cluster execution contracts used in the project.

## Current status

- The DocPrune implementation and Task 6–9 history are consolidated on `main`.
- The full top-4 M3DocVQA comparison completed on 2,441 questions.
- The controlled 245-question stage study localized the clearest quality loss to CTP.
- The 48-question regional ContextCite development pilot produced four exact rescues among 24 baseline-wrong questions while preserving all 24 baseline-correct questions.
- Reducing the regional surrogate from 256 to 192 fitting masks failed the frozen rescue/F1 gate.
- Two H200 follow-ups are prepared: a locked 100-question baseline-wrong confirmation and a separate 600-question shared-probe study.

These findings concern the local reconstruction and an adapted answer-conditioned, region-level ContextCite diagnostic. They are not evidence of parity with unpublished author code, vanilla ContextCite, or a deployable token oracle.

## Start here

1. [Project results](docs/results/README.md)
2. [Current task](agent-context/CURRENT_TASK.md)
3. [Regional attribution experiments](docs/experiments/regional-attribution/INDEX.md)
4. [Repository map](docs/NAVIGATION.md)
5. [Reproduction guide](docs/reproduction/DOCPRUNE.md)
6. [SOL and H200 operations](sol/README.md)

## Repository layout

```text
agent-context/   Current authority and machine handoffs
configs/         Versioned experiment configuration
docs/
  experiments/   Active scientific plans and immutable ledgers
  results/       Human-facing consolidated findings and artifact index
  reproduction/  Method fidelity, gaps, and benchmark documentation
  history/       Superseded plans and conversation records
examples/        Reusable command-line and scheduler entry points
h200/            H200-specific execution contracts
references/      DocPrune paper record
sol/             SOL policy and historical execution handoffs
src/docprune/    Implementation
tests/           Contract and regression tests
```

Generated datasets, model weights, caches, and run outputs are intentionally outside Git. On SOL, local non-Git material is consolidated under `/home/lmalveau/docprune-data`; large experiment outputs remain under `/scratch/lmalveau/docprune`.

## Local verification

```bash
python -m venv .venv
.venv/bin/pip install -e '.[dev,model]'
.venv/bin/pytest -q
.venv/bin/ruff check .
```

Experiment documents do not authorize a launch by themselves. Follow the current task and the applicable machine handoff.
