# Query-Relevant Document Token Pruning

This repository is a research skeleton for token compression in document QA.
The goal is to reduce the number of visual/document tokens sent to an
expensive multimodal answerer by pruning content that is irrelevant to the
specific query while preserving all information required to answer correctly.

The central target is:

\[
P(\text{token/region needed for answer}\mid \text{document}, q)
\]

The central research question is where query relevance should be estimated.
No model architecture, dataset, experiment, or cluster job is active yet.

## Start here

- [Agent context](agent-context/INDEX.md)
- [Repository navigation](docs/NAVIGATION.md)
- [Source inventory](docs/SOURCE_INVENTORY.md)
- [Research references](references/README.md)
- [Inherited specifications](docs/specifications/README.md)
- [SOL operating guidance](docs/SOL_INSTRUCTIONS.md)
- [SBATCH examples](examples/sbatch/README.md)

Inherited research and historical cluster material are retained for reference.
They do not authorize implementation or execution in this repository.

