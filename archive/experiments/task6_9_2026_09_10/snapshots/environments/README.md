# Environments

[`docprune-sol.yml`](docprune-sol.yml) is the pinned GPU environment
specification for the reproduction and includes Poppler 26.05.0. The active
M3DocVQA handoff uses the existing acquisition environment
`/home/lmalveau/mamba-envs/m3docvqa-acquisition` for PDF rendering because
its Poppler executables are separately hash- and version-pinned.

Inherited environment files are reference-only examples under
[`../examples/environments/legacy/`](../examples/environments/legacy/).
