# SOL one-question adaptive acquisition smoke

Owner authorized on 2026-09-14: run the existing Q12 smoke on SOL instead of the
busy shared H200 server. This authorizes setup, sealed-input transfer, the exact
pinned model download, CPU checks and one GPU smoke. No 17-question scoring run.

Use /home/lmalveau/DocPrune, main, at the commit recorded in the submitted
ACQUISITION_CODE_COMMIT. The batch launcher rejects checkout drift. Read
../../docs/SOL_INSTRUCTIONS.md and ../AGENTS.md. Reuse the current browser
lightwork allocation for setup; no additional lightwork allocation is needed.

All inputs, downloads, caches, temporary files, logs and outputs go under
/scratch/lmalveau/docprune-adaptive-acquisition/20260914-smoke01.
Source environment.sh for exact paths. The environment is a separate Python
3.10 venv at /home/lmalveau/mamba-envs/docprune-acquisition-sol, reusing the
existing docprune-sol packages through system-site-packages with the pinned
Transformers 4.49.0 installed only in the new overlay. Never alter docprune-sol.
Validate all six frozen versions and reader imports before GPU submission.

Transfer only the 31 MB sealed stage0-adaptive-acquisition-v1 input package from
H200 using the private runbook's rsync route, into this scratch root's inputs/.
No historical pilot receipt or old 600-question data is needed. Validate the
sealed package before use. Preserve byte-identical runtime.json; override only
the local model path using --snapshot. Download Qwen/Qwen2.5-VL-7B-Instruct at
cc594898137f460bfe9f0759e9844b3ce807cfb5 into the task HF cache on scratch.
Model execution is offline. No fresh retrieval, rerendering, or feature extraction.

Submit smoke.sbatch: one compatible GPU (native BF16, >=23000 MiB VRAM), two CPUs, 24000 MiB RAM, 20 minutes
on htc/public. Slurm owns GPU assignment; never use CoRAL IDs on SOL. The runner
checks allocation ownership, node membership, a single assigned idle GPU and
scratch paths. Preserve all scientific settings, G/S targets, 50% token mask,
Q12 identity, SDPA/BF16, original answer identity and 1e-4 parity tolerances.

A failure stops the smoke; preserve logs. Narrow setup/path/API fixes that keep
this contract may be committed and retried in a new job output directory. Do
not relax parity, change the checkpoint or expand to full scoring. A passed
SOL smoke is hardware-specific integration evidence, not an H200 validation or
a comparison of the three acquisition policies. Return the log and receipt.

Owner resource update: use any compatible GPU and the admitted 24000 MiB host
RAM minimum. This is a bounded memory-fit trial; actual peak RAM/VRAM will be
measured. Never relax scientific parity checks for a different GPU.
