## 1. Official Sol rules and constraints

These are the main rules that matter in practice.

### 1.1 Use login nodes only for light, non-computational work

ASU RC defines login nodes as shared nodes for simple, non-computational tasks
and says compute and storage-heavy activity should go to compute nodes. The
Acceptable Use Policy is explicit that running jobs on login nodes is prohibited,
will generate a warning, and may be terminated without advance notice.

What this means operationally:

- do file navigation, editing, small inspections, and job submission on login nodes
- do not run long Python jobs, model inference, training, or heavy compilation there
- request a compute allocation with `salloc` or submit with `sbatch` for anything substantial

Relevant official sources:

- [New User Guide](https://docs.rc.asu.edu/new-user-guide/)
- [Acceptable Use Policy](https://cores.research.asu.edu/research-computing/policies)

### 1.3 Scratch is temporary, shared, and should be actively used

The official docs say:

- `/scratch/[asurite]` is available to all users
- scratch is a temporary, shared resource
- users must actively use files stored there
- inactive files may be removed after 90 days
- scratch is intended for immediate computational use

The policy page also notes:

- scratch has a per-user limit
- scratch has a 20 million file limit per user
- scratch does not provide the same protection guarantees as home storage

Practical implication:

- keep active job outputs and large temporary artifacts on scratch
- do not treat scratch as permanent archival storage
- periodically move important artifacts elsewhere

Relevant official sources:

- [Resource Limits](https://docs.rc.asu.edu/resource-limits/)
- [Acceptable Use Policy](https://cores.research.asu.edu/research-computing/policies)

### 1.4 Home is persistent but smaller

ASU RC documents that:

- `/home` has a quota of 100 GiB
- it is intended for important files and persistent storage
- it is the right place for software installations and package environments
- unlike scratch, it is not a purge-style temporary area

Practical implication:

- keep code, small configs, environments, and important scripts in home
- do not park large experiment artifacts there unless you understand your quota usage

Relevant official sources:

- [Resource Limits](https://docs.rc.asu.edu/resource-limits/)
- [Acceptable Use Policy](https://cores.research.asu.edu/research-computing/policies)

### 1.5 Use the right partition and QoS

The official partition guidance matters a lot.

Key public-facing rules:

- `public` is the general default for most users and can run up to 7 days
- `htc` is for jobs that complete within 4 hours and has scheduling advantages
- `lightwork` is for lighter tasks like building environments, compiling software,
  VSCode tunnels, or bulk file operations
- `lightwork` has a 24-hour limit and a maximum of 8 CPU cores per node
- misuse of `lightwork` can result in cancellation and loss of eligibility
- `private` jobs on private nodes are preemptible

Practical implication:

- use `lightwork` for env creation, lightweight setup, or low-duty-cycle tasks
- use `htc` for short real compute jobs
- use `public` for normal longer-running research jobs
- do not burn full cores at high sustained utilization on `lightwork`

For debugging and smoke runs, prefer the shortest suitable debug-class
partition so the job can avoid normal production queue delays. SOL currently
has no partition literally named `debug`; `htc` is the available equivalent
for jobs that fit its four-hour limit, including short GPU diagnostics. Check
`sinfo` before submission in case the available debug-class partition changes.
Do not move a full production run to a debug-class partition merely to bypass
the queue.

Relevant official source:

- [Partitions and QoS](https://docs.rc.asu.edu/partitions-and-qos/)

### 2.1 Do environment creation and package installation from a compute allocation

ASU RC's Python docs explicitly warn not to install packages on the login nodes
or inside Jupyter notebooks. Their examples use an interactive allocation first.

Good generic pattern:

```bash
ssh <asurite>@sol.asu.edu
```

Then request light setup resources. Two good options are:

For lightweight env creation or setup:

```bash
salloc -p lightwork -q public -t 02:00:00 -c 4
```

For short real compute/setup work:

```bash
salloc -p htc -q public -t 04:00:00 -c 4
```

Why:

- `lightwork` is explicitly documented as a good fit for creating Mamba environments
- `htc` is a good fit when the setup task is short but computationally real

Relevant official sources:

- [Python overview](https://docs.rc.asu.edu/python/)
- [Python Envs and Mamba](https://docs.rc.asu.edu/mamba)
- [Partitions and QoS](https://docs.rc.asu.edu/partitions-and-qos/)

### 2.2 Load Mamba

Once you are in the shell where you want to work:

```bash
module load mamba/latest
```

Or:

```bash
ml mamba
```

This is the official ASU RC path for Python environment management on Sol.

Relevant official source:

- [Python Envs and Mamba](https://docs.rc.asu.edu/mamba)

### 2.3 See what environments already exist

```bash
mamba info --envs
```

This helps confirm:

- whether the environment already exists
- where it lives
- whether you are about to recreate something unnecessarily

Relevant official source:

- [Python Envs and Mamba](https://docs.rc.asu.edu/mamba)

### 2.4 Create a new environment

Generic example in your home directory:

```bash
mamba create -n myenv -c conda-forge python=3.12
```

Then activate it:

```bash
source activate myenv
```

If you need to install packages during creation:

```bash
mamba create -n myenv -c conda-forge python=3.12 numpy pandas
```

If you need a reproducible env from a file:

```bash
mamba env create -n myenv --file environment.yml
```

Relevant official source:

- [Python Envs and Mamba](https://docs.rc.asu.edu/mamba)

### 2.5 Install more packages into the active environment

After activation:

```bash
mamba install -c conda-forge scipy matplotlib
```

If a package only exists via `pip`, activate the env first and then:

```bash
pip install <package>
```

Practical rule:

- prefer `mamba` when possible
- use `pip` inside the active env only when needed

### 2.6 Typical session pattern we have actually used

This is the compact pattern that has worked well in practice:

```bash
ssh <asurite>@sol.asu.edu
salloc -p lightwork -q public -t 02:00:00 -c 4
module load mamba/latest
mamba create -n myenv -c conda-forge python=3.12
source activate myenv
python -V
which python
python -c "import sys; print(sys.executable)"
```

Then install what you need:

```bash
mamba install -c conda-forge <packages>
```

Or:

```bash
pip install <packages>
```

### 2.7 If the environment contains compiled Python packages

If imports later fail with errors mentioning:

- `libstdc++`
- `CXXABI_*`
- compiled extensions

then a useful runtime fix is:

```bash
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$CONDA_PREFIX/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

This is not an official Mamba creation step, but it is a real cross-project fix
we have had to use when compiled dependencies picked up the wrong shared
libraries at runtime.

### 3.7 Python package version mismatch

Common symptoms:

- API attributes missing
- package imports succeed but runtime objects do not match expectations
- one library is too new or too old for another

Typical example class:

- `transformers` version too new for the installed `vllm`

Typical fix:

- pin the package set to a known-compatible range
- record the working versions after the fix

Practical rule:

- for ML stacks on Sol, treat package compatibility as part of the runtime, not an afterthought

## 5. Practical Sol workflow that generalizes well

For a new project, a conservative workflow looks like this:

1. Keep code in `~/some_project`
2. Keep large runtime outputs on `/scratch/$USER/...`
3. Build or activate environments from a light setup context, not a login-node abuse pattern
4. Smoke-test on a small allocation before scaling up
5. Record working package versions once the stack is healthy
6. Explicitly set backend env vars when using complex ML runtimes
7. Move finished artifacts off scratch if they matter long-term


## 7. Recommended generic post-failure checklist

If a Sol job fails, check in this order:

1. Was the failure at submit time, startup time, import time, model-init time, or true runtime?
2. Did Slurm reject the job because of partition/QoS/walltime?
3. Was the correct env active?
4. Is the Sol checkout stale relative to local changes?
5. Is this a compiled-library / `LD_LIBRARY_PATH` issue?
6. Did the runtime silently choose a bad accelerator backend?
7. Did the job fail because a requested setting exceeds what the installed runtime supports?
8. Are the output and log paths actually what you think they are?

## 8. Bottom line

The biggest Sol lessons that generalize across projects are:

- do not compute on login nodes
- use the right partition for the workload
- treat scratch as temporary working storage
- keep environments and compiled-library paths explicit
- assume ML runtime compatibility issues are real until proven otherwise
- smoke-test backend and parameter assumptions on Sol itself
- suspect remote checkout drift early when local and remote behavior disagree
