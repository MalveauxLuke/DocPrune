"""SOL scheduler admission for the unchanged acquisition reader."""
import os
from pathlib import Path
import pwd
import socket
import subprocess


def sol_preflight(args, *, check_occupancy=True):
    if args.command not in ('smoke', 'score'):
        raise ValueError('SOL admission requires an acquisition smoke or score job')
    if args.physical_gpu is not None:
        raise ValueError('SOL uses the Slurm-assigned device; do not override it')
    job = os.environ.get('SLURM_JOB_ID', '')
    visible = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not job.isdigit() or not visible or ',' in visible or visible == '-1':
        raise ValueError('A Slurm job with exactly one visible GPU is required')
    scratch = Path('/scratch') / pwd.getpwuid(os.getuid()).pw_name
    for path in (args.package, args.output):
        if not Path(path).resolve().is_relative_to(scratch):
            raise ValueError('SOL input/output paths must stay in user scratch')
    for name in ('HF_HOME', 'TRANSFORMERS_CACHE', 'PIP_CACHE_DIR', 'UV_CACHE_DIR',
                 'TORCH_HOME', 'XDG_CACHE_HOME', 'CONDA_PKGS_DIRS', 'TMPDIR'):
        if not os.environ.get(name) or not Path(os.environ[name]).resolve().is_relative_to(scratch):
            raise ValueError(f'{name} must be pinned to SOL user scratch')
    if not check_occupancy:
        return None
    # The placeholder shard can have SLURM_JOB_ID equal to the array ID.
    # Querying that number returns multiple sibling records, not one allocation.
    array_job = os.environ.get('SLURM_ARRAY_JOB_ID')
    array_task = os.environ.get('SLURM_ARRAY_TASK_ID')
    selector = job
    if array_job is not None or array_task is not None:
        if not (array_job and array_job.isdigit() and array_task and array_task.isdigit()):
            raise ValueError('Invalid Slurm array identity')
        selector = f'{array_job}_{array_task}'
    record = subprocess.check_output(['scontrol', 'show', 'job', selector, '-o'], text=True)
    rows = [line for line in record.splitlines() if line.strip()]
    if len(rows) != 1:
        raise ValueError(f'Expected exactly one Slurm record for {selector}, received {len(rows)}')
    fields = dict(word.split('=', 1) for word in rows[0].split() if '=' in word)
    if array_job is not None and (fields.get('ArrayJobId') != array_job or fields.get('ArrayTaskId') != array_task):
        raise ValueError('Slurm returned a different array task')
    if fields.get('JobState') != 'RUNNING' or fields.get('Partition') == 'lightwork':
        raise ValueError(f'GPU compute allocation required: selector={selector}, state={fields.get("JobState")}, partition={fields.get("Partition")}')
    if fields.get('UserId', '').split('(')[0] != scratch.name:
        raise ValueError('Slurm allocation owner mismatch')
    allocated = dict(item.split('=', 1) for item in fields.get('AllocTRES', '').split(',') if '=' in item)
    if allocated.get('gres/gpu') != '1':
        raise ValueError('Request exactly one GPU in the Slurm allocation')
    nodes = subprocess.check_output(['scontrol', 'show', 'hostnames', fields['NodeList']], text=True).split()
    if socket.gethostname().split('.')[0] not in nodes:
        raise ValueError('Process is outside the allocated compute node')
    # Resolve CUDA ordinal zero through the loaded CUDA runtime. Slurm/cgroups
    # may renumber ordinals; nvidia-smi numeric indexes need not match them.
    import torch
    if torch.cuda.device_count() != 1 or torch.cuda.get_device_capability(0)[0] < 8:
        raise ValueError('The assigned GPU must support native BF16 (compute capability 8+)')
    capacity = torch.cuda.get_device_properties(0).total_memory / (1024**2)
    if capacity < 23000:
        raise ValueError(f'Assigned CUDA device has {capacity:.0f} MiB; need at least 23000 MiB')
    if visible.startswith('MIG-') or any('.20gb' in key or '.35gb' in key or '.71gb' in key for key in allocated):
        raise ValueError('This smoke requires a whole GPU; MIG allocation is not admitted')
    pci = cuda_pci_bus_id()
    query = subprocess.check_output(['nvidia-smi', '-i', pci,
        '--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True).strip()
    if len(query.splitlines()) != 1:
        raise ValueError('Expected one assigned physical GPU')
    uuid, name, used, total, util = [part.strip() for part in query.split(',')]
    if abs(float(total) - capacity) > 1024:
        raise ValueError('CUDA and physical GPU capacities disagree; refusing ambiguous device')
    processes = subprocess.check_output(['nvidia-smi', '-i', uuid,
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    foreign = [pid.strip() for pid in processes.splitlines() if pid.strip() != str(os.getpid())]
    if foreign:
        raise RuntimeError(f'Assigned GPU {uuid} at {pci} has other process IDs {foreign}; no model loaded')
    # CUDA initialization can itself consume memory and cause transient activity.
    # Do not mistake our own context or a sampled utilization value for another job.
    return {'platform': 'sol', 'job_id': job, 'job_selector': selector, 'node': socket.gethostname(),
            'assigned_device': visible, 'pci_bus_id': pci, 'uuid': uuid, 'name': name,
            'memory_used_mb': used, 'memory_total_mb': total, 'utilization_percent': util}


def cuda_pci_bus_id():
    """Ask the already-loaded CUDA runtime for visible ordinal zero's PCI ID."""
    import ctypes
    paths = {line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines()
             if '/libcudart.so' in line}
    if len(paths) != 1:
        raise RuntimeError(f'Expected one loaded CUDA runtime, found {len(paths)}')
    runtime = ctypes.CDLL(paths.pop())
    query = runtime.cudaDeviceGetPCIBusId
    query.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_int]
    query.restype = ctypes.c_int
    buffer = ctypes.create_string_buffer(32)
    status = query(buffer, len(buffer), 0)
    if status:
        raise RuntimeError(f'CUDA PCI identity query failed with status {status}')
    return buffer.value.decode('ascii')
