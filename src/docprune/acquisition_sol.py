"""SOL scheduler admission for the unchanged acquisition reader."""
import os
from pathlib import Path
import pwd
import socket
import subprocess


def sol_preflight(args, *, check_occupancy=True):
    if args.command != 'smoke':
        raise ValueError('SOL admission is limited to the one-question smoke')
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
    record = subprocess.check_output(['scontrol', 'show', 'job', job, '-o'], text=True)
    fields = dict(word.split('=', 1) for word in record.split() if '=' in word)
    if fields.get('JobState') != 'RUNNING' or fields.get('Partition') == 'lightwork':
        raise ValueError('Smoke requires a running GPU compute job, not lightwork')
    if fields.get('UserId', '').split('(')[0] != scratch.name:
        raise ValueError('Slurm allocation owner mismatch')
    allocated = dict(item.split('=', 1) for item in fields.get('AllocTRES', '').split(',') if '=' in item)
    if allocated.get('gres/gpu') != '1':
        raise ValueError('Request exactly one GPU in the Slurm allocation')
    nodes = subprocess.check_output(['scontrol', 'show', 'hostnames', fields['NodeList']], text=True).split()
    if socket.gethostname().split('.')[0] not in nodes:
        raise ValueError('Process is outside the allocated compute node')
    query = subprocess.check_output(['nvidia-smi', '-i', visible,
        '--query-gpu=uuid,name,memory.used,memory.total,utilization.gpu',
        '--format=csv,noheader,nounits'], text=True).strip()
    if len(query.splitlines()) != 1:
        raise ValueError('Expected one assigned physical GPU')
    uuid, name, used, total, util = [part.strip() for part in query.split(',')]
    processes = subprocess.check_output(['nvidia-smi', '-i', visible,
        '--query-compute-apps=pid', '--format=csv,noheader,nounits'], text=True).strip()
    if processes or float(used) > 512 or float(util) > 5:
        raise RuntimeError('Assigned GPU is occupied; no model loaded')
    if not any(model in name for model in ('H100', 'H200')) or float(total) < 70000:
        raise ValueError('This smoke requires a Hopper GPU with at least 70 GiB')
    return {'platform': 'sol', 'job_id': job, 'node': socket.gethostname(),
            'assigned_device': visible, 'uuid': uuid, 'name': name,
            'memory_used_mb': used, 'memory_total_mb': total, 'utilization_percent': util}
