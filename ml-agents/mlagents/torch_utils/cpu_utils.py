from typing import Optional
from mlagents_envs.logging_util import get_logger

import os

logger = get_logger(__name__)

def get_num_threads_to_use() -> Optional[int]:
    """
    Gets the number of threads to use. For most problems, 4 is all you
    need, but for smaller machines, we'd like to scale to less than that.
    """
    num_cpus = _get_num_available_cpus()
    print(f"CHANGED PYTOCH AVAILABLE CPUS TO: {num_cpus}")
    return num_cpus


def _get_num_available_cpus() -> Optional[int]:
    """
    Returns number of CPUs using cgroups if possible. This accounts
    for Docker containers that are limited in cores.
    """
    period = _read_in_integer_file("/sys/fs/cgroup/cpu/cpu.cfs_period_us")
    quota = _read_in_integer_file("/sys/fs/cgroup/cpu/cpu.cfs_quota_us")
    share = _read_in_integer_file("/sys/fs/cgroup/cpu/cpu.shares")
    is_kubernetes = os.getenv("KUBERNETES_SERVICE_HOST") is not None

    if period > 0 and quota > 0:
        return int(quota // period)
    elif period > 0 and share > 0 and is_kubernetes:
        # In kubernetes, each requested CPU is 1024 CPU shares
        # https://kubernetes.io/docs/concepts/configuration/manage-resources-containers/#how-pods-with-resource-limits-are-run
        return int(share // 1024)
    else:
        return os.cpu_count()


def _read_in_integer_file(filename: str) -> int:
    try:
        with open(filename) as f:
            return int(f.read().rstrip())
    except FileNotFoundError:
        return -1
