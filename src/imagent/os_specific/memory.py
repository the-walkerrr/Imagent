"""
os_specific/memory.py — How much memory a program can use right now, as each operating system tells it.

Linux counts what the kernel says is available (``MemAvailable``), and, when the program runs in a container or under a
memory limit (a cgroup), also what is left of that limit: ``/proc/meminfo`` shows the whole computer, so without this a
program in a 2 GB container would count on the free memory of a 64 GB host. The smaller of the two is used.
"""

from __future__ import annotations

import ctypes
import platform
import re
import subprocess
from pathlib import Path

_CGROUP_ROOT = Path("/sys/fs/cgroup")
_PROC_CGROUP = Path("/proc/self/cgroup")
_NO_LIMIT = 2 ** 60  # cgroup v1 writes "no limit" as a number a little below 2**63; v2 writes "max"


def _parse_meminfo(text: str) -> int | None:
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    return None


def _parse_vm_stat(text: str) -> int | None:
    size = re.search(r"page size of (\d+) bytes", text)
    pages = {name: int(count) for name, count in re.findall(r"Pages (free|inactive|speculative):\s+(\d+)", text)}
    if not size or len(pages) < 3:
        return None
    return (pages["free"] + pages["inactive"] + pages["speculative"]) * int(size.group(1))


class _MemoryStatus(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong), ("ullTotalPhys", ctypes.c_ulonglong),
                ("ullAvailPhys", ctypes.c_ulonglong), ("ullTotalPageFile", ctypes.c_ulonglong),
                ("ullAvailPageFile", ctypes.c_ulonglong), ("ullTotalVirtual", ctypes.c_ulonglong),
                ("ullAvailVirtual", ctypes.c_ulonglong), ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def _read_int(path: Path) -> int | None:
    """The number in a cgroup file, or None if the file is missing or holds something else (v2 writes ``max`` for no limit)."""
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _read_stat(path: Path, key: str) -> int:
    """The value of ``key`` in a cgroup ``memory.stat`` file, or 0."""
    try:
        for line in path.read_text().splitlines():
            name, _, value = line.partition(" ")
            if name == key:
                return int(value)
    except (OSError, ValueError):
        pass
    return 0


def _cgroup_available(root: Path = _CGROUP_ROOT, proc_cgroup: Path = _PROC_CGROUP) -> int | None:
    """Bytes left under the tightest memory limit that applies to this process, or None when no limit is set or it cannot be read.

    The limit may sit on the process's own cgroup or on any cgroup above it (a container, a systemd slice), so each is
    looked at, up to the top. What is used leaves out the cache the kernel can take back (``inactive_file``), the way
    ``docker stats`` counts it. Both cgroup versions are read: v2 (``memory.max``) and v1 (``memory.limit_in_bytes``)."""
    try:
        lines = [line.split(":", 2) for line in proc_cgroup.read_text().splitlines()]
    except OSError:
        return None
    if (root / "cgroup.controllers").exists():  # v2: one tree for every controller
        base = root
        own = next((parts[2] for parts in lines if len(parts) == 3 and parts[0] == "0"), "/")
        limit_file, used_file, stat_key = "memory.max", "memory.current", "inactive_file"
    else:  # v1: a tree for each controller
        base = root / "memory"
        own = next((parts[2] for parts in lines if len(parts) == 3 and "memory" in parts[1].split(",")), "/")
        limit_file, used_file, stat_key = "memory.limit_in_bytes", "memory.usage_in_bytes", "total_inactive_file"
    folder = base / own.strip().lstrip("/")
    if not folder.is_dir():  # inside a container the cgroup is often mounted as the top of the tree
        folder = base
    left = None
    while True:
        limit = _read_int(folder / limit_file)
        used = _read_int(folder / used_file)
        if limit is not None and used is not None and limit < _NO_LIMIT:
            free = max(limit - max(used - _read_stat(folder / "memory.stat", stat_key), 0), 0)
            left = free if left is None else min(left, free)
        if folder == base or base not in folder.parents:
            return left
        folder = folder.parent


def available_memory() -> int | None:
    """Bytes of memory free for a program to use right now, or None if this computer will not say."""
    try:
        system = platform.system()
        if system == "Linux":
            counts = [count for count in (_parse_meminfo(Path("/proc/meminfo").read_text()), _cgroup_available()) if count is not None]
            return min(counts) if counts else None
        if system == "Darwin":
            return _parse_vm_stat(subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=5).stdout)
        if system == "Windows":
            status = _MemoryStatus()
            status.dwLength = ctypes.sizeof(status)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
            return int(status.ullAvailPhys)
    except Exception:
        pass
    return None
