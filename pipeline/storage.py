"""
pipeline/storage.py — Tell whether a folder is on this computer's own disk.

Imagent works with pictures stored on the computer it runs on. A network drive can disappear in the middle of a run and
an external drive can be unplugged, so folders on them are refused with a clear message instead of failing image by image.

Each system says it differently, and none says it perfectly, so this only refuses what it is sure of: when the answer
cannot be found the folder is allowed, and the rest of the program copes with a drive that goes away (see config.MAX_CONSECUTIVE_SERVICE_FAILURES).
Not checked: a USB hard drive on Windows, which Windows reports as a fixed disk.
"""

from __future__ import annotations

import ctypes
import os
import platform
import plistlib
import re
import subprocess
from pathlib import Path

NETWORK = "network"
EXTERNAL = "external"

NETWORK_FILESYSTEMS = frozenset({
    "nfs", "nfs4", "cifs", "smb", "smb2", "smb3", "smbfs", "afpfs", "afs", "webdav", "davfs", "9p", "ceph", "glusterfs",
    "lustre", "ncpfs", "sshfs", "fuse.sshfs", "fuse.rclone",
})


def _parse_mount_output(text: str) -> list[tuple[str, str, str]]:
    """The ``(mount point, file system, source)`` of each line of macOS's ``mount``: ``source on /mount/point (fstype, options)``."""
    found = []
    for line in text.splitlines():
        match = re.match(r"^(.+?) on (/.*?) \(([^,)]+)", line)
        if match:
            found.append((match.group(2), match.group(3), match.group(1)))
    return found


def _parse_mountinfo(text: str) -> list[tuple[str, str, str]]:
    """The ``(mount point, file system, source)`` of each line of Linux's ``/proc/self/mountinfo``."""
    found = []
    for line in text.splitlines():
        if " - " not in line:
            continue
        left, right = line.split(" - ", 1)
        fields, rest = left.split(), right.split()
        if len(fields) >= 5 and len(rest) >= 2:
            mount_point = re.sub(r"\\([0-7]{3})", lambda m: chr(int(m.group(1), 8)), fields[4])
            found.append((mount_point, rest[0], rest[1]))
    return found


def _owner(path: str, mounts: list[tuple[str, str, str]]) -> tuple[str, str, str] | None:
    """The mount that holds ``path``: the one whose mount point is the longest start of it."""
    best = None
    for mount in mounts:
        point = mount[0].rstrip(os.sep) or os.sep
        if path == point or path.startswith(point.rstrip(os.sep) + os.sep):
            if best is None or len(point) > len(best[0].rstrip(os.sep) or os.sep):
                best = mount
    return best


def _macos_is_external(mount_point: str) -> bool:
    info = plistlib.loads(subprocess.run(["diskutil", "info", "-plist", mount_point], capture_output=True, timeout=20).stdout)
    return info.get("Internal") is False


def _linux_is_external(source: str) -> bool:
    """True if the device behind ``source`` hangs off a USB port or is marked removable."""
    name = os.path.basename(os.path.realpath(source))
    real = os.path.realpath(f"/sys/class/block/{name}")
    if "/usb" in real:
        return True
    disk = os.path.basename(os.path.dirname(real)) if os.path.exists(os.path.join(real, "partition")) else name
    try:
        return Path(f"/sys/block/{disk}/removable").read_text().strip() == "1"
    except OSError:
        return False


def _classify(path: str) -> str | None:
    """``NETWORK``, ``EXTERNAL`` or None (the computer's own disk, or not known) for a folder."""
    system = platform.system()
    if system == "Windows":
        if path.startswith("\\\\"):
            return NETWORK
        kind = ctypes.windll.kernel32.GetDriveTypeW(os.path.splitdrive(path)[0] + "\\")  # 2 removable, 4 network, 5 optical
        return {4: NETWORK, 2: EXTERNAL, 5: EXTERNAL}.get(kind)
    if system == "Darwin":
        mounts = _parse_mount_output(subprocess.run(["mount"], capture_output=True, text=True, timeout=10).stdout)
    elif system == "Linux":
        mounts = _parse_mountinfo(Path("/proc/self/mountinfo").read_text())
    else:
        return None
    owner = _owner(path, mounts)
    if owner is None:
        return None
    mount_point, file_system, source = owner
    if file_system.lower() in NETWORK_FILESYSTEMS:
        return NETWORK
    if system == "Darwin":
        return EXTERNAL if _macos_is_external(mount_point) else None
    return EXTERNAL if source.startswith("/dev/") and _linux_is_external(source) else None


def storage_problem(path: str | Path) -> str | None:
    """A sentence saying why the folder cannot be used, or None if it is on this computer's own disk or that cannot be told."""
    try:
        kind = _classify(os.path.realpath(path))
    except Exception:
        return None
    if kind is None:
        return None
    where = "a network drive" if kind == NETWORK else "an external or removable drive"
    return (f"{path} is on {where}. Imagent only works with pictures stored on this computer's own disk, "
            "because a drive like that can disappear in the middle of a run. Copy the pictures to this computer, then scan that folder.")
