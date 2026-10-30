from __future__ import annotations

import errno
import os
import time
from pathlib import Path


if os.name == "nt":
    import ctypes
    import msvcrt
    from ctypes import wintypes

    class _Overlapped(ctypes.Structure):
        _fields_ = [
            ("Internal", ctypes.c_size_t),
            ("InternalHigh", ctypes.c_size_t),
            ("Offset", wintypes.DWORD),
            ("OffsetHigh", wintypes.DWORD),
            ("hEvent", wintypes.HANDLE),
        ]

    _kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _lock_file_ex = _kernel32.LockFileEx
    _lock_file_ex.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_Overlapped),
    ]
    _lock_file_ex.restype = wintypes.BOOL
    _unlock_file_ex = _kernel32.UnlockFileEx
    _unlock_file_ex.argtypes = [
        wintypes.HANDLE,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(_Overlapped),
    ]
    _unlock_file_ex.restype = wintypes.BOOL

    def _try_lock(fd: int, exclusive: bool) -> bool:
        # Lock one byte of the separate lock file, including when it is empty.
        flags = 0x1 | (0x2 if exclusive else 0)  # FAIL_IMMEDIATELY | EXCLUSIVE_LOCK
        if _lock_file_ex(
            msvcrt.get_osfhandle(fd), flags, 0, 1, 0, ctypes.byref(_Overlapped())
        ):
            return True
        error = ctypes.get_last_error()
        if error == 33:  # ERROR_LOCK_VIOLATION
            return False
        raise ctypes.WinError(error)

    def _unlock(fd: int) -> None:
        if not _unlock_file_ex(
            msvcrt.get_osfhandle(fd), 0, 1, 0, ctypes.byref(_Overlapped())
        ):
            raise ctypes.WinError(ctypes.get_last_error())

else:
    import fcntl

    def _try_lock(fd: int, exclusive: bool) -> bool:
        lock_type = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
        try:
            fcntl.flock(fd, lock_type | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                raise
            return False
        return True

    def _unlock(fd: int) -> None:
        fcntl.flock(fd, fcntl.LOCK_UN)


class FileLock:
    r"""Shared/exclusive file lock with timeout on Unix and Windows."""

    def __init__(self, lock_path: Path, exclusive: bool, timeout: float = 15):
        self.lock_path = lock_path
        self.exclusive = exclusive
        self.timeout = timeout
        self._fd: int = -1

    def __enter__(self) -> "FileLock":
        flags = os.O_WRONLY | os.O_CREAT if self.exclusive else os.O_RDONLY | os.O_CREAT
        self._fd = os.open(str(self.lock_path), flags)
        try:
            deadline = time.monotonic() + self.timeout
            while not _try_lock(self._fd, self.exclusive):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(f"Timed out waiting for lock: {self.lock_path}")
                time.sleep(min(0.1, remaining))
            return self
        except BaseException:
            fd, self._fd = self._fd, -1
            os.close(fd)
            raise

    def __exit__(self, *exc) -> None:
        if self._fd >= 0:
            fd, self._fd = self._fd, -1
            try:
                _unlock(fd)
            finally:
                os.close(fd)


def _flock_held_exclusively(lock_path: str) -> bool:
    r"""Return whether another handle currently holds the lock exclusively."""
    fd = os.open(lock_path, os.O_RDONLY | os.O_CREAT)
    try:
        if not _try_lock(fd, exclusive=False):
            return True
        _unlock(fd)
        return False
    finally:
        os.close(fd)
