# Owner(s): ["module: dsl-native-ops"]

import errno
import importlib.util
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


CACHE_ROOT = Path(__file__).resolve().parents[2] / "torch" / "_vendor" / "quack" / "cache"
LOCK_MODULE_PATH = CACHE_ROOT / "_filelock.py"
_spec = importlib.util.spec_from_file_location("quack_filelock", LOCK_MODULE_PATH)
filelock = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(filelock)

LOCK_WORKER = """
import importlib.util
import json
import os
import sys
import time
from pathlib import Path

spec = importlib.util.spec_from_file_location("quack_filelock", sys.argv[1])
filelock = importlib.util.module_from_spec(spec)
spec.loader.exec_module(filelock)
path, exclusive, action, timeout = sys.argv[2:]
lock = filelock.FileLock(Path(path), exclusive == "True", float(timeout))
if action == "try":
    start = time.monotonic()
    acquired = False
    try:
        with lock:
            acquired = True
    except RuntimeError as exc:
        if not str(exc).startswith("Timed out waiting for lock:"):
            raise
    elapsed = time.monotonic() - start
    print(json.dumps({
        "acquired": acquired,
        "writer": filelock._flock_held_exclusively(path),
        "elapsed": elapsed,
        "fd": lock._fd,
    }))
else:
    with lock:
        print("locked", flush=True)
        sys.stdin.readline()
        if action == "exit":
            os._exit(0)
"""

IMPORT_WORKER = """
import importlib.util
import os
import sys
import types
from pathlib import Path

# Stub only the unrelated CuTeDSL dependencies and package initializers.
for name in ("torch", "torch._vendor", "torch._vendor.quack", "torch._vendor.quack.cache",
             "cutlass", "cutlass.cute", "tvm_ffi"):
    module = types.ModuleType(name)
    module.__path__ = [sys.argv[1]] if name == "torch._vendor.quack.cache" else []
    sys.modules[name] = module
    parent, _, child = name.rpartition(".")
    if parent:
        setattr(sys.modules[parent], child, module)

for name in ("_filelock", sys.argv[2]):
    fullname = "torch._vendor.quack.cache." + name
    spec = importlib.util.spec_from_file_location(fullname, Path(sys.argv[1]) / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = module
    spec.loader.exec_module(module)
helper = sys.modules["torch._vendor.quack.cache._filelock"]
attribute = "FileLock" if sys.argv[2] == "jit" else "_flock_held_exclusively"
if getattr(module, attribute) is not getattr(helper, attribute):
    raise AssertionError(f"{name} did not use the shared lock helper")
if os.name == "nt" and "fcntl" in sys.modules:
    raise AssertionError("Windows cache imports must not require fcntl")
"""


@instantiate_parametrized_tests
class TestQuackFileLock(TestCase):
    def setUp(self):
        super().setUp()
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.lock_path = Path(directory.name) / "cache.lock"

    def _worker_command(self, exclusive, action="try", timeout=0):
        return [
            sys.executable,
            "-I",
            "-B",
            "-c",
            LOCK_WORKER,
            str(LOCK_MODULE_PATH),
            str(self.lock_path),
            str(exclusive),
            action,
            str(timeout),
        ]

    def _try_in_subprocess(self, exclusive, timeout=0):
        result = subprocess.run(
            self._worker_command(exclusive, timeout=timeout),
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        return json.loads(result.stdout)

    @parametrize("held_exclusive", [False, True])
    @parametrize("requested_exclusive", [False, True])
    def test_cross_process_locking(self, held_exclusive, requested_exclusive):
        with filelock.FileLock(self.lock_path, held_exclusive):
            result = self._try_in_subprocess(requested_exclusive)
            self.assertEqual(
                result["acquired"], not (held_exclusive or requested_exclusive)
            )
            self.assertEqual(result["writer"], held_exclusive)
            self.assertEqual(result["fd"], -1)
            self.assertLess(result["elapsed"], 5)
        self.assertTrue(self._try_in_subprocess(requested_exclusive)["acquired"])

    @parametrize("exclusive", [False, True])
    def test_timeout(self, exclusive):
        with filelock.FileLock(self.lock_path, exclusive=True):
            result = self._try_in_subprocess(exclusive, timeout=0.1)
            self.assertFalse(result["acquired"])
            self.assertTrue(result["writer"])
            self.assertEqual(result["fd"], -1)
            self.assertGreaterEqual(result["elapsed"], 0.08)
            self.assertLess(result["elapsed"], 5)

    def test_timeout_closes_descriptor(self):
        lock = filelock.FileLock(self.lock_path, exclusive=False, timeout=0)
        with filelock.FileLock(self.lock_path, exclusive=True):
            with mock.patch.object(
                filelock, "_try_lock", wraps=filelock._try_lock
            ) as attempt:
                with self.assertRaisesRegex(RuntimeError, "Timed out waiting for lock"):
                    lock.__enter__()
            fd = attempt.call_args.args[0]
            self.assertEqual(lock._fd, -1)
            with self.assertRaises(OSError) as error:
                os.fstat(fd)
            self.assertEqual(error.exception.errno, errno.EBADF)

    @parametrize("action", ["release", "exit"])
    def test_release_and_process_exit(self, action):
        with subprocess.Popen(
            self._worker_command(True, action=action, timeout=5),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        ) as process:
            self.assertEqual(process.stdout.readline().strip(), "locked")
            self.assertTrue(filelock._flock_held_exclusively(str(self.lock_path)))
            with self.assertRaisesRegex(RuntimeError, "Timed out waiting for lock"):
                with filelock.FileLock(self.lock_path, exclusive=False, timeout=0):
                    self.fail("reader acquired a writer's lock")
            _, stderr = process.communicate(input="\n", timeout=30)
            self.assertEqual(process.returncode, 0, stderr)
        self.assertFalse(filelock._flock_held_exclusively(str(self.lock_path)))
        with filelock.FileLock(self.lock_path, exclusive=True, timeout=0):
            pass

    def test_context_exception_releases_lock(self):
        lock = filelock.FileLock(self.lock_path, exclusive=True)
        with self.assertRaisesRegex(ValueError, "body failed"):
            with lock:
                raise ValueError("body failed")
        self.assertEqual(lock._fd, -1)
        self.assertTrue(self._try_in_subprocess(True)["acquired"])

    @parametrize("operation", ["acquire", "release", "probe", "probe_release"])
    def test_os_errors_close_descriptor(self, operation):
        lock = filelock.FileLock(self.lock_path, exclusive=True)
        if operation == "release":
            lock.__enter__()
        failure = OSError(errno.EIO, "lock operation failed")
        descriptors = []

        def fail(fd, *args, **kwargs):
            descriptors.append(fd)
            raise failure

        target = "_unlock" if operation in ("release", "probe_release") else "_try_lock"
        with mock.patch.object(filelock, target, side_effect=fail):
            with self.assertRaises(OSError) as error:
                if operation == "acquire":
                    lock.__enter__()
                elif operation == "release":
                    lock.__exit__()
                else:
                    filelock._flock_held_exclusively(str(self.lock_path))
        self.assertIs(error.exception, failure)
        self.assertEqual(lock._fd, -1)
        self.assertEqual(len(descriptors), 1)
        with self.assertRaises(OSError) as closed:
            os.fstat(descriptors[0])
        self.assertEqual(closed.exception.errno, errno.EBADF)
        with filelock.FileLock(self.lock_path, exclusive=True, timeout=0):
            pass

    def test_probe_propagates_open_error(self):
        with self.assertRaises(FileNotFoundError):
            filelock._flock_held_exclusively(str(self.lock_path / "missing.lock"))

    def test_invalid_descriptor_is_not_contention(self):
        fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR)
        os.close(fd)
        with self.assertRaises(OSError):
            filelock._try_lock(fd, exclusive=True)

    @unittest.skipUnless(os.name == "nt", "requires Windows locking APIs")
    @parametrize("operation", ["acquire", "release"])
    @parametrize("error_code", [33, 5, 6])
    def test_windows_lock_errors(self, operation, error_code):
        fd = os.open(self.lock_path, os.O_CREAT | os.O_RDWR)
        self.addCleanup(os.close, fd)
        target = "_lock_file_ex" if operation == "acquire" else "_unlock_file_ex"
        with (
            mock.patch.object(filelock, target, return_value=0),
            mock.patch.object(filelock.ctypes, "get_last_error", return_value=error_code),
        ):
            if operation == "acquire" and error_code == 33:
                self.assertFalse(filelock._try_lock(fd, exclusive=True))
            else:
                with self.assertRaises(OSError) as error:
                    if operation == "acquire":
                        filelock._try_lock(fd, exclusive=True)
                    else:
                        filelock._unlock(fd)
                self.assertEqual(error.exception.winerror, error_code)

    @unittest.skipIf(os.name == "nt", "requires fcntl")
    @parametrize("error_code", [errno.EACCES, errno.EAGAIN, errno.EIO])
    def test_unix_lock_errors(self, error_code):
        failure = OSError(error_code, "flock failed")
        with mock.patch.object(filelock.fcntl, "flock", side_effect=failure):
            if error_code in (errno.EACCES, errno.EAGAIN):
                self.assertFalse(filelock._try_lock(-1, exclusive=True))
            else:
                with self.assertRaises(OSError) as error:
                    filelock._try_lock(-1, exclusive=True)
                self.assertIs(error.exception, failure)

    @parametrize("name", ["jit", "async_compile"])
    def test_cache_module_imports(self, name):
        result = subprocess.run(
            [sys.executable, "-I", "-B", "-c", IMPORT_WORKER, str(CACHE_ROOT), name],
            capture_output=True,
            text=True,
            timeout=30,
        )
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    run_tests()
