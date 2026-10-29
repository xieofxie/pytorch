# Owner(s): ["module: inductor"]

import os
import subprocess
import sys
import textwrap
import unittest

import torch
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    parametrize,
    run_tests,
    TestCase,
)


@unittest.skipUnless(IS_WINDOWS, "Windows DLL lifetime regression")
class TestAOTIWindowsLifetime(TestCase):
    @parametrize("size", [1024, 65536])
    @parametrize("concurrent", [False, True])
    def test_cpu_model_dll_retained(self, device, size, concurrent):
        class Model(torch.nn.Module):
            def forward(self, x):
                return (x + 1,)

        inputs = (torch.randn(size, device=device),)
        exported = torch.export.export(Model(), inputs)
        model_path = torch._inductor.aot_compile(
            exported.module(), inputs, options={"cpp.threads": 2}
        )
        # Isolate access violations and let process exit release the DLL mapping.
        script = textwrap.dedent(
            """
            import ctypes
            import gc
            import os
            import sys
            from concurrent.futures import ThreadPoolExecutor
            from threading import Barrier

            import torch
            from torch.testing._internal.common_utils import TestCase

            path, size, concurrent = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
            torch.set_num_threads(2)
            check = TestCase()
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
            kernel32.GetModuleHandleW.restype = ctypes.c_void_p
            path = os.path.abspath(path)
            check.assertIsNone(kernel32.GetModuleHandleW(path))
            barrier = Barrier(4) if concurrent else None

            def run_once(index):
                x = torch.full((size,), float(index))
                expected = x + 1
                runner = torch._C._aoti.AOTIModelContainerRunnerCpu(path, 1)
                output = runner.run([x])[0]
                check.assertEqual(output, expected)
                if barrier is not None:
                    barrier.wait(timeout=60)
                del runner
                gc.collect()
                check.assertEqual(output, expected)
                check.assertIsNotNone(kernel32.GetModuleHandleW(path))

            if concurrent:
                with ThreadPoolExecutor(max_workers=4) as executor:
                    list(executor.map(run_once, range(8)))
            else:
                for index in range(8):
                    run_once(index)

            # Retaining the module also keeps the file locked until process exit.
            with check.assertRaises(PermissionError):
                os.remove(path)
            """
        )
        env = {**os.environ, "OMP_NUM_THREADS": "2"}
        env.pop("OMP_WAIT_POLICY", None)
        env.pop("KMP_BLOCKTIME", None)
        result = subprocess.run(
            [
                sys.executable,
                "-I",
                "-X",
                "faulthandler",
                "-c",
                script,
                model_path,
                str(size),
                str(int(concurrent)),
            ],
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


instantiate_device_type_tests(TestAOTIWindowsLifetime, globals(), only_for="cpu")


if __name__ == "__main__":
    run_tests()
