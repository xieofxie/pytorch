# Owner(s): ["module: inductor"]

import os
import subprocess
import sys
import textwrap
import unittest
from unittest import mock

import torch
from torch._inductor.codecache import AotCodeCompiler
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
    @parametrize("packaged", [False, True])
    def test_cpu_model_dll_unloaded(self, device, size, concurrent, packaged):
        class Model(torch.nn.Module):
            def forward(self, x):
                return (x + 1,)

        inputs = (torch.randn(size, device=device),)
        exported = torch.export.export(Model(), inputs)
        options = {"cpp.threads": 2, "cpp.min_chunk_size": 4096}
        with mock.patch.object(
            AotCodeCompiler, "compile", wraps=AotCodeCompiler.compile
        ) as compile_model:
            if packaged:
                model_path = torch._inductor.aoti_compile_and_package(
                    exported, inductor_configs=options
                )
            else:
                model_path = torch._inductor.aot_compile(
                    exported.module(), inputs, options=options
                )
        kernel_code = compile_model.call_args.args[2]
        self.assertEqual("#pragma omp parallel" in kernel_code, size == 65536)
        # A fresh process must not inherit another model's OpenMP runtime reference.
        script = textwrap.dedent(
            """
            import ctypes
            import gc
            import os
            import shutil
            import sys
            import tempfile
            import zipfile
            from concurrent.futures import ThreadPoolExecutor
            from pathlib import Path
            from threading import Barrier

            import torch
            from torch.testing._internal.common_utils import TestCase

            source = sys.argv[1]
            size, concurrent, packaged = map(int, sys.argv[2:])
            torch.set_num_threads(2)
            check = TestCase()
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            kernel32.GetModuleHandleW.argtypes = [ctypes.c_wchar_p]
            kernel32.GetModuleHandleW.restype = ctypes.c_void_p
            barrier = Barrier(4) if concurrent else None

            def run_once(index):
                x = torch.full((size,), float(index))
                expected = x + 1
                with tempfile.TemporaryDirectory() as directory:
                    if packaged:
                        with zipfile.ZipFile(source) as archive:
                            archive.extractall(directory)
                        paths = [
                            path for path in Path(directory).rglob("*")
                            if path.suffix in (".dll", ".pyd")
                        ]
                        check.assertEqual(len(paths), 1)
                        path = str(paths[0].resolve())
                    else:
                        path = os.path.join(directory, "model.dll")
                        shutil.copyfile(source, path)
                    check.assertIsNone(kernel32.GetModuleHandleW(path))
                    if packaged:
                        runner = torch._inductor.aoti_load_package(directory)
                    else:
                        runner = torch._C._aoti.AOTIModelContainerRunnerCpu(path, 1)
                    check.assertIsNotNone(kernel32.GetModuleHandleW(path))
                    if barrier is not None:
                        barrier.wait(timeout=60)
                    output = runner(x)[0] if packaged else runner.run([x])[0]
                    del runner
                    check.assertIsNone(kernel32.GetModuleHandleW(path))
                    os.remove(path)
                    check.assertFalse(os.path.exists(path))
                    gc.collect()
                    check.assertEqual(output, expected)
                    return output, expected

            if concurrent:
                with ThreadPoolExecutor(max_workers=4) as executor:
                    results = list(executor.map(run_once, range(16)))
            else:
                results = [run_once(index) for index in range(16)]
            for output, expected in results:
                check.assertEqual(output, expected)
            """
        )
        env = {**os.environ, "OMP_NUM_THREADS": "2", "OMP_DYNAMIC": "FALSE"}
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
                str(int(packaged)),
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
