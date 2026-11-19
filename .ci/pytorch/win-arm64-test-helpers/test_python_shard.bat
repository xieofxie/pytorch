@echo off
setlocal EnableExtensions

if not "%SHARD_NUMBER%"=="1" goto python_tests

echo Running smoke_test.py...
python .ci\pytorch\smoke_test\smoke_test.py --package torchonly
:: A Windows process failure can have a negative exit code.
if not "%ERRORLEVEL%"=="0" exit /b %ERRORLEVEL%

:python_tests
echo Running Windows CPU Python tests...
pushd test
if not "%ERRORLEVEL%"=="0" exit /b %ERRORLEVEL%

:: Keep selection aligned with win-test-helpers\test_python_shard.bat.
python run_test.py --exclude-distributed-tests --exclude-quantization-tests --shard "%SHARD_NUMBER%" "%NUM_TEST_SHARDS%" --verbose
set "TEST_EXIT_CODE=%ERRORLEVEL%"
popd
endlocal & exit /b %TEST_EXIT_CODE%
