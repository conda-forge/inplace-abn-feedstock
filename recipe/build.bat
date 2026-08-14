@echo on

if "%cuda_compiler_version%" == "13.0" (
    set "TORCH_CUDA_ARCH_LIST=7.5;8.0;9.0"
)

python -m pip install . -vv --no-deps --no-build-isolation
if errorlevel 1 exit /b 1
