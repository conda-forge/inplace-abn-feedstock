@echo on

if not "%cuda_compiler_version%" == "None" (
    set TORCH_CUDA_ARCH_LIST=7.0;8.0;9.0
)

python -m pip install . -vv
if errorlevel 1 exit 1
