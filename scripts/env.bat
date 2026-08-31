@echo off
REM Sets up the build environment for gsplat on Windows.
REM Must be run before compiling or training. See README.md for why each line exists.

call "C:\Program Files (x86)\Microsoft Visual Studio\2019\BuildTools\VC\Auxiliary\Build\vcvars64.bat"

REM v12.9 is first on PATH by default but does not match the torch cu128 build.
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "CUDA_PATH=%CUDA_HOME%"
set "PATH=%CUDA_HOME%\bin;%PATH%"

REM Ada only. Without this the build targets every architecture and takes ~8x longer.
set "TORCH_CUDA_ARCH_LIST=8.9"
set "MAX_JOBS=4"

REM torch's cpp_extension._check_abi raises when it sees an activated VC
REM environment without this set, which stops fused-ssim and fused-bilagrid
REM from building at all. Confirmed on a clean venv on 2026-08-31:
REM "UserWarning: It seems that the VC environment is activated but
REM DISTUTILS_USE_SDK is not set."
set "DISTUTILS_USE_SDK=1"

echo Environment ready. CUDA_HOME=%CUDA_HOME%
