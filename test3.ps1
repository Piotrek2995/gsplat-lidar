$env:DISTUTILS_USE_SDK = "1"
$env:NVCC_APPEND_FLAGS = "-allow-unsupported-compiler"
$env:CUDA_HOME = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
$env:CUDA_PATH = $env:CUDA_HOME
$env:PATH = "$env:CUDA_HOME\bin;$env:CUDA_HOME\libnvvp;$env:PATH"
cmd /c " "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul 2>&1 && .\gsplat_env\Scripts\activate.bat && pip install --no-build-isolation -v git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5 > build_log5.txt 2>&1 "
