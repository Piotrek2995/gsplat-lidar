$env:DISTUTILS_USE_SDK = "1"
$env:NVCC_APPEND_FLAGS = "-allow-unsupported-compiler"
& "C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
.\gsplat_env\Scripts\activate.ps1
pip install --no-build-isolation -v git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5 > build_log3.txt 2>&1
