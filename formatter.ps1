# formatter.ps1 — Windows equivalent of formatter.sh
# Requires: clang-format (e.g. via choco install llvm) and black (pip install black==22.3.0)

Write-Host "Running clang-format on CUDA sources..."
Get-ChildItem -Path "gsplat\cuda\include" -Recurse -Include "*.cpp","*.cuh","*.cu","*.h" |
    Where-Object { $_.FullName -notlike "*third_party*" } |
    ForEach-Object {
        Write-Host "  Formatting: $($_.FullName)"
        clang-format -i $_.FullName
    }

Write-Host ""
Write-Host "Running black on Python sources..."
black . gsplat\ tests\ examples\ profiling\

Write-Host "Done."
