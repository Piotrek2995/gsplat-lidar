# setup_env_windows.ps1
# Tworzy srodowisko wirtualne i instaluje wszystkie zaleznosci gsplat-lidar na Windows.
#
# Uzycie:
#   .\setup_env_windows.ps1                      # domyslnie CUDA 12.1
#   .\setup_env_windows.ps1 -CudaVersion "11.8"  # dla CUDA 11.8
#
# Wymagania:
#   - Python 3.8+ w PATH
#   - CUDA Toolkit zainstalowany
#   - Visual Studio Build Tools (MSVC) zainstalowane
#     (otworz "x64 Native Tools Command Prompt for VS 2022" lub uruchom vcvars64.bat przed tym skryptem)

param(
    [string]$CudaVersion = "12.8",
    [string]$VenvName = "gsplat_env"
)

$ErrorActionPreference = "Stop"

# --- Funkcja do sprawdzania kodow wyjscia polecen zewnetrznych (np. pip) ---
function Assert-Success {
    if ($LASTEXITCODE -ne 0) {
        Write-Host "BLAD: Ostatnie polecenie zwrocilo blad (kod $LASTEXITCODE). Przerywam dzialanie skryptu." -ForegroundColor Red
        exit 1
    }
}

# --- Mapowanie CUDA -> PyTorch index URL ---
$cudaMap = @{
    "11.8" = "cu118"
    "12.1" = "cu121"
    "12.4" = "cu124"
    "12.6" = "cu126"
    "12.8" = "cu128"
}

if (-not $cudaMap.ContainsKey($CudaVersion)) {
    Write-Host "Nieobslugiwana wersja CUDA: $CudaVersion" -ForegroundColor Red
    Write-Host "Dostepne: $($cudaMap.Keys -join ', ')" -ForegroundColor Yellow
    exit 1
}
$cudaTag = $cudaMap[$CudaVersion]
$torchIndexUrl = "https://download.pytorch.org/whl/$cudaTag"

# --- Auto-detect CUDA toolkit and add to PATH ---
# Always prefer the version matching $CudaVersion parameter
function Find-CudaToolkit {
    param([string]$PreferredVersion)

    $basePath = "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
    if (-not (Test-Path $basePath)) { return $null }

    # Try exact match first (e.g. "12.8" -> "v12.8")
    $exactMatch = Join-Path $basePath "v$PreferredVersion"
    if (Test-Path (Join-Path $exactMatch "bin\nvcc.exe")) {
        return $exactMatch
    }

    # Fall back to latest available version
    $versions = Get-ChildItem $basePath -Directory | Sort-Object Name -Descending
    foreach ($ver in $versions) {
        $nvccPath = Join-Path $ver.FullName "bin\nvcc.exe"
        if (Test-Path $nvccPath) {
            return $ver.FullName
        }
    }
    return $null
}

$cudaHome = Find-CudaToolkit -PreferredVersion $CudaVersion
if ($cudaHome) {
    $cudaBin = Join-Path $cudaHome "bin"
    $cudaLibnvvp = Join-Path $cudaHome "libnvvp"
    # Prepend to PATH so this version takes priority over any older nvcc
    $env:PATH = "$cudaBin;$cudaLibnvvp;$env:PATH"
    $env:CUDA_HOME = $cudaHome
    $env:CUDA_PATH = $cudaHome
    Write-Host "  CUDA Toolkit: $cudaHome" -ForegroundColor Green
    Write-Host "  Dodano do PATH na czas tej sesji." -ForegroundColor Green
}

# --- Sprawdzenie narzedzi ---
Write-Host "=== Sprawdzanie narzedzi ===" -ForegroundColor Cyan

# Python
try {
    $pyVer = python --version 2>&1
    Write-Host "  Python: $pyVer" -ForegroundColor Green
} catch {
    Write-Host "  BLAD: Python nie znaleziony w PATH!" -ForegroundColor Red
    exit 1
}

# nvcc
$nvccCmd = Get-Command nvcc.exe -ErrorAction SilentlyContinue
if ($nvccCmd) {
    $nvccVer = nvcc --version 2>&1 | Select-String "release"
    Write-Host "  NVCC:   $nvccVer" -ForegroundColor Green
} else {
    Write-Host "  UWAGA: nvcc nie znaleziony w PATH ani w standardowych lokalizacjach." -ForegroundColor Yellow
    Write-Host "         Zainstaluj CUDA Toolkit: https://developer.nvidia.com/cuda-downloads" -ForegroundColor Yellow
    Write-Host "         gsplat zostanie zainstalowany bez CUDA (wolniejszy, uzyje JIT pozniej)." -ForegroundColor Yellow
}

# cl.exe (MSVC) - auto-detect via vcvars64.bat
$clPath = Get-Command cl.exe -ErrorAction SilentlyContinue
if ($clPath) {
    Write-Host "  MSVC:   $($clPath.Source)" -ForegroundColor Green
    $env:DISTUTILS_USE_SDK = "1"
} else {
    Write-Host "  cl.exe nie znaleziony - szukam vcvars64.bat..." -ForegroundColor Yellow

    # Search for vcvars64.bat in standard Visual Studio / Build Tools locations
    $vcvarsSearch = @(
        "C:\Program Files\Microsoft Visual Studio",
        "C:\Program Files (x86)\Microsoft Visual Studio"
    )
    $vcvarsBat = $null
    foreach ($base in $vcvarsSearch) {
        if (Test-Path $base) {
            $found = Get-ChildItem -Path $base -Recurse -Filter "vcvars64.bat" -ErrorAction SilentlyContinue | Select-Object -First 1
            if ($found) {
                $vcvarsBat = $found.FullName
                break
            }
        }
    }

    if ($vcvarsBat) {
        Write-Host "  Znaleziono: $vcvarsBat" -ForegroundColor Green
        Write-Host "  Importowanie zmiennych srodowiskowych MSVC..." -ForegroundColor Yellow

        # Run vcvars64.bat in cmd and capture the resulting environment variables
        $envLines = cmd /c "`"$vcvarsBat`" >nul 2>&1 && set" 2>&1
        foreach ($line in $envLines) {
            $lineStr = "$line"
            $eqIdx = $lineStr.IndexOf('=')
            if ($eqIdx -gt 0) {
                $varName  = $lineStr.Substring(0, $eqIdx)
                $varValue = $lineStr.Substring($eqIdx + 1)
                [System.Environment]::SetEnvironmentVariable($varName, $varValue, "Process")
            }
        }

        # Verify cl.exe is now available
        $clPath = Get-Command cl.exe -ErrorAction SilentlyContinue
        if ($clPath) {
            Write-Host "  MSVC:   $($clPath.Source)" -ForegroundColor Green
            # PyTorch cpp_extension wymaga tej flagi gdy VC env jest aktywny
            $env:DISTUTILS_USE_SDK = "1"
            Write-Host "  Ustawiono DISTUTILS_USE_SDK=1" -ForegroundColor Green
        } else {
            Write-Host "  UWAGA: vcvars64.bat uruchomiony, ale cl.exe wciaz nie znaleziony!" -ForegroundColor Red
            Write-Host "         Sprawdz instalacje Visual Studio Build Tools." -ForegroundColor Yellow
        }
    } else {
        Write-Host "  UWAGA: vcvars64.bat nie znaleziony!" -ForegroundColor Red
        Write-Host "         Zainstaluj Visual Studio Build Tools: https://visualstudio.microsoft.com/visual-cpp-build-tools/" -ForegroundColor Yellow
        Write-Host "         Lub otworz 'x64 Native Tools Command Prompt' i uruchom ten skrypt stamtad." -ForegroundColor Yellow
    }
}

# --- Tworzenie venv ---
Write-Host ""
Write-Host "=== Tworzenie srodowiska: $VenvName ===" -ForegroundColor Cyan

if (Test-Path $VenvName) {
    Write-Host "  Folder $VenvName juz istnieje - pomijam tworzenie." -ForegroundColor Yellow
} else {
    python -m venv $VenvName
    Write-Host "  Utworzono: $VenvName" -ForegroundColor Green
}

# Aktywacja
$activateScript = ".\$VenvName\Scripts\Activate.ps1"
if (-not (Test-Path $activateScript)) {
    Write-Host "  BLAD: Nie znaleziono $activateScript" -ForegroundColor Red
    exit 1
}
& $activateScript
Write-Host "  Aktywowano srodowisko." -ForegroundColor Green

# --- Upgrade pip + install build tools ---
Write-Host ""
Write-Host "=== Aktualizacja pip i narzedzi budowania ===" -ForegroundColor Cyan
python -m pip install --upgrade pip setuptools wheel ninja
Assert-Success

# --- PyTorch ---
Write-Host ""
Write-Host "=== Instalacja PyTorch (CUDA $CudaVersion) ===" -ForegroundColor Cyan
pip install torch torchvision --index-url $torchIndexUrl
Assert-Success

# Weryfikacja
python -c "import torch; print(f'  Torch {torch.__version__}, CUDA: {torch.cuda.is_available()}, GPU count: {torch.cuda.device_count()}')"

# --- gsplat z kodu zrodlowego ---
Write-Host ""
Write-Host "=== Instalacja gsplat (kompilacja z kodu zrodlowego) ===" -ForegroundColor Cyan
Write-Host "  To moze potrwac kilka minut przy pierwszym uruchomieniu..." -ForegroundColor Yellow

# Upewnij sie ze submoduly sa pobrane
git submodule update --init --recursive

# --no-build-isolation: uzywa torch juz zainstalowanego w venv
# (bez tego pip tworzy izolowane srodowisko BEZ torch i setup.py nie widzi torch)

# Wymagane dla nowszych wersji MSVC (np. VS 2025/18.0), których CUDA 12.8 oficjalnie nie wspiera
# NVCC_APPEND_FLAGS jest czytane bezposrednio przez kompilator nvcc (wymagane m.in dla fused-ssim)
$env:NVCC_APPEND_FLAGS = "-allow-unsupported-compiler"
Write-Host "  Ustawiono NVCC_APPEND_FLAGS=-allow-unsupported-compiler" -ForegroundColor Green

$nvccAvailable = [bool](Get-Command nvcc.exe -ErrorAction SilentlyContinue)
$clAvailable = [bool](Get-Command cl.exe -ErrorAction SilentlyContinue)

if ($nvccAvailable -and $clAvailable) {
    Write-Host "  Kompilacja z CUDA (nvcc + cl.exe znalezione)..." -ForegroundColor Green
    pip install --no-build-isolation -e .
    Assert-Success
} else {
    Write-Host "  Brak nvcc lub cl.exe - instalacja bez kompilacji CUDA." -ForegroundColor Yellow
    Write-Host "  gsplat uzyje JIT kompilacji przy pierwszym uzyciu (jesli nvcc bedzie w PATH)." -ForegroundColor Yellow
    $env:BUILD_NO_CUDA = "1"
    pip install --no-build-isolation -e .
    Assert-Success
    Remove-Item Env:\BUILD_NO_CUDA -ErrorAction SilentlyContinue
}

# Weryfikacja
python -c "import gsplat; print('  gsplat zainstalowany pomyslnie')"

# --- Zaleznosci dla przykladow ---
Write-Host ""
Write-Host "=== Instalacja zaleznosci examples/requirements.txt ===" -ForegroundColor Cyan
Write-Host "  Krok 1/2: Pakiety Python (bez CUDA)..." -ForegroundColor White

# Instaluj zwykle pakiety (bez tych wymagajacych kompilacji CUDA)
$nonCudaDeps = @(
    "viser",
    "imageio[ffmpeg]",
    "numpy<2.0.0",
    "scikit-learn",
    "tqdm",
    "torchmetrics[image]",
    "opencv-python",
    "tyro>=0.8.8",
    "Pillow",
    "piexif",
    "tensorboard",
    "tensorly",
    "pyyaml",
    "matplotlib",
    "splines"
)
pip install $nonCudaDeps
Assert-Success

# Pakiety git ktore NIE wymagaja CUDA
Write-Host "  Krok 1b: Pakiety git (Python-only)..." -ForegroundColor White
pip install "git+https://github.com/rmbrualla/pycolmap@cc7ea4b7301720ac29287dbe450952511b32125e"
Assert-Success
pip install "git+https://github.com/nerfstudio-project/nerfview@4538024fe0d15fd1a0e4d760f3695fc44ca72787"
Assert-Success

# Pakiety git wymagajace torch/CUDA przy budowie
Write-Host "  Krok 2/2: Pakiety CUDA (fused-ssim, fused-bilagrid, ppisp)..." -ForegroundColor White
pip install --no-build-isolation "git+https://github.com/rahul-goel/fused-ssim@328dc9836f513d00c4b5bc38fe30478b4435cbb5"
Assert-Success
pip install --no-build-isolation "git+https://github.com/harry7557558/fused-bilagrid@49f0ef06c9f81810fb9b5dd9027cf1844950cc16"
Assert-Success
pip install --no-build-isolation "ppisp @ git+https://github.com/nv-tlabs/ppisp@v1.0.0"
Assert-Success

# --- Podsumowanie ---
Write-Host ""
Write-Host "========================================" -ForegroundColor Green
Write-Host "  GOTOWE! Srodowisko skonfigurowane." -ForegroundColor Green
Write-Host "========================================" -ForegroundColor Green
Write-Host ""
Write-Host "Aby aktywowac srodowisko w nowej sesji:" -ForegroundColor Cyan
Write-Host "  .\$VenvName\Scripts\activate" -ForegroundColor White
Write-Host ""
Write-Host "Aby uruchomic trening:" -ForegroundColor Cyan
Write-Host "  cd examples" -ForegroundColor White
Write-Host '  python simple_trainer.py default --data_dir <scena> --disable_viewer' -ForegroundColor White
Write-Host ""
Write-Host "Aby uruchomic trening z LiDAR:" -ForegroundColor Cyan
Write-Host "  cd examples" -ForegroundColor White
Write-Host '  python simple_trainer_z_lidar.py default --data_dir <scena> --lidar_ply <plik.ply>' -ForegroundColor White
