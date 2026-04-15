# instrukcje_windows.md

## Wymagania wstepne (Windows)

1. **Python 3.8+** — najlepiej przez conda lub oficjalny installer
2. **CUDA Toolkit** — zgodny z wersja PyTorch (np. CUDA 11.8 lub 12.1)
3. **Visual Studio Build Tools** (MSVC 142 lub 143) — wymagane do kompilacji CUDA
4. **Git** z obsługa submodules
5. **COLMAP** — zainstalowany i dodany do PATH (opcjonalnie, do rekonstrukcji)

### Aktywacja Visual Studio (wymagana przed instalacja)

Otwórz **PowerShell** i uruchom:

```powershell
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
```

Alternatywnie otwórz **"x64 Native Tools Command Prompt for VS 2022"** z menu Start.

---

## Szybki start

```powershell
cd C:\sciezka\do\gsplat-lidar\examples
.\venv\Scripts\activate    # lub: conda activate gsplat
```

Sprawdz GPU i torch:

```powershell
python -c "import torch; print('cuda:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"
```

---

## Instalacja gsplat z kodu zrodlowego

```powershell
# 1. Klonowanie z submodulami
git clone --recursive https://github.com/Piotrek2995/gsplat-lidar.git
cd gsplat-lidar

# 2. Utworzenie srodowiska wirtualnego
python -m venv venv
.\venv\Scripts\activate

# 3. Instalacja PyTorch (przyklad dla CUDA 12.1)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# 4. Instalacja gsplat (kompilacja CUDA)
pip install -e .

# 5. Instalacja zaleznosci dla przykladow
cd examples
pip install -r requirements.txt
```

> **Uwaga:** Jesli `pip install -e .` sie nie powiedzie, upewnij sie ze:
> - Visual Studio Build Tools sa zainstalowane i aktywowane
> - `cl.exe` jest na PATH (sprawdz: `where cl`)
> - CUDA Toolkit jest zainstalowany (sprawdz: `nvcc --version`)

---

## 1) Uczenie modelu (standard)

Przyklad jak w `moja_scena`:

```powershell
cd C:\sciezka\do\gsplat-lidar\examples
.\venv\Scripts\activate

python simple_trainer.py default `
  --data_dir C:\sciezka\do\moja_scena `
  --data_factor 1 `
  --max_steps 30000 `
  --save_steps 7000 30000 `
  --eval_steps 7000 30000 `
  --result_dir results\moja_scena
```

> **Uwaga PowerShell:** Znak kontynuacji linii to `` ` `` (backtick), NIE `\`.

Pliki wynikowe:
- checkpointy: `results\moja_scena\ckpts\`
- logi/metryki: `results\moja_scena\`

---

## 2) Uczenie dla atlas (wieksze zdjecia, mniej OOM)

```powershell
cd C:\sciezka\do\gsplat-lidar\examples
.\venv\Scripts\activate

$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"

python simple_trainer.py default `
  --data_dir C:\sciezka\do\gsplat-lidar\examples\data\atlas `
  --data_factor 4 `
  --packed `
  --max_steps 30000 `
  --save_steps 7000 30000 `
  --eval_steps 7000 30000 `
  --result_dir results\atlas_drive
```

Uwagi:
- `--data_factor 4` zmniejsza zuzycie VRAM.
- `--packed` zwykle pomaga na mniejszych GPU.

---

## 3) Podglad wyniku (viewer)

Do ogladania checkpointa uzywaj `simple_viewer.py`:

```powershell
cd C:\sciezka\do\gsplat-lidar\examples
.\venv\Scripts\activate

python simple_viewer.py --ckpt results\moja_scena\ckpts\ckpt_29999_rank0.pt
```

Wazne:
- Nie dawaj `--disable_viewer`, jesli chcesz okno podgladu.

---

## 4) Minimalna struktura danych COLMAP

Dataset dla `simple_trainer.py` powinien miec:

```text
<scene>\
  images\
    00000.png
    ...
  sparse\
    cameras.txt  (lub .bin)
    images.txt   (lub .bin)
    points3D.txt (lub .bin)
```

Jesli obrazy sa luzem w katalogu, przenies je do `images\`.

---

## 5) Najczestsze bledy

### `ModuleNotFoundError: No module named 'datasets'`
Uruchamiasz z niewlasciwego katalogu.

Poprawnie:

```powershell
cd C:\sciezka\do\gsplat-lidar\examples
python simple_trainer.py ...
```

### OOM / brak VRAM
Sprobuj:
- `--data_factor 2` lub `--data_factor 4`
- `--packed`
- `$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"`

### Brak `images_<factor>`
Dla wyzszych factorow pipeline moze oczekiwac folderu pochodnego.
Najprostsze wyjscie: ustaw `--data_factor 1` albo przygotuj odpowiedni cache/downscale.

### `Exception: Input quaternion should be a 3- or 4-vector` (COLMAP txt na Python 3.12)
Problem: parser `pycolmap` na Pythonie 3.12 nie moze czytac COLMAP `.txt`.

Rozwiazanie: konwertuj COLMAP model z `.txt` na `.bin`:

```powershell
colmap model_converter --input_path <dataset>\sparse\0 --output_path <dataset>\sparse\0 --output_type BIN
```

### Kompilacja CUDA nie dziala — `cl.exe` not found
Upewnij sie ze Visual Studio Build Tools sa aktywowane:

```powershell
& "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
```

Lub otwórz "x64 Native Tools Command Prompt for VS 2022".

### `RuntimeError: Distributed package doesn't have NCCL built in`
NCCL nie jest dostepny na Windows. Jesli uzywasz wersji z brancha `windows-install`, to jest juz naprawione (uzywa `gloo`). Jesli nie — unikaj multi-GPU na Windows lub zaktualizuj do tego brancha.

---

## 6) Przydatne komendy diagnostyczne

Status gita:

```powershell
cd C:\sciezka\do\gsplat-lidar
git status --short --branch
```

Sprawdzenie checkpointow:

```powershell
Get-ChildItem .\examples\results\moja_scena\ckpts\
```

Szybkie szukanie bledow w logach:

```powershell
Select-String -Pattern "Traceback|ERROR|Exception" -Path .\log.txt
```

Sprawdzenie CUDA i nvcc:

```powershell
nvcc --version
python -c "import torch; print(torch.version.cuda)"
```

---

## 7) Fork + branch (bezpieczna praca)

Aktualny model pracy:
- `upstream`: oryginalne repo
- `origin`: Twoj fork
- branch roboczy, np. `windows-install`

Push zmian:

```powershell
cd C:\sciezka\do\gsplat-lidar
git push -u origin windows-install
```

Potem otworz PR z forka do upstream.

---

## 8) Fuzja LiDAR + GSPLAT (przygotowanie do ICP)

### Krok 1: Konwersja obrazow atlas z RGBA do RGB

Powod: przy RGBA COLMAP moze rzucac `BITMAP_ERROR`.

```powershell
cd C:\sciezka\do\gsplat-lidar\examples
.\venv\Scripts\activate

python -c @"
from pathlib import Path
from PIL import Image

src = Path('data/atlas/images')
dst = Path('data/atlas/images_rgb')
dst.mkdir(parents=True, exist_ok=True)

count = 0
for p in sorted(src.glob('*.png')):
    with Image.open(p) as im:
        im.convert('RGB').save(dst / p.name)
    count += 1

print('converted', count)
"@
```

> **Uwaga:** W PowerShell wieloliniowe stringi uzywamy `@" ... "@` (here-string).

### Krok 2: Pelna rekonstrukcja COLMAP na `images_rgb`

```powershell
cd C:\sciezka\do\gsplat-lidar\examples\data\atlas

# Wyczysc poprzednie wyniki
Remove-Item -Recurse -Force database.db, sparse -ErrorAction SilentlyContinue
New-Item -ItemType Directory -Force -Path sparse

colmap feature_extractor `
  --database_path database.db `
  --image_path images_rgb

colmap exhaustive_matcher `
  --database_path database.db

colmap mapper `
  --database_path database.db `
  --image_path images_rgb `
  --output_path sparse

colmap model_converter `
  --input_path sparse\0 `
  --output_path sparse\0 `
  --output_type TXT

colmap model_converter `
  --input_path sparse\0 `
  --output_path sparse\atlas_sparse_points.ply `
  --output_type PLY
```

Kontrola wyniku:

```powershell
Get-ChildItem sparse\atlas_sparse_points.ply
colmap model_analyzer --path sparse\0
```

### Krok 3: Wyrownanie ICP (CloudCompare) — recznie

1. Otworz w CloudCompare:
   - `sparse\atlas_sparse_points.ply` (z COLMAP)
   - chmure LiDAR (docelowa referencja)
2. Zrob wstepne ustawienie (jesli trzeba), potem ICP.
3. Zapisz:
   - transformacje COLMAP->LiDAR (macierz 4x4),
   - wyrownana chmure.

### Krok 4: Start treningu z geometrii LiDAR

```powershell
cd C:\sciezka\do\gsplat-lidar\examples
.\venv\Scripts\activate

python simple_trainer_z_lidar.py default `
  --data_dir C:\sciezka\do\gsplat-lidar\examples\data\atlas `
  --no-normalize-world-space `
  --lidar_ply lidar\atlas_wyciete_NAJLEPSZE_SKALA.ply `
  --lidar_max_points 300000 `
  --data_factor 4 `
  --packed `
  --max_steps 30000 `
  --save_steps 7000 30000 `
  --eval_steps 1000000 `
  --result_dir results\atlas_lidar_init
```

Uwagi:
- `--lidar_ply` moze byc sciezka wzgledna wzgledem `--data_dir` albo absolutna.
- Jesli chmura jest bardzo duza, zwiekszaj/zmniejszaj `--lidar_max_points` pod VRAM.
- `--no-normalize-world-space` zostawia skale metryczna po ICP.
