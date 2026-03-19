# instrukcje.md

## Szybki start

```bash
cd ~/gsplat/examples
source ~/gsplat/gsplat_env/bin/activate
```

Sprawdz GPU i torch:

```bash
python -c "import torch; print('cuda:', torch.cuda.is_available(), 'count:', torch.cuda.device_count())"
```

---

## 1) Uczenie modelu (standard)

Przyklad jak w `moja_scena`:

```bash
cd ~/gsplat/examples
source ~/gsplat/gsplat_env/bin/activate

python simple_trainer.py default \
  --data_dir /sciezka/do/moja_scena \
  --data_factor 1 \
  --max_steps 30000 \
  --save_steps 7000 30000 \
  --eval_steps 7000 30000 \
  --result_dir results/moja_scena
```

Pliki wynikowe:
- checkpointy: `results/moja_scena/ckpts/`
- logi/metryki: `results/moja_scena/`

---

## 2) Uczenie dla atlas (wieksze zdjecia, mniej OOM)

```bash
cd ~/gsplat/examples
source ~/gsplat/gsplat_env/bin/activate

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python simple_trainer.py default \
  --data_dir ~/gsplat/examples/data/atlas/drive-download-20260318T082918Z-1-001 \
  --data_factor 4 \
  --packed \
  --max_steps 30000 \
  --save_steps 7000 30000 \
  --eval_steps 7000 30000 \
  --result_dir results/atlas_drive_20260318
```

Uwagi:
- `--data_factor 4` zmniejsza zuzycie VRAM.
- `--packed` zwykle pomaga na mniejszych GPU.

---

## 3) Podglad wyniku (viewer)

Do ogladania checkpointa uzywaj `simple_viewer.py`:

```bash
cd ~/gsplat/examples
source ~/gsplat/gsplat_env/bin/activate

python simple_viewer.py --ckpt results/moja_scena/ckpts/ckpt_29999_rank0.pt
```

Wazne:
- Nie dawaj `--disable_viewer`, jesli chcesz okno podgladu.
- `simple_trainer.py --ckpt ... --disable_viewer` sluzy raczej do eval/render, nie do interaktywnego podgladu.

---

## 4) Minimalna struktura danych COLMAP

Dataset dla `simple_trainer.py` powinien miec:

```text
<scene>/
  images/
    00000.png
    ...
  sparse/
    cameras.txt  (lub .bin)
    images.txt   (lub .bin)
    points3D.txt (lub .bin)
```

Jesli obrazy sa luzem w katalogu, przenies je do `images/`.

---

## 5) Najczestsze bledy

### `ModuleNotFoundError: No module named 'datasets'`
Uruchamiasz z niewlasciwego katalogu.

Poprawnie:

```bash
cd ~/gsplat/examples
python simple_trainer.py ...
```

### OOM / brak VRAM
Sprobuj:
- `--data_factor 2` lub `--data_factor 4`
- `--packed`
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

### Brak `images_<factor>`
Dla wyzszych factorow pipeline moze oczekiwac folderu pochodnego.
Najprostsze wyjscie: ustaw `--data_factor 1` albo przygotuj odpowiedni cache/downscale.

### `Exception: Input quaternion should be a 3- or 4-vector` (COLMAP txt na Python 3.12)
Problem: parser `pycolmap` na Pythonie 3.12 nie moze czytac COLMAP `.txt` (bug z `map()` objektami).

Rozwiazanie: konwertuj COLMAP model z `.txt` na `.bin`:

```bash
colmap model_converter --input_path <dataset>/sparse/0 --output_path <dataset>/sparse/0 --output_type BIN
```

Po konwersji parser wykorzysta `.bin` pliki, ktore dzialaja bez bledow.

---

## 6) Przydatne komendy diagnostyczne

Status gita:

```bash
cd ~/gsplat
git status --short --branch
```

Sprawdzenie checkpointow:

```bash
ls -lah ~/gsplat/examples/results/moja_scena/ckpts
```

Szybkie szukanie bledow w logach (jesli logujesz do pliku):

```bash
rg -n "Traceback|ERROR|Exception" /sciezka/do/logu.txt
```

---

## 7) Fork + branch (bezpieczna praca)

Aktualny model pracy:
- `upstream`: oryginalne repo
- `origin`: Twoj fork
- branch roboczy, np. `atlas-lidar-fixes`

Push zmian:

```bash
cd ~/gsplat
git push -u origin atlas-lidar-fixes
```

Potem otworz PR z forka do upstream.
