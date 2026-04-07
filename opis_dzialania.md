# opis_dzialania.md

## 1) Cel rozwiazania

Celem bylo uruchomienie treningu GSPLAT tak, aby:
- kamery i geometria sceny (uklad odniesienia) pochodzily z rekonstrukcji COLMAP,
- inicjalne punkty Gaussow pochodzily z chmury LiDAR wyrownanej do COLMAP przez ICP,
- zostala zachowana kompatybilnosc z dotychczasowym pipeline treningowym.

Zamiast modyfikowac bezposrednio `examples/simple_trainer.py`, powstal osobny plik:
- `examples/simple_trainer_z_lidar.py`

To daje bezpieczny wariant eksperymentalny bez ryzyka popsucia bazowego trenera.

## 2) Przeplyw end-to-end

1. Przygotowanie obrazow do COLMAP:
- atlas mial PNG RGBA, co powodowalo `BITMAP_ERROR` w COLMAP.
- obrazy zostaly przekonwertowane do RGB (`images_rgb`).

2. Rekonstrukcja COLMAP:
- uruchomione: `feature_extractor`, `exhaustive_matcher`, `mapper`, `model_converter`.
- wynik: kompletne `sparse/0` + `atlas_sparse_points.ply`.

3. ICP w CloudCompare:
- chmura LiDAR zostala recznie wyrownana do ukladu COLMAP.
- wynik zapisany jako PLY w:
  - `examples/data/atlas/lidar/atlas_wyciete_NAJLEPSZE_SKALA.ply`

4. Trening LiDAR-init:
- uruchamiany przez `simple_trainer_z_lidar.py`.
- nadal korzysta z poz kamer z COLMAP, ale punkty inicjalizacyjne bierze z LiDAR.

## 3) Co dokladnie zmieniono w nowym trenerze

Plik bazowy:
- `examples/simple_trainer.py`

Plik docelowy:
- `examples/simple_trainer_z_lidar.py`

Nowe elementy (dokladnie):

### A) Loader PLY i narzedzia pomocnicze

Dodano blok funkcji:
- `_PLY_DTYPE_MAP`
- `_parse_ply_header(...)`
- `load_ply_points_rgb(...)`
- `apply_transform_to_points(...)`
- `random_downsample_points(...)`

Lokalizacja w kodzie:
- okolice linii 44-202 w `simple_trainer_z_lidar.py`.

Co robia:
- obsluguja PLY `ascii` i `binary_little_endian`,
- wymagaja wspolrzednych `x,y,z`,
- kolor czytaja z `red,green,blue` lub `r,g,b`,
- jesli koloru brak: podstawiaja neutralny kolor (127,127,127),
- umozliwiaja losowy downsample chmury do limitu punktow,
- umozliwiaja transformacje punktow macierza 4x4 (wspolrzedne jednorodne).

### B) Rozszerzenie konfiguracji CLI

Do `Config` dodano dwa pola:
- `lidar_ply: Optional[str] = None`
- `lidar_max_points: Optional[int] = None`

Lokalizacja:
- okolice linii 259-261.

Sens:
- `lidar_ply` wskazuje plik chmury LiDAR,
- `lidar_max_points` ogranicza liczbe punktow (VRAM/stabilnosc).

### C) Podmiana punktow inicjalizacji w `Runner.__init__`

Dodano blok warunkowy:
- `if cfg.lidar_ply is not None: ...`

Lokalizacja:
- okolice linii 524-548.

Krok po kroku:
1. Resolver sciezki:
- jesli sciezka wzgledna, doklejana do `cfg.data_dir`.

2. Walidacja istnienia pliku:
- brak pliku => `FileNotFoundError`.

3. Wczytanie danych z PLY:
- `lidar_points, lidar_rgbs = load_ply_points_rgb(...)`.

4. Dopasowanie ukladu wspolrzednych:
- gdy `cfg.normalize_world_space == True`, punkty LiDAR sa transformowane przez `self.parser.transform`.
- gdy uruchamiasz z `--no-normalize-world-space`, transformacja nie jest wykonywana.

5. Opcjonalny downsample:
- jesli ustawione `lidar_max_points`, wykonywany jest losowy wybor bez zwracania.

6. Wstrzykniecie punktow do parsera:
- `self.parser.points = lidar_points.astype(np.float32)`
- `self.parser.points_rgb = lidar_rgbs.astype(np.uint8)`
- `self.parser.points_err = np.ones(...)`

To jest klucz: dalszy kod trenera nie musi byc zmieniany, bo i tak bierze punkty z `parser.points`.

## 4) Dlaczego to dziala bez zmian w petli treningowej

W `create_splats_with_optimizers(...)` dla `init_type == "sfm"` jest:
- `points = torch.from_numpy(parser.points).float()`
- `rgbs = torch.from_numpy(parser.points_rgb / 255.0).float()`

Czyli:
- kamery i metadane nadal sa z parsera COLMAP,
- ale skoro podmieniono `parser.points`, inicjalizacja Gaussow idzie z LiDAR.

Reszta pipeline (densification, optymalizacja, logowanie, checkpointy, viewer) pozostaje bez zmian.

## 5) Co zostalo nietkniete

Nie zmieniano:
- logiki rasteryzacji,
- strategii treningowych,
- harmonogramow LR,
- zapisu checkpointow,
- eval/render/video,
- viewer flow.

Zmiana dotyczy tylko etapu zasilenia geometrii startowej.

## 6) Istotne uwagi uruchomieniowe

1. `data_factor=4` wymaga folderu `images_4`.
- dla PNG parser nie zrobi automatycznie cache tak jak dla JPG,
- `images_4` trzeba przygotowac recznie.

2. Skala metryczna:
- jesli chcesz zachowac skale po ICP: uruchamiaj z `--no-normalize-world-space`.

3. Duze chmury LiDAR:
- kontroluj VRAM przez `--lidar_max_points`.

4. Sciezka do PLY:
- moze byc absolutna albo wzgledna do `--data_dir`.

## 7) Przyklad komendy (uzytej w praktyce)

```bash
cd ~/gsplat/examples
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
/home/piotr-pawlus/gsplat/gsplat_env/bin/python simple_trainer_z_lidar.py default \
  --data_dir /home/piotr-pawlus/gsplat/examples/data/atlas \
  --no-normalize-world-space \
  --lidar_ply lidar/atlas_wyciete_NAJLEPSZE_SKALA.ply \
  --lidar_max_points 300000 \
  --data_factor 4 \
  --packed \
  --max_steps 30000 \
  --save_steps 7000 30000 \
  --eval_steps 1000000 \
  --result_dir /home/piotr-pawlus/gsplat/examples/results/atlas_lidar_init_df4
```

## 8) Efekt koncowy architektury

Architektura po zmianie:
- POZY KAMER: z COLMAP,
- INICJALNE PUNKTY GAUSSOW: z LiDAR po ICP,
- RESZTA TRENINGU: identyczna jak w standardowym `simple_trainer.py`.

To jest minimalnie inwazyjna i bezpieczna integracja LiDAR z istniejacym pipeline GSPLAT.