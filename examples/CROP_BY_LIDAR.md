# Crop gaussianów wg chmury LiDAR (`crop_by_lidar.py`)

Narzędzie do czyszczenia trenowanej sceny Gaussian Splatting przez
przefiltrowanie splatów względem **referencyjnej chmury punktów** (typowo
LiDAR-owej, wyciętej w CloudCompare do samego obiektu). Pozwala wywalić
floatery i syf z tła bez ręcznego klikania w edytorze.

Plik źródłowy: `examples/crop_by_lidar.py`.

---

## 1. Co robi

Bierze dwa pliki PLY:

| plik | rola |
|------|------|
| `--reference` | **chmura referencyjna** (LiDAR, CloudCompare itp.) — wyznacza "gdzie naprawdę jest obiekt" |
| `--splats`    | **wytrenowana scena gsplat** (`.ply` z trenera lub z `clean_ckpt.py`) |

I zapisuje nowy PLY (`--output`) z **podzbiorem splatów** spełniających
wybrane kryterium przestrzenne.

Dwa tryby filtrowania:

- **`aabb`** — Axis-Aligned Bounding Box. Liczy prostopadłościan wokół
  chmury referencyjnej, zostawia splaty wewnątrz. Szybkie, zachowawcze,
  nie wytnie syfu który leży *w obrębie* bryły obiektu.
- **`distance`** — dla każdego splatu liczy odległość do najbliższego
  punktu chmury referencyjnej (KD-tree). Zostawia splaty bliższe niż
  `--max_distance`. Wycina obiekt **w dokładnym jego kształcie**.

---

## 2. Wymagania

- PLY referencyjny musi mieć właściwości `x`, `y`, `z` (typu `float`
  lub `double`). RGB / scalary mogą być dowolne — parser je zignoruje.
- PLY ze splatami musi być w formacie gsplat (output `gsplat.export_splats`):
  pola `x,y,z`, `f_dc_*`, `f_rest_*` (opcjonalne), `opacity`, `scale_*`,
  `rot_*`, wszystkie `float32`.
- **Te same współrzędne!** Obie chmury muszą być w tym samym układzie
  (ten sam świat, ta sama skala). Jeśli scena trenowała się z
  `--no-normalize-world-space` i lidarowym `--lidar_ply` — to już jest.
  W przeciwnym wypadku najpierw wyrównaj chmurę referencyjną do układu
  rekonstrukcji (np. similarity transform / ICP), bo inaczej dostaniesz
  pustą maskę.
- Tryb `distance` wymaga `scipy` (już zainstalowane w środowisku).

---

## 3. Tryb `aabb` — szybki bbox crop

```bash
python crop_by_lidar.py --mode aabb \
    --reference atlas_wyciete_local.ply \
    --splats   results/atlas_v3_reg/clean/clean.ply \
    --output   results/atlas_v3_reg/clean/clean_cropped.ply \
    --pad_pct  0.05
```

Flagi specyficzne dla `aabb`:

| flaga | opis |
|-------|------|
| `--pad`     | absolutny margines (w jednostkach sceny) dodany do każdej ściany |
| `--pad_pct` | margines jako procent rozmiaru bboxa, **per oś**. `0.05` = +5% w każdą stronę |

Margines dodaje się: najpierw `--pad`, potem `--pad_pct * size_axis`.

### Kiedy używać AABB

- Chcesz **szybko** odfiltrować daleki bałagan (sky, sąsiednie budynki).
- Obiekt jest mniej-więcej prostopadłościenny, masz mało floaterów w
  bezpośrednim otoczeniu.
- Chcesz **konserwatywnego** wyniku — AABB nigdy nie obetnie żadnej
  części obiektu (o ile bbox ref obejmuje cały obiekt).

### Przykładowy wynik

```
AABB min: [-7.47, 13.03,  0.71]
AABB max: [-1.54, 18.22,  2.56]
AABB size: [5.93, 5.18, 1.85]  (pad=0.0+5.0%)
inside: 576728 / 599168 (96.3%)
```

→ wywaliło 22k splatów z dalekiego tła. Reszta floaterów (te bliżej
obiektu, *wewnątrz* bboxa) zostaje.

---

## 4. Tryb `distance` — crop wg dokładnego kształtu

```bash
python crop_by_lidar.py --mode distance \
    --reference atlas_wyciete_local.ply \
    --splats   results/atlas_v3_reg/clean/clean.ply \
    --output   results/atlas_v3_reg/clean/clean_shape.ply \
    --max_distance 0.10
```

Flagi specyficzne dla `distance`:

| flaga | opis |
|-------|------|
| `--max_distance`   | maksymalna dopuszczalna odległość splatu od najbliższego punktu lidaru |
| `--ref_downsample` | użyj co N-tego punktu referencyjnego (1 = wszystkie). Przyspiesza, kosztem precyzji |

### Jak dobrać próg

Skrypt po przetworzeniu wypisuje statystyki odległości — patrz na nie
zanim zdecydujesz:

```
distance stats: min=0.0000, median=0.0133, p95=0.6890, max=3.8116
```

Tu median 1.3 cm = większość splatów siedzi tuż na powierzchni lidaru.
p95 = 69 cm = piąty percentyl to wyraźne floatery. Próg dobierasz tak,
żeby siedzieć **powyżej szumu pomiarowego** (≥ kilka cm), ale **poniżej
floaterów** (≪ p95).

Heurystyka dla naszej sceny atlas:

| `--max_distance` | efekt |
|------------------|-------|
| `0.03` (3 cm)    | bardzo agresywny — tylko splaty dokładnie na powierzchni; ryzyko dziur |
| `0.05` (5 cm)    | agresywny, dobre dla czystego obiektu z dużą gęstością lidaru |
| **`0.10` (10 cm)** | **bezpieczny default** |
| `0.20` (20 cm)   | luźny — zostawi sporo szumu blisko powierzchni |

### Kiedy używać distance

- Obiekt o **nietrywialnym kształcie** (zaokrąglony, organiczny, cienki).
- W AABB zostaje za dużo floaterów wewnątrz bboxa.
- Masz dużą gęstość lidaru (~kilka cm między punktami) → możesz odważnie
  obniżać próg.

### Przykładowy wynik

```
inside: 481545 / 599168 (80.4%)
done — 113.6 MB, 481545 splats
```

→ wycięło **5× więcej** floaterów niż AABB (117k vs 22k). To te splaty
które były w bboxie, ale nie miały żadnego punktu lidaru w promieniu 10 cm.

---

## 5. Pełna lista flag

```
--mode {aabb,distance}   tryb filtrowania (default: aabb)
--reference PATH         PLY referencyjny (LiDAR / inna chmura)              [wymagane]
--splats    PATH         PLY ze splatami gsplat                              [wymagane]
--output    PATH         gdzie zapisać wynik                                 [wymagane]
--pad         FLOAT      [aabb] margines absolutny (default 0.0)
--pad_pct     FLOAT      [aabb] margines jako frakcja rozmiaru (default 0.0)
--max_distance FLOAT     [distance] maks. odległość w jednostkach sceny (default 0.10)
--ref_downsample INT     [distance] użyj co N-tego pkt referencyjnego (default 1)
```

---

## 6. Zalecany workflow

1. **Najpierw `aabb`** z `--pad_pct 0.05` — szybki sanity check.
   Jeśli `inside %` > 90 — Twoje współrzędne są spójne. Jeśli < 5% —
   chmury są w innych układach, sprawdź wyrównanie.
2. **Potem `distance`** z `--max_distance 0.10` na tym samym wejściu.
   Porównaj `_cropped.ply` (aabb) i `_shape.ply` (distance) w viewerze:
   ```bash
   python ply_view.py --ply results/atlas_v3_reg/clean/clean_shape.ply --port 8088
   ```
3. **Dostrój próg** jeśli trzeba. Patrz na wypisane `p95` / `max` —
   one mówią ile masz floaterów i jak daleko.
4. **Resztę floaterów** (tych blisko powierzchni których distance nie
   złapał) doczyść ręcznie w `ply_editor.py`:
   ```bash
   python ply_editor.py --ply results/atlas_v3_reg/clean/clean_shape.ply \
       --output_dir results/atlas_v3_reg/edited --port 8088
   ```

---

## 7. Jak to działa (pod maską)

### Parser PLY

Self-contained — bez `plyfile`. Czyta header tekstowy, buduje
`np.dtype` strukturalny per właściwość (`float32` / `float64` / `uint8`
/ pozostałe), wczytuje całą sekcję wierzchołków jednym `np.frombuffer`.
Obsługuje mix typów (np. lidarowe `x,y,z` w `double` + RGB `uchar` +
scalary `float`).

### Tryb AABB

```python
bb_min = ref_xyz.min(axis=0) - pad
bb_max = ref_xyz.max(axis=0) + pad
inside = ((means >= bb_min) & (means <= bb_max)).all(axis=1)
```

Liniowy w liczbie splatów, ms-y.

### Tryb distance

```python
tree = scipy.spatial.cKDTree(ref_xyz)
dists, _ = tree.query(means, k=1, workers=-1)
inside = dists <= max_distance
```

Drzewo: O(M log M) na 1.4 M punktów ≈ kilka sekund. Query: O(N log M)
parallel — kilka sekund dla 600 k splatów. Cały skrypt typowo < 30 s.

### Zapis

```python
gsplat.export_splats(means=splats["means"][inside], ..., format="ply")
```

Plik wyjściowy ma identyczny format jak wejście — można go od razu
otworzyć w `ply_view.py` / `ply_editor.py` albo w SuperSplat / WebGL
viewerze.

---

## 8. Najczęstsze problemy

- **`inside: 0 / N (0.0%)`** — chmury są w różnych układach
  współrzędnych. Sprawdź czy splaty wytrenowały się z lidarem
  (`--no-normalize-world-space --lidar_ply ...`). Jeśli nie — najpierw
  wyrównaj ref do układu rekonstrukcji (np. `transform_colmap_to_lidar.py`
  w tym repo).
- **`inside: 100%`** — twoja chmura referencyjna jest większa niż
  obszar splatów (np. wzięto pełny skan zamiast wycięcia obiektu).
  Powtórz z mniejszą referencją albo użyj większego `--pad` ze znakiem
  ujemnym (`--pad -0.5` = 50 cm marginesu *do wewnątrz*).
- **`distance` długo działa** — daj `--ref_downsample 4` albo `8`.
  Dla gęstego lidaru spadek jakości jest minimalny.
- **Dziury w obiekcie po `distance`** — `--max_distance` za niski.
  Podnieś o 50%, sprawdź wynik.
- **MemoryError przy KD-tree** — to się dzieje dla >10 M punktów
  referencyjnych. Zrób downsample chmury lidaru przed crop-em (np.
  CloudCompare → "Subsample" → "Random" lub "Octree").

---

## 9. Powiązane skrypty w tym katalogu

| skrypt | rola |
|--------|------|
| `clean_ckpt.py`            | post-hoc filtr (visibility + opacity + scale) z checkpointu treningowego → PLY |
| **`crop_by_lidar.py`**     | **filtr przestrzenny względem chmury referencyjnej (ten plik)** |
| `ply_view.py`              | read-only viewer dla gsplat PLY (bez edit GUI) |
| `ply_editor.py`            | interaktywny edytor (box select, slidery, save) |
| `simple_viewer.py`         | viewer dla `.pt` (raw checkpoint) — nie .ply |

Typowa pipeline: trening → `clean_ckpt.py` → `crop_by_lidar.py` →
`ply_editor.py` (doczyszczenie) → finalny `.ply`.
