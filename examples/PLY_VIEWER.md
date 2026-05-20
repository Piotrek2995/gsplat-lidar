# PLY editor (`ply_editor.py`)

Interaktywny webowy edytor scen Gaussian Splatting w formacie PLY.
Pozwala na wycinanie floaterów / artefaktów przez box-selection
i suwaki opacity/scale, z możliwością cofania i zapisu do nowego PLY.

Plik źródłowy: `examples/ply_editor.py`.

---

## 1. Jak zamienić checkpoint `.pt` na `.ply`

`ply_editor.py` przyjmuje **tylko PLY** — checkpointy treningowe (`.pt`)
trzeba najpierw wyeksportować. Są trzy sposoby.

### A. Z trenera (najczystsze — dostajesz PLY automatycznie podczas treningu)

`simple_trainer_z_lidar.py` ma flagi `--save_ply` + `--ply_steps`. Domyślnie
PLY są zapisywane do `<result_dir>/ply/point_cloud_<step>.ply`. Jeśli
trenujesz od nowa, wystarczy:

```bash
python simple_trainer_z_lidar.py default \
    --data_dir data/atlas \
    --lidar_ply atlas_wyciete_local.ply \
    --no-normalize-world-space \
    --result_dir results/atlas_v3_reg \
    --save_ply \
    --ply_steps 7000 30000 \
    ...
```

Po treningu PLY są w `results/atlas_v3_reg/ply/`.

### B. Z `clean_ckpt.py` (jednoczesny eksport + filtr widoczności/opacity)

Skrypt `clean_ckpt.py` używaliśmy do post-hoc czyszczenia — przy okazji
zapisuje czysty PLY. Jeśli chcesz tylko skonwertować bez filtrowania,
wyłącz wszystkie filtry:

```bash
python clean_ckpt.py \
    --ckpt results/atlas_v3_reg/ckpts/ckpt_29999_rank0.pt \
    --data-dir data/atlas \
    --data-factor 2 \
    --no-normalize-world-space \
    --no-visibility --no-opacity --no-scale \
    --output results/atlas_v3_reg/ply_raw
```

Wynik: `results/atlas_v3_reg/ply_raw/clean.ply` (cały zbiór 1:1).

### C. Inline (gdy chcesz tylko jednorazowo z linii poleceń)

```bash
python - <<'EOF'
import torch
from gsplat import export_splats

ckpt = torch.load("results/atlas_v3_reg/ckpts/ckpt_29999_rank0.pt",
                  map_location="cuda")["splats"]
export_splats(
    means=ckpt["means"],
    scales=ckpt["scales"],
    quats=ckpt["quats"],
    opacities=ckpt["opacities"].flatten(),
    sh0=ckpt["sh0"],
    shN=ckpt["shN"],
    format="ply",
    save_to="results/atlas_v3_reg/converted.ply",
)
print("done")
EOF
```

---

## 2. Jak odpalić edytor

```bash
cd /home/piotr-pawlus/gsplat/examples
/home/piotr-pawlus/gsplat/gsplat_env/bin/python ply_editor.py \
    --ply  results/atlas_v3_reg/clean/clean.ply \
    --output_dir results/atlas_v3_reg/edited \
    --port 8088
```

Argumenty:

| flaga          | wymagana | opis |
|----------------|----------|------|
| `--ply`        | tak      | wejściowy PLY (format gsplat) |
| `--output_dir` | tak      | gdzie zapisać `edited_*.ply` |
| `--port`       | nie      | port HTTP serwera viser (default `8088`) |
| `--device`     | nie      | `cuda` / `cpu` (default `cuda`) |

Po starcie otwórz `http://localhost:8088` w przeglądarce.

---

## 3. Interfejs

Cały edytor jest w panelu **Edit** w prawym GUI viewera.

### Live thresholds (filtry na żywo — niedestrukcyjne)
- **opacity min** (0–1) — chowa splaty z `sigmoid(opacity) < próg`.
- **scale max (×radius)** (0.001–2.0) — chowa splaty których największy
  scale przekracza `próg × scene_radius`. Wartość `2.0` = wyłączone.
- **Bake thresholds → mask** — bierze aktualne suwaki i wpisuje je
  trwale do aktywnej maski. Suwaki resetują się do neutralnych.

Suwaki **nie usuwają** splatów dopóki nie klikniesz Bake — możesz
swobodnie eksperymentować i cofać przyciskiem.

### Box selection
- Żółta drutowa skrzynka z uchwytami 3D w środku sceny:
  - strzałki = translacja
  - pierścienie = rotacja
- **width / height / depth** — numeryczne pola dla rozmiaru skrzynki.
- **Show box** — chowa/pokazuje skrzynkę (sama operacja Delete dalej
  używa ostatniego stanu).
- **Center box on scene** — wraca do środka sceny + identycznej rotacji.
- **Delete INSIDE box** — usuwa splaty wewnątrz.
- **Delete OUTSIDE box** — usuwa splaty na zewnątrz (przydatne do crop
  do obiektu).

### Camera
- **Focus on scene center** — ustawia orbit-target na środek sceny
  i odsuwa kamerę.
- **Focus on box** — to samo, ale względem pozycji skrzynki (przydatne
  do podlecenia do konkretnego floatera).
- **distance (×radius)** — jak daleko ma być kamera od targetu.

### Actions
- **Undo last** — cofa ostatnią destrukcyjną operację (historia 30
  kroków: Delete IN/OUT, Bake, Reset).
- **Reset mask** — przywraca wszystkie splaty.
- **Save .ply** — zapisuje aktualnie *widoczne* splaty (active mask +
  live thresholds) do `<output_dir>/edited_<unix_timestamp>.ply`.

---

## 4. Sugerowany workflow

1. **Wstępne sito sliderami.** Podnieś `opacity min` do 0.02–0.05 —
   zobaczysz na żywo, ile floaterów zniknie. Jeśli detale obiektu
   zostają nienaruszone → kliknij **Bake**.
2. **Crop sceny.** Otocz cały obiekt sporą skrzynką → **Delete OUTSIDE**.
   To wytnie wszystkie syfy w dalekim tle / niebie.
3. **Punktowe czyszczenie.** Znajdź pojedyncze floatery, ustaw skrzynkę
   na nich (Focus on box pomaga), **Delete INSIDE**. Powtarzaj.
4. **Save .ply.** Wynik leci do `<output_dir>/edited_<timestamp>.ply`.
   Możesz go wczytać z powrotem do tego samego edytora (`--ply ...`)
   i kontynuować.

---

## 5. Jak to działa (pod maską)

### Wczytywanie PLY (`load_gsplat_ply`)

Parser czyta header tekstowy, sprawdza że plik jest
`binary_little_endian` i że wszystkie pola są `float32`. Każdy splat
to flat-record `N × 4 bajty × liczba_pól`. Z headera wyciąga mapę
`property name → kolumna`, potem ciągnie kolumny po nazwie:

- `x, y, z` → `means (N,3)`
- `f_dc_0..2` → `sh0 (N,1,3)`
- `f_rest_*` → channel-major flat → reshape `(N, 3, K-1)` → transpose
  do `(N, K-1, 3)` (odwrotność tego co robi `gsplat.export_splats`)
- `opacity` → `opacities (N,)` *(raw, pre-sigmoid)*
- `scale_0..2` → `scales (N,3)` *(raw, pre-exp)*
- `rot_0..3` → `quats (N,4)` *(raw, pre-normalize)*

### Stan edytora (`EditorState`)

Trzyma dwa zestawy tensorów:
- **`raw_*`** — wartości 1:1 z PLY (pre-aktywacja). Te są używane przy
  zapisie do PLY, żeby plik wynikowy był identycznego formatu jak
  wejściowy (poza usuniętymi splatami).
- **`means/quats/scales/opacities/colors`** — wartości po aktywacji
  (`sigmoid`, `exp`, `F.normalize`, concat sh0+shN). Używane do
  rasteryzacji.

Plus:
- `active: bool[N]` — maska "który splat zostawiamy" (modyfikowana
  przez Delete/Bake/Reset/Undo).
- `history: list[Tensor]` — stack snapshotów maski przed każdą operacją
  (max 30).
- `opacity_min`, `scale_max` — live thresholds (nie modyfikują `active`).

### `view_mask()`

Każda klatka liczy `active & opacity_filter & scale_filter`. Tylko
splaty z `True` lecą do rasteryzacji.

### Box-test (`points_in_box`)

Dla zadanej skrzynki: pozycja `p`, kwaternion `wxyz` (Hamilton, world
from local), rozmiar `(w,h,d)`. Robi:

```python
R = quat_to_rotmat(wxyz)               # world-from-local
local = (means - p) @ R                # world-to-local (row vectors)
inside = (|local| <= half_size).all(-1)
```

To jest klasyczny test OBB.

### Rendering

`viewer_render_fn` dostaje camera_state z viewera (nerfview), buduje
`viewmat = c2w⁻¹`, woła `gsplat.rasterization(...)` z aktualnie
zamaskowanymi tensorami. Wszystko po stronie GPU serwera — klient
dostaje gotowy obraz przez websocket. Klient nie liczy żadnego
splattingu (więc duże modele nie zabijają przeglądarki).

### Zapis

`Save .ply` używa `gsplat.export_splats(format="ply")` przekazując
**raw** tensory zindeksowane aktualnym `view_mask()`. Plik wyjściowy
ma identyczny format jak wejście — można go z powrotem wczytać
edytorem albo dowolnym viewerem.

### Bezpieczeństwo (safe callbacks)

Każdy callback GUI jest obudowany dekoratorem `@safe`. Jeśli któryś
rzuci wyjątek — leci traceback do stderr serwera, a viewer żyje
dalej. Bez tego pojedynczy błąd (np. odwołanie do usuniętej skrzynki)
zabijał cały serwer.

---

## 6. Najczęstsze potknięcia

- **„Loaded 0 splats" / pusty render** — PLY nie jest w formacie gsplat
  (np. plik LiDAR-owy z punktami, nie splatami). Edytor wczyta tylko
  PLY mający pola `f_dc_*`, `opacity`, `scale_*`, `rot_*`.
- **Port zajęty** — coś już słucha na 8088. Daj inny `--port 8089`
  albo ubij stary proces: `pkill -f ply_editor.py`.
- **Brak GUI po `Save .ply`** — to normalne, klik tylko loguje na konsoli
  serwera (`Wrote ... (N splats)`). Sprawdź `<output_dir>` i log w terminalu.
- **Cały obiekt zniknął po podniesieniu opacity** — `sigmoid(opacity)`
  w typowych checkpointach po treningu plasuje się głównie 0.01–0.5;
  progi > 0.3 obetną sporą część sceny. Zacznij od 0.02–0.05.
- **Skrzynka skacze w nieoczekiwane miejsce po rotacji** — uchwyty
  obrotu działają względem aktualnej orientacji TC. Klik **Center box
  on scene** zresetuje rotację do `(1,0,0,0)`.
