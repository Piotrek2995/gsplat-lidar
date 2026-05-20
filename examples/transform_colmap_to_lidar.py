"""
Transformacja modelu COLMAP (sparse/0/*.bin) do ukladu wspolrzednych LiDAR.

Stosuje similarity T = [s*R | t] do:
  - points3D.bin: P_new = (s*R) @ P_old + t
  - images.bin:   pozy world->camera przeliczane przez:
                  R_wc_new = R_wc @ R^T  (skala NIE wplywa na obrot)
                  t_wc_new = s * t_wc - R_wc_new @ t
  - cameras.bin:  intrynsyki niezmienione (similarity ich nie dotyka)

Uzycie:
    python transform_colmap_to_lidar.py \
        --in_dir  /sciezka/do/sparse/0 \
        --out_dir /sciezka/do/sparse/0_lidar \
        --transform transform_colmap_to_lidar.txt
"""
import argparse
import os
import struct
from pathlib import Path

import numpy as np


# COLMAP camera model_id -> liczba parametrow w bloku params
CAMERA_MODEL_NUM_PARAMS = {
    0: 3,    # SIMPLE_PINHOLE          (f, cx, cy)
    1: 4,    # PINHOLE                 (fx, fy, cx, cy)
    2: 4,    # SIMPLE_RADIAL           (f, cx, cy, k)
    3: 5,    # RADIAL                  (f, cx, cy, k1, k2)
    4: 8,    # OPENCV                  (fx, fy, cx, cy, k1, k2, p1, p2)
    5: 8,    # OPENCV_FISHEYE
    6: 12,   # FULL_OPENCV
    7: 5,    # FOV
    8: 4,    # SIMPLE_RADIAL_FISHEYE
    9: 5,    # RADIAL_FISHEYE
    10: 12,  # THIN_PRISM_FISHEYE
}


# ---------- COLMAP binary I/O ----------

def _read(f, n, fmt):
    return struct.unpack(fmt, f.read(n))


def read_cameras_binary(path):
    cameras = {}
    with open(path, "rb") as f:
        n = _read(f, 8, "<Q")[0]
        for _ in range(n):
            cam_id, model_id, w, h = _read(f, 24, "<iiQQ")
            num_params = CAMERA_MODEL_NUM_PARAMS[model_id]
            params = _read(f, 8 * num_params, "<" + "d" * num_params)
            cameras[cam_id] = {
                "id": cam_id, "model_id": model_id,
                "width": w, "height": h,
                "params": np.asarray(params, dtype=np.float64),
            }
    return cameras


def write_cameras_binary(cameras, path):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(cameras)))
        for c in cameras.values():
            f.write(struct.pack("<iiQQ", c["id"], c["model_id"],
                                c["width"], c["height"]))
            f.write(struct.pack("<" + "d" * len(c["params"]), *c["params"]))


def read_images_binary(path):
    images = {}
    with open(path, "rb") as f:
        n = _read(f, 8, "<Q")[0]
        for _ in range(n):
            image_id = _read(f, 4, "<i")[0]
            qvec = _read(f, 32, "<dddd")        # qw, qx, qy, qz
            tvec = _read(f, 24, "<ddd")
            camera_id = _read(f, 4, "<i")[0]
            name = b""
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name += ch
            num_pts = _read(f, 8, "<Q")[0]
            raw = _read(f, 24 * num_pts, "<" + "ddq" * num_pts) if num_pts else ()
            xys = np.array([(raw[3*i], raw[3*i+1]) for i in range(num_pts)],
                           dtype=np.float64).reshape(-1, 2)
            point3D_ids = np.array([raw[3*i+2] for i in range(num_pts)],
                                   dtype=np.int64)
            images[image_id] = {
                "id": image_id,
                "qvec": np.asarray(qvec, dtype=np.float64),
                "tvec": np.asarray(tvec, dtype=np.float64),
                "camera_id": camera_id,
                "name": name.decode("utf-8"),
                "xys": xys,
                "point3D_ids": point3D_ids,
            }
    return images


def write_images_binary(images, path):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(images)))
        for im in images.values():
            f.write(struct.pack("<i", im["id"]))
            f.write(struct.pack("<dddd", *im["qvec"]))
            f.write(struct.pack("<ddd", *im["tvec"]))
            f.write(struct.pack("<i", im["camera_id"]))
            f.write(im["name"].encode("utf-8") + b"\x00")
            f.write(struct.pack("<Q", len(im["xys"])))
            for (x, y), pid in zip(im["xys"], im["point3D_ids"]):
                f.write(struct.pack("<ddq", float(x), float(y), int(pid)))


def read_points3D_binary(path):
    points = {}
    with open(path, "rb") as f:
        n = _read(f, 8, "<Q")[0]
        for _ in range(n):
            pid = _read(f, 8, "<Q")[0]
            xyz = _read(f, 24, "<ddd")
            rgb = _read(f, 3, "<BBB")
            err = _read(f, 8, "<d")[0]
            tlen = _read(f, 8, "<Q")[0]
            traw = _read(f, 8 * tlen, "<" + "ii" * tlen) if tlen else ()
            image_ids = np.array([traw[2*i] for i in range(tlen)], dtype=np.int32)
            point2D_idxs = np.array([traw[2*i+1] for i in range(tlen)], dtype=np.int32)
            points[pid] = {
                "id": pid,
                "xyz": np.asarray(xyz, dtype=np.float64),
                "rgb": np.asarray(rgb, dtype=np.uint8),
                "error": err,
                "image_ids": image_ids,
                "point2D_idxs": point2D_idxs,
            }
    return points


def write_points3D_binary(points, path):
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(points)))
        for p in points.values():
            f.write(struct.pack("<Q", p["id"]))
            f.write(struct.pack("<ddd", *p["xyz"]))
            f.write(struct.pack("<BBB", *p["rgb"]))
            f.write(struct.pack("<d", float(p["error"])))
            f.write(struct.pack("<Q", len(p["image_ids"])))
            for img_id, p2d in zip(p["image_ids"], p["point2D_idxs"]):
                f.write(struct.pack("<ii", int(img_id), int(p2d)))


# ---------- Quaterniony (konwencja COLMAP: qw, qx, qy, qz) ----------

def quat_to_rot(qvec):
    qw, qx, qy, qz = qvec
    return np.array([
        [1 - 2*qy*qy - 2*qz*qz,   2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw,       1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw,       2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy],
    ])


def rot_to_quat(R):
    """Shepperd's method - stabilna konwersja R -> kwaternion (qw, qx, qy, qz)."""
    tr = np.trace(R)
    if tr > 0:
        S = 2.0 * np.sqrt(tr + 1.0)
        qw = 0.25 * S
        qx = (R[2, 1] - R[1, 2]) / S
        qy = (R[0, 2] - R[2, 0]) / S
        qz = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / S
        qx = 0.25 * S
        qy = (R[0, 1] + R[1, 0]) / S
        qz = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / S
        qx = (R[0, 1] + R[1, 0]) / S
        qy = 0.25 * S
        qz = (R[1, 2] + R[2, 1]) / S
    else:
        S = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / S
        qx = (R[0, 2] + R[2, 0]) / S
        qy = (R[1, 2] + R[2, 1]) / S
        qz = 0.25 * S
    q = np.array([qw, qx, qy, qz])
    return q / np.linalg.norm(q)


# ---------- Glowna logika transformacji ----------

def transform_model(in_dir: str, out_dir: str, T_path: str) -> None:
    T = np.loadtxt(T_path)
    assert T.shape == (4, 4), f"Macierz musi byc 4x4, dostalem {T.shape}"

    M = T[:3, :3]    # s * R
    t = T[:3, 3]

    # Dekompozycja M = s*R (CloudCompare zwraca te dwie rzeczy sklejone)
    col_norms = np.linalg.norm(M, axis=0)
    s = float(col_norms.mean())
    R = M / s
    det = float(np.linalg.det(R))

    print("=== Wczytana transformacja COLMAP -> LiDAR ===")
    print(f"  s        = {s:.6f}")
    print(f"  det(R)   = {det:+.6f}  {'(OK)' if det > 0.99 else '(!!! ODBICIE)'}")
    print(f"  spread   = {col_norms.max() - col_norms.min():.2e}")
    print(f"  t        = {t}")
    if det < 0.99:
        raise SystemExit("Det(R) niewlasciwy. Sprawdz alignment w CloudCompare.")

    # --- Wczytaj model ---
    print("\nWczytuje COLMAP...")
    cameras = read_cameras_binary(os.path.join(in_dir, "cameras.bin"))
    images  = read_images_binary(os.path.join(in_dir, "images.bin"))
    points  = read_points3D_binary(os.path.join(in_dir, "points3D.bin"))
    print(f"  {len(cameras)} cameras, {len(images)} images, {len(points)} points")

    # --- Transformuj punkty (wektoryzacja dla szybkosci) ---
    print("\nTransformuje points3D...")
    keys = list(points.keys())
    xyz_old = np.stack([points[k]["xyz"] for k in keys])         # (N, 3)
    xyz_new = xyz_old @ M.T + t                                   # (N, 3)
    for k, p in zip(keys, xyz_new):
        points[k]["xyz"] = p
    # error w COLMAP to blad reprojekcji w pikselach -> niezmienny

    # --- Transformuj pozy kamer (sanity check na pierwszej) ---
    print("Transformuje pozy kamer...")
    first_id = next(iter(images))
    R_wc0 = quat_to_rot(images[first_id]["qvec"])
    t_wc0 = images[first_id]["tvec"].copy()
    C_old0 = -R_wc0.T @ t_wc0

    for im in images.values():
        R_wc = quat_to_rot(im["qvec"])
        t_wc = im["tvec"]
        R_wc_new = R_wc @ R.T
        t_wc_new = s * t_wc - R_wc_new @ t
        im["qvec"] = rot_to_quat(R_wc_new)
        im["tvec"] = t_wc_new

    # Sanity check: srodek 1. kamery w nowym ukladzie musi sie zgadzac z s*R*C+t
    R_wc_new0 = quat_to_rot(images[first_id]["qvec"])
    t_wc_new0 = images[first_id]["tvec"]
    C_new_from_pose = -R_wc_new0.T @ t_wc_new0
    C_new_expected = s * (R @ C_old0) + t
    err = np.linalg.norm(C_new_from_pose - C_new_expected)
    print(f"  sanity check srodka 1. kamery: blad = {err:.2e}  "
          f"{'(OK)' if err < 1e-6 else '(!!! niezgodnosc)'}")

    # --- Zapisz ---
    print(f"\nZapisuje do {out_dir}/")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    write_cameras_binary(cameras, os.path.join(out_dir, "cameras.bin"))
    write_images_binary(images,   os.path.join(out_dir, "images.bin"))
    write_points3D_binary(points, os.path.join(out_dir, "points3D.bin"))

    # --- Podsumowanie ---
    pts_arr = np.stack([p["xyz"] for p in points.values()])
    print("\n=== Wymiary modelu po transformacji (powinny sie zgadzac z LiDAR-em) ===")
    print(f"  bbox min: {pts_arr.min(axis=0)}")
    print(f"  bbox max: {pts_arr.max(axis=0)}")
    print(f"  rozmiar:  {pts_arr.max(axis=0) - pts_arr.min(axis=0)}  (jednostki LiDAR)")
    print("\nGotowe.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--in_dir", required=True,
                    help="folder z oryginalnym sparse/0 (cameras.bin, images.bin, points3D.bin)")
    ap.add_argument("--out_dir", required=True,
                    help="folder docelowy, np. sparse/0_lidar")
    ap.add_argument("--transform", required=True,
                    help="plik .txt z macierza 4x4 COLMAP->LiDAR")
    args = ap.parse_args()
    transform_model(args.in_dir, args.out_dir, args.transform)
