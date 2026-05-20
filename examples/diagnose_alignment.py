"""
Diagnostyka alignment-u (poprawiona): poprawnie parsuje rozne formaty PLY
niezaleznie od dodatkowych wlasciwosci (RGB, intensity itp.).
"""
import struct
import numpy as np
from pathlib import Path


PLY_DTYPE = {
    "char": "i1", "int8": "i1",
    "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2",
    "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4",
    "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4",
    "double": "f8", "float64": "f8",
}


def read_ply_xyz(path):
    with open(path, 'rb') as f:
        first = f.readline().decode('ascii').strip()
        assert first == 'ply', f"To nie jest plik PLY: {first!r}"

        fmt = None
        n_vertices = None
        props = []
        in_vertex = False

        while True:
            line = f.readline().decode('ascii').strip()
            if line == 'end_header':
                break
            if line.startswith('format'):
                fmt = line.split()[1]
            elif line.startswith('element'):
                tokens = line.split()
                in_vertex = (tokens[1] == 'vertex')
                if in_vertex:
                    n_vertices = int(tokens[2])
            elif line.startswith('property') and in_vertex:
                tokens = line.split()
                if len(tokens) == 3:
                    ptype, pname = tokens[1], tokens[2]
                    props.append((pname, PLY_DTYPE[ptype]))

        assert fmt == 'binary_little_endian', f"Nieobslugiwany format: {fmt}"

        struct_dtype = np.dtype([(name, '<' + dt) for name, dt in props])
        data = np.fromfile(f, dtype=struct_dtype, count=n_vertices)
        xyz = np.stack([data['x'], data['y'], data['z']], axis=1).astype(np.float64)
    return xyz


def read_colmap_images_bin(path):
    centers = []
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            f.read(4)
            qw, qx, qy, qz = struct.unpack('<dddd', f.read(32))
            tx, ty, tz = struct.unpack('<ddd', f.read(24))
            f.read(4)
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
            n_pts = struct.unpack('<Q', f.read(8))[0]
            f.read(24 * n_pts)
            R = np.array([
                [1 - 2*qy*qy - 2*qz*qz,   2*qx*qy - 2*qz*qw,     2*qx*qz + 2*qy*qw],
                [2*qx*qy + 2*qz*qw,       1 - 2*qx*qx - 2*qz*qz, 2*qy*qz - 2*qx*qw],
                [2*qx*qz - 2*qy*qw,       2*qy*qz + 2*qx*qw,     1 - 2*qx*qx - 2*qy*qy],
            ])
            t = np.array([tx, ty, tz])
            centers.append(-R.T @ t)
    return np.array(centers)


def read_colmap_points3d_bin(path):
    pts = []
    with open(path, 'rb') as f:
        n = struct.unpack('<Q', f.read(8))[0]
        for _ in range(n):
            f.read(8)
            xyz = struct.unpack('<ddd', f.read(24))
            f.read(11)
            tlen = struct.unpack('<Q', f.read(8))[0]
            f.read(8 * tlen)
            pts.append(xyz)
    return np.array(pts)


def show_bbox(name, pts):
    if pts is None or len(pts) == 0:
        print(f"\n{name}: BRAK DANYCH")
        return
    mn, mx = pts.min(0), pts.max(0)
    print(f"\n{name}:")
    print(f"  N points:  {len(pts):,}")
    print(f"  bbox min:  [{mn[0]:>14.4f} {mn[1]:>14.4f} {mn[2]:>14.4f}]")
    print(f"  bbox max:  [{mx[0]:>14.4f} {mx[1]:>14.4f} {mx[2]:>14.4f}]")
    print(f"  rozmiar:   [{mx[0]-mn[0]:>14.4f} {mx[1]-mn[1]:>14.4f} {mx[2]-mn[2]:>14.4f}]")
    print(f"  centroid:  [{pts.mean(0)[0]:>14.4f} {pts.mean(0)[1]:>14.4f} {pts.mean(0)[2]:>14.4f}]")


BASE = Path("/home/piotr-pawlus/gsplat/examples")
print("=" * 72)
print("DIAGNOSTYKA: porownanie ukladow LiDAR vs kamery COLMAP")
print("=" * 72)

# Wszystkie pliki PLY ktore moga byc ciekawe
candidates = [
    BASE / "atlas_wyciete_local.ply",
    BASE / "data" / "atlas" / "lidar" / "atlas_wyciete_NAJLEPSZE_SKALA.ply",
]
# dorzucam wszystkie *.ply z folderu lidar/
lidar_dir = BASE / "data" / "atlas" / "lidar"
if lidar_dir.exists():
    for p in lidar_dir.glob("*.ply"):
        if p not in candidates:
            candidates.append(p)

for path in candidates:
    if path.exists():
        try:
            pts = read_ply_xyz(str(path))
            show_bbox(f"PLY: {path}", pts)
        except Exception as e:
            print(f"\n!!! {path}: BLAD - {e}")
    else:
        print(f"\n(Brak: {path})")

# COLMAP po transformacji
SPARSE = BASE / "data" / "atlas" / "sparse" / "0"
if (SPARSE / "points3D.bin").exists():
    show_bbox("Punkty COLMAP (po transformacji do LiDAR)",
              read_colmap_points3d_bin(str(SPARSE / "points3D.bin")))
if (SPARSE / "images.bin").exists():
    show_bbox("Centra kamer COLMAP (po transformacji do LiDAR)",
              read_colmap_images_bin(str(SPARSE / "images.bin")))

# Bonus: stary sparse (przed transformacja)
SPARSE_OLD = BASE / "data" / "atlas" / "sparse" / "0_colmap_original"
if (SPARSE_OLD / "points3D.bin").exists():
    show_bbox("Punkty COLMAP (ORYGINALNE, przed transformacja)",
              read_colmap_points3d_bin(str(SPARSE_OLD / "points3D.bin")))
if (SPARSE_OLD / "images.bin").exists():
    show_bbox("Centra kamer COLMAP (ORYGINALNE, przed transformacja)",
              read_colmap_images_bin(str(SPARSE_OLD / "images.bin")))

print("\n" + "=" * 72)
print("Sprawdz ktory LiDAR ma centroid najblizszy centroidy kamer COLMAP")
print("po transformacji. Ten plik byl referencja w CloudCompare i powinien")
print("isc do --lidar_ply.")
print("=" * 72)
