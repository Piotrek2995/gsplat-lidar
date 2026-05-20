"""
Tworzy lokalna wersje atlas_wyciete.ply z odjetym global shift (0, 10000, 0).
Po tym zabiegu LiDAR pasuje do COLMAP-a po naszej transformacji 1:1.
"""
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


def remove_shift(src: str, dst: str, shift_xyz=(0.0, 10000.0, 0.0)):
    shift = np.asarray(shift_xyz, dtype=np.float64)

    with open(src, 'rb') as f:
        # zachowaj naglowek dokladnie taki jaki byl
        header_lines = []
        first = f.readline()
        header_lines.append(first)
        assert first.decode('ascii').strip() == 'ply'

        fmt = None
        n_vertices = None
        props = []
        in_vertex = False

        while True:
            raw = f.readline()
            header_lines.append(raw)
            line = raw.decode('ascii').strip()
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
                    props.append((tokens[2], PLY_DTYPE[tokens[1]]))

        assert fmt == 'binary_little_endian', f"Format: {fmt} - nie obslugiwany"

        struct_dtype = np.dtype([(name, '<' + dt) for name, dt in props])
        data = np.fromfile(f, dtype=struct_dtype, count=n_vertices)

    # Odejmij shift od XYZ
    print(f"Wczytano {len(data):,} wierzcholkow.")
    print(f"Przed shift: x mean={data['x'].mean():.4f}, "
          f"y mean={data['y'].mean():.4f}, z mean={data['z'].mean():.4f}")

    # Operacja w-place na strukturze - zachowuje typy
    data['x'] = (data['x'].astype(np.float64) - shift[0]).astype(data['x'].dtype)
    data['y'] = (data['y'].astype(np.float64) - shift[1]).astype(data['y'].dtype)
    data['z'] = (data['z'].astype(np.float64) - shift[2]).astype(data['z'].dtype)

    print(f"Po shift:    x mean={data['x'].mean():.4f}, "
          f"y mean={data['y'].mean():.4f}, z mean={data['z'].mean():.4f}")

    # Zapisz: ten sam naglowek + zmodyfikowane dane binarne
    with open(dst, 'wb') as f:
        for line in header_lines:
            f.write(line)
        data.tofile(f)

    print(f"\nZapisano: {dst}")


if __name__ == "__main__":
    SRC = "/home/piotr-pawlus/gsplat/examples/atlas_wyciete.ply"
    DST = "/home/piotr-pawlus/gsplat/examples/atlas_wyciete_local.ply"
    remove_shift(SRC, DST, shift_xyz=(0.0, 10000.0, 0.0))
