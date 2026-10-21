#!/usr/bin/env python3
"""
prepare_data.py - build the binary particle files used by every implementation.

Binary format (native endian): int32 N, then N x (m, x, y, z, vx, vy, vz) as float32.

Sub-commands
------------
  csv        Star Cluster Simulations snapshot (x,y,z,vx,vy,vz,m,id; 64,000 stars)
             -> data/star_<N>.bin for each requested N (random subsample, fixed seed).
  princeton  Princeton COS 126 files (planets.txt, galaxy10k.txt, ...)
             rows are "px py vx vy mass" (2-D), so z = vz = 0.   -> data/<name>.bin
  random     Uniform random particles (development / correctness testing).
             -> data/random_<N>.bin
  info       Print a summary of a .bin file.

Examples
--------
  python3 prepare_data.py csv c_0000.csv --sizes 1000 5000 10000 20000 32000 64000
  python3 prepare_data.py princeton galaxy10k.txt galaxy20k.txt galaxy30k.txt
  python3 prepare_data.py random --sizes 100 1000 2000
  python3 prepare_data.py info data/star_1000.bin
"""
import argparse
import os
import sys

import numpy as np

DTYPE = np.dtype("<f4")


def write_bin(path, arr):
    """arr: (N, 7) array with columns m, x, y, z, vx, vy, vz."""
    arr = np.ascontiguousarray(arr, dtype=DTYPE)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "wb") as f:
        np.array([arr.shape[0]], dtype="<i4").tofile(f)
        arr.tofile(f)
    print(f"wrote {path}: N={arr.shape[0]}, {os.path.getsize(path)/1e6:.2f} MB")


def read_bin(path):
    with open(path, "rb") as f:
        n = int(np.fromfile(f, dtype="<i4", count=1)[0])
        arr = np.fromfile(f, dtype=DTYPE, count=7 * n).reshape(n, 7)
    return arr


def cmd_csv(a):
    import pandas as pd

    df = pd.read_csv(a.input)
    need = ["x", "y", "z", "vx", "vy", "vz", "m"]
    missing = [c for c in need if c not in df.columns]
    if missing:
        sys.exit(f"missing columns in CSV: {missing}")
    data = df[["m", "x", "y", "z", "vx", "vy", "vz"]].to_numpy(dtype=np.float64)
    if not np.isfinite(data).all():
        sys.exit("CSV contains NaN/inf values")
    total = len(data)
    print(f"read {total} stars; total mass = {data[:, 0].sum():.6f}")
    rng = np.random.default_rng(a.seed)
    for n in a.sizes:
        if n > total:
            print(f"skip N={n} (> {total} available)")
            continue
        if n == total:
            sub = data.copy()
        else:
            idx = np.sort(rng.choice(total, size=n, replace=False))
            sub = data[idx].copy()
            if not a.keep_mass:
                # keep total mass = 1 (N-body units) so the subsample stays a sensible cluster
                sub[:, 0] *= data[:, 0].sum() / sub[:, 0].sum()
        write_bin(os.path.join(a.outdir, f"star_{n}.bin"), sub)


def cmd_princeton(a):
    for path in a.inputs:
        with open(path) as f:
            tokens = f.read().split()
        n = int(tokens[0])
        # tokens[1] = universe radius, then n rows of 5 numbers
        rows = np.array(tokens[2:2 + 5 * n], dtype=np.float64).reshape(n, 5)
        px, py, vx, vy, m = rows.T
        z = np.zeros(n)
        arr = np.stack([m, px, py, z, vx, vy, z], axis=1)
        name = os.path.splitext(os.path.basename(path))[0]
        write_bin(os.path.join(a.outdir, f"{name}.bin"), arr)


def cmd_random(a):
    for n in a.sizes:
        rng = np.random.default_rng(a.seed)
        arr = np.empty((n, 7))
        arr[:, 0] = 1.0 / n
        arr[:, 1:4] = rng.uniform(-1, 1, size=(n, 3))
        arr[:, 4:7] = rng.uniform(-0.5, 0.5, size=(n, 3))
        write_bin(os.path.join(a.outdir, f"random_{n}.bin"), arr)


def cmd_info(a):
    arr = read_bin(a.input)
    print(f"N = {len(arr)}")
    for k, name in enumerate(["m", "x", "y", "z", "vx", "vy", "vz"]):
        col = arr[:, k]
        print(f"  {name:>2}: min={col.min(): .4e}  max={col.max(): .4e}  mean={col.mean(): .4e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("csv")
    p.add_argument("input")
    p.add_argument("--sizes", type=int, nargs="+", default=[1000, 5000, 10000, 20000, 32000, 64000])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--keep-mass", action="store_true", help="do not renormalise total mass of subsamples to the full-cluster mass")
    p.add_argument("--outdir", default="data")
    p.set_defaults(fn=cmd_csv)

    p = sub.add_parser("princeton")
    p.add_argument("inputs", nargs="+")
    p.add_argument("--outdir", default="data")
    p.set_defaults(fn=cmd_princeton)

    p = sub.add_parser("random")
    p.add_argument("--sizes", type=int, nargs="+", default=[100, 1000, 2000])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="data")
    p.set_defaults(fn=cmd_random)

    p = sub.add_parser("info")
    p.add_argument("input")
    p.set_defaults(fn=cmd_info)

    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
