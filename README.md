# Parallel N-Body Gravitational Simulation (Sequential / OpenMP / MPI / CUDA)

Direct O(N²) summation, N-body units (G = 1), softened gravity, symplectic Euler
(`v += a·dt`, then `r += v·dt`, with all accelerations computed from start-of-step positions).

## Files

| File | Purpose |
|---|---|
| `common.h` | Shared `Particle` struct, physics constants (`DT`, `EPS2`), file I/O, the force kernel used by all CPU versions |
| `nbody_seq.c` | Sequential baseline |
| `nbody_omp.c` | OpenMP: outer particle loop split over threads |
| `nbody_mpi.c` | MPI: block decomposition + `MPI_Allgatherv(MPI_IN_PLACE)` each step |
| `nbody_cuda.cu` | CUDA: one thread per particle, shared-memory tiling, configurable block size |
| `compare.c` | Correctness checker (max / RMS difference between two result files) |
| `prepare_data.py` | CSV / Princeton / random data → binary input files |
| `run_experiments.sh` | Full benchmark suite → `results/results.csv` |
| `analyze.py` | Tables (CSV + LaTeX) and figures (PNG + PDF) for the report |

## Quick start

```bash
make                       # builds what your machine supports (mpicc / nvcc auto-detected)
make cuda ARCH=sm_75       # e.g. Colab T4; use sm_86 / sm_89 / sm_80 for other GPUs

python3 prepare_data.py csv c_0000.csv --sizes 1000 5000 10000 20000 32000 64000
python3 prepare_data.py princeton planets.txt galaxy10k.txt galaxy20k.txt galaxy30k.txt   # optional extra datasets
python3 prepare_data.py random --sizes 100 1000                                            # debugging only

# single runs  (iterations = 10)
./nbody_seq  data/star_10000.bin 10 results/seq.bin
./nbody_omp  data/star_10000.bin 10 4 results/omp.bin          # 4 threads
mpirun -np 4 ./nbody_mpi data/star_10000.bin 10 results/mpi.bin
./nbody_cuda data/star_10000.bin 10 256 results/cuda.bin       # block size 256
./compare results/seq.bin results/omp.bin 1e-6                 # PASS/FAIL

# whole study, then report material
CSV=c_0000.csv ./run_experiments.sh
python3 analyze.py
```

`run_experiments.sh` knobs (environment variables): `ITERS`, `REPS`, `SIZES`, `SCALE_N`,
`UNITS_LIST` (threads/processes), `BLOCKS` (CUDA block sizes), `FIXED_UNITS`, `MAX_SEQ_N`.
On a machine with more cores use e.g. `UNITS_LIST="1 2 4 8 16"`.
Set `MAX_SEQ_N=64000` to also time the sequential code at 64,000 particles (slow: minutes).

## Binary format

`int32 N` followed by `N × {m, x, y, z, vx, vy, vz}` as `float32` (native endian).
The particle count comes from the file, so there is no separate N argument.

## Data notes

* The 64,000-star King-model snapshot is subsampled **at random (seed 42)**; subsample masses are rescaled
  so the total mass stays 1 (`--keep-mass` disables this). The full 64,000 is used unchanged.
* Princeton COS 126 files are 2-D (`px py vx vy mass`); they are converted with `z = vz = 0`.

## Mapping to the report (Mini Project Guidelines §3)

| Report section | Where it comes from |
|---|---|
| Experimental Setup | `results/system_info.txt` (CPU, GPU, compiler versions) |
| Results / Performance Analysis | `results/summary_*.csv|tex`, `figures/*.pdf` |
| Verification of correctness | `results/correctness.txt` |
| Demonstrates O(N²) | `results/complexity_fit.txt` (log-log slope ≈ 2) and `figures/time_vs_n` |
| Speedup, efficiency, scalability | `summary_scaling` (speedup, efficiency, Karp–Flatt serial fraction), `summary_size` |
| GPU configuration study | `summary_cuda_block`, `figures/cuda_block_size` |

## Measurement rules (so the comparison is fair)

* Only the simulation loop is timed. File I/O is excluded everywhere; CUDA host↔device transfer is reported
  separately (`aux_s` column).
* No output is written inside the iteration loop.
* Every configuration is repeated `REPS` times and the **median** is used.
* Speedup `S = T_seq / T_par`; efficiency `E = S / p` is reported for OpenMP and MPI only
  (a GPU has no core count comparable to `p`, so CUDA is reported as speedup).
* CPU builds use `-ffp-contract=off` and no `-ffast-math`, so OpenMP and MPI results are bit-identical to
  sequential. The GPU may differ by float round-off (use tolerance ~1e-4).
