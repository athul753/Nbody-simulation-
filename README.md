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
| `nbody_cuda.cu` | CUDA: one thread per particle; selectable naive / shared-memory tiled force kernels |
| `compare.c` | Correctness checker (max / RMS difference between two result files) |
| `prepare_data.py` | CSV / Princeton / random data → binary input files |
| `run_experiments.sh` | Repeated benchmark studies → a fresh `results/runs/<run>/` directory |
| `analyze.py` | Tables (CSV + LaTeX) and figures (PNG + PDF) for the report |
| `generate_diagrams.py` | Workflow, OpenMP, MPI and CUDA diagrams (PNG, PDF, SVG) |

## Quick start

```bash
make                       # builds what your machine supports (mpicc / nvcc auto-detected)
make cuda ARCH=sm_75       # e.g. Colab T4; use sm_86 / sm_89 / sm_80 for other GPUs

python3 prepare_data.py csv c_0000.csv --sizes 1000 2000 5000 10000 20000 32000 64000
python3 prepare_data.py princeton planets.txt galaxy10k.txt galaxy20k.txt galaxy30k.txt   # optional extra datasets
python3 prepare_data.py random --sizes 100 1000                                            # debugging only

# single runs  (iterations = 10)
./nbody_seq  data/star_10000.bin 10 results/seq.bin
./nbody_omp  data/star_10000.bin 10 4 results/omp.bin          # 4 threads
mpirun -np 4 ./nbody_mpi data/star_10000.bin 10 results/mpi.bin
./nbody_cuda data/star_10000.bin 10 256 results/cuda.bin       # block size 256
./compare results/seq.bin results/omp.bin 1e-6                 # PASS/FAIL

# whole study, then report material (use a new directory for each study)
RUN_DIR=results/runs/study_01 bash run_experiments.sh
python3 -m pip install numpy pandas matplotlib
python3 analyze.py --input results/runs/study_01/results.csv
python3 generate_diagrams.py
```

The suite defaults to 100 iterations and five repetitions. The full study can take
a long time, especially the larger sequential runs. Run a small smoke check first:

```bash
ITERS=10 REPS=2 SIZES="1000 5000" SCALE_SIZES="1000 5000" \
UNITS_LIST="1 2 4" BLOCKS="64 256" RUN_DIR=results/runs/smoke_01 \
bash run_experiments.sh
```

Prepare all requested `data/star_<N>.bin` inputs first, including `star_2000.bin`
for the default correctness check. Rebuild CUDA after changing its source.
The runner requires Sequential and `compare`; unavailable parallel executables are
recorded in `skipped.txt`. Any executed command or correctness check that fails
stops the suite. `COMPLETE` marks a successfully finished study; partial runs are
retained for diagnosis. `commands.log`, `execution.log`, `config.txt` and
`system_info.txt` record commands, settings, source/binary hashes and hardware.

| Setting | Default / meaning |
|---|---|
| `ITERS`, `REPS` | 100 steps per run, 5 repeats |
| `SIZES` | 1000 5000 10000 20000 32000 64000 |
| `SCALE_SIZES` | 1000 10000 20000; legacy `SCALE_N` is also accepted |
| `UNITS_LIST` | 1 2 4 8 16 CPU threads / MPI processes |
| `BLOCKS` | 32 64 128 256 512 1024 CUDA threads per block |
| `FIXED_UNITS`, `CUDA_BLOCK` | 4 CPU workers, 256 CUDA threads for size/comparison studies |
| `MAX_SEQ_N` | 32000; caps sequential size-study runs, not scaling baselines |
| `CORR_N`, `CORR_ITERS` | 2000 particles, 10 steps for correctness checks |
| `BIN_DIR` | Directory containing compiled simulators; default `.` |
| `RUN_DIR` | Fresh timestamped directory under `results/runs/` |

Use worker counts supported by the machine and record affinity settings. On the
recorded 8-core / 16-logical-CPU machine, 16 workers test SMT, not 16 physical cores.
Single-machine MPI results demonstrate single-node scaling, not multi-node scaling.

## Expanded performance figures

Analysis writes into `<input directory>/analysis/` by default. Use
`--output-dir PATH` for another destination. Output directories must be new, which
protects earlier figures and tables. `python3 analyze.py` still reads the original
`results/results.csv`; legacy experiment IDs cannot be reconstructed, so the
analysis explicitly reports pooled duplicate configurations and inferred fixed units.
New CSVs label each row with `experiment` and `rep`. Different experiments are
analyzed separately; mixed iteration counts require `--iters N`.

| Figure stem | Contents |
|---|---|
| `time_vs_n` | Runtime versus particle count, medians with observed min-max bars |
| `speedup_vs_n` | Speedup over a measured sequential baseline at the same size and iteration count |
| `strong_scaling_speedup` | One panel per tested size, CPU speedup and ideal line |
| `strong_scaling_efficiency` | One panel per tested size, CPU efficiency and ideal line |
| `mpi_communication` | Collective time and median per-run collective/total percentage versus ranks |
| `cuda_transfer_cost` | Device-loop and initial H2D + final D2H transfer time versus size |
| `cuda_block_size` | Block-size sensitivity at each measured scaling size |
| `cuda_naive_vs_tiled` | Measured runtime comparison at identical block sizes |
| `cuda_tiling_speedup` | Naive runtime / tiled runtime; below one means tiling was slower |
| `complexity_loglog` | Log-log runtime curves, fitted slopes and a quadratic reference |

Each figure is saved as PNG and PDF. CSV/LaTeX tables, fitted slopes and
`analysis_notes.txt` accompany the figures. Missing experiments are not invented:
for example, the original CSV cannot produce naive-versus-tiled comparisons.
Missing sequential baselines yield missing speedups, not extrapolated measurements.

Error bars are observed ranges, not confidence intervals. Speedup bounds use
`minimum baseline / maximum parallel` and `maximum baseline / minimum parallel`;
they are conservative unpaired ranges. A single repetition provides no variability
estimate. Efficiency above one is retained for investigation rather than clipped.
Fitted timing slopes describe the measured range, not a change to the direct
algorithm's O(N²) operation count. Use longer runs if timings are too short.

CUDA's `total_s` is device-loop timing. `aux_s` covers initial host-to-device and
final device-to-host copies; it excludes the warm-up reset. Neither their sum nor
rendering FPS measures full application time. MPI's `aux_s` is the largest
accumulated collective time across ranks and includes waiting. It cannot be
subtracted from rank-0 total time to claim an exact computation-phase breakdown.

## CUDA optimization comparison

Generated examples from validation are included in this checkout:

- `results/expanded_analysis/`: eight figure types generated from the original Linux benchmark CSV, including MPI overhead.
- `results/runs/validation_cpu_cuda/`: a separate native Windows study with 99 timed runs, 20 steps and three repetitions; size studies at 1000/5000/10000 particles and OpenMP scaling at 1000/5000 particles with 1/2/4/8/16 threads. Its `analysis/figures/` contains the new CUDA comparisons.
- `figures/diagrams/`: all four implementation diagrams.

The Windows study checks OpenMP and both CUDA kernels against Sequential at
2000 particles for 10 steps. It is validation evidence, not a replacement for the
full default study. MPI was unavailable locally; no new MPI measurements are
claimed. Read the run's `validation_notes.txt` for compiler details and the
temporary Windows timing compatibility header used only for that build. Keep
these Windows measurements separate from the original Linux data.

The default remains the tiled kernel. Select the global-memory baseline with:

```bash
NBODY_CUDA_KERNEL=naive ./nbody_cuda data/star_1000.bin 100 256 results/naive.bin
NBODY_CUDA_KERNEL=tiled ./nbody_cuda data/star_1000.bin 100 256 results/tiled.bin
./compare results/naive.bin results/tiled.bin 1e-4
```

Both kernels use the same force expression, source iteration order, timestep,
position kernel and warm-up/reset procedure. Results are labelled `cuda_naive`
and `cuda`, respectively. The runner checks both against sequential output and
alternates their benchmark order across repetitions. Tiling's benefit must be
measured; cache behavior and workload size can affect the outcome.

## Implementation diagrams

`python3 generate_diagrams.py` produces `workflow`, `openmp_partitioning`,
`mpi_exchange`, and `cuda_tiling` under `figures/diagrams/` in PNG, PDF and SVG.
Use `--output-dir PATH` to regenerate into a fresh directory. The diagrams describe
the current implementation, including OpenMP barriers, replicated MPI arrays and
the separate CUDA position-update kernel. They are static report illustrations.

## Particle snapshot grids

`generate_snapshots.py` creates a time-labelled 2 × 4 grid of black particle dots,
with square panels and a shared scale, using actual simulation states. Install
the plotting dependencies with `python -m pip install numpy matplotlib`.

```bash
make seq
python3 generate_snapshots.py data/star_1000.bin

# GPU example (build CUDA first): eight snapshots at chosen step numbers
python3 generate_snapshots.py data/star_5000.bin --impl cuda --units 256 \
  --steps 320 360 400 440 480 520 560 600 \
  --output-dir results/snapshots_cuda --figure figures/snapshots_cuda
```

Defaults are steps 0, 40, 80, 120, 160, 200, 240, 280 and the XY projection.
Use `--plane xz` or `--plane yz` for other views. `--impl omp --units 4` selects
four OpenMP threads; `--impl mpi --units 4` selects four MPI processes.
`--executable PATH` selects a custom executable (on Windows the default has `.exe`).
Run the script in the same environment as the simulator, e.g. Linux/WSL for Linux binaries.

Outputs are `results/snapshots/step_*.bin`, `results/snapshots/metadata.json`, and
`figures/snapshots.png` / `.pdf`. Existing outputs are protected: choose a fresh
`--output-dir` and `--figure` to generate another set.

Each interval resumes from the preceding saved state, including velocities.
The script uses the existing simulators without changing their physics or the
benchmark suite. These separate visualization runs are not benchmark measurements.
Labels use `start_time + step * dt`, with `--start-time 0` and `--dt 0.01` by
default. If compiled with a different `DT`, pass that value with `--dt` (this option
only sets labels). Step 320 therefore means time 3.2 with the default timestep.
All particles are shown within common limits; the cluster's appearance depends
on the input data and evolution, and may differ from a reference image.

## Binary particle layout

`int32 N` followed by `N × {m, x, y, z, vx, vy, vz}` as `float32` (native endian).
The particle count comes from the file, so there is no separate N argument.

## Data notes

* The 64,000-star King-model snapshot is subsampled **at random (seed 42)**; subsample masses are rescaled
  so the total mass stays 1 (`--keep-mass` disables this). The full 64,000 is used unchanged.
* Princeton COS 126 files are 2-D (`px py vx vy mass`); they are converted with `z = vz = 0`.

## Mapping to the report (Mini Project Guidelines §3)

| Report section | Where it comes from |
|---|---|
| Experimental Setup | Selected run's `system_info.txt` and `config.txt` |
| Results / Performance Analysis | Selected run's `analysis/summary_*.csv|tex`, `analysis/figures/*.pdf` |
| Verification of correctness | Selected run's `correctness.txt` |
| Investigates O(N²) scaling | `analysis/complexity_fit.txt` and `analysis/figures/complexity_loglog` |
| Speedup, efficiency, scalability | `analysis/summary_scaling`, `analysis/summary_size` |
| GPU configuration / optimization | `analysis/summary_cuda_block`, `analysis/summary_cuda_comparison` |
| System architecture | `figures/diagrams/*.pdf` |

## Measurement rules (so the comparison is fair)

* Only the simulation loop is timed. File I/O is excluded everywhere; CUDA host↔device transfer is reported
  separately (`aux_s` column).
* No output is written inside the iteration loop.
* Every configuration is repeated `REPS` times and the **median** is used.
* Speedup `S = T_seq / T_par`; efficiency `E = S / p` is reported for OpenMP and MPI only
  (a GPU has no core count comparable to `p`, so CUDA is reported as speedup).
* CPU builds use `-ffp-contract=off` and no `-ffast-math`, so OpenMP and MPI results are bit-identical to
  sequential. The GPU may differ by float round-off (use tolerance ~1e-4).
