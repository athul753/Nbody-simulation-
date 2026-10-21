#!/usr/bin/env python3
"""
analyze.py - turns results/results.csv into the tables and figures for the report.

Outputs (in results/ and figures/):
  tables:  summary_size.csv/.tex      time & speedup vs particle count (fixed parallelism)
           summary_scaling.csv/.tex   speedup, efficiency, Karp-Flatt vs threads/procs (fixed N)
           summary_cuda_block.csv/.tex  CUDA time vs block size
           complexity_fit.txt         log-log slope of time vs N (should be ~2 for O(N^2))
  figures: time_vs_n, speedup_vs_n, strong_scaling_speedup, strong_scaling_efficiency,
           cuda_block_size, complexity_loglog   (PNG for slides, PDF for LaTeX)

Definitions (match the project synopsis):
  Speedup    S = T_seq / T_par
  Efficiency E = S / p    (p = threads or MPI processes; reported for CPU paradigms only -
                           a GPU has no meaningful "p" comparable to CPU cores)
  Karp-Flatt e = (1/S - 1/p) / (1 - 1/p)   experimentally determined serial fraction
Medians over the repeated runs are used.
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

RES = "results/results.csv"
FIG = "figures"
os.makedirs(FIG, exist_ok=True)

if not os.path.exists(RES):
    sys.exit(f"{RES} not found - run ./run_experiments.sh first")

df = pd.read_csv(RES)
med = (df.groupby(["impl", "n", "units"], as_index=False)
         .agg(total_s=("total_s", "median"), avg_iter_s=("avg_iter_s", "median"),
              aux_s=("aux_s", "median"), iters=("iters", "first"), runs=("total_s", "count")))

STYLE = {"seq": ("Sequential", "k", "o"), "omp": ("OpenMP", "tab:blue", "s"),
         "mpi": ("MPI", "tab:green", "^"), "cuda": ("CUDA", "tab:red", "D")}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(f"{FIG}/{name}.png", dpi=200)
    fig.savefig(f"{FIG}/{name}.pdf")
    plt.close(fig)
    print(f"figure: {FIG}/{name}.png|pdf")


def to_tex(table, path, caption, label):
    with open(path, "w") as f:
        f.write(table.to_latex(index=False, float_format=lambda v: f"{v:.4g}",
                               caption=caption, label=label, escape=True))


# --------- sequential baseline lookup (per N) ---------
seq = med[med.impl == "seq"].set_index("n")["total_s"]

# =====================================================================
# B. problem size: one row per (impl, N) for the "fixed parallelism" runs
# =====================================================================
# pick the units value that was used in experiment B for each impl (most frequently measured across N)
size_rows = []
for impl in ["seq", "omp", "mpi", "cuda"]:
    sub = med[med.impl == impl]
    if sub.empty:
        continue
    u = sub.groupby("units")["n"].nunique().idxmax()
    sub = sub[sub.units == u].sort_values("n")
    for _, r in sub.iterrows():
        s = seq.get(r.n, np.nan) / r.total_s if r.n in seq.index else np.nan
        size_rows.append(dict(impl=impl, n=int(r.n), units=int(u), total_s=r.total_s,
                              avg_iter_s=r.avg_iter_s, speedup=s,
                              efficiency=(s / u if impl in ("omp", "mpi") else np.nan)))
size = pd.DataFrame(size_rows)
size.to_csv("results/summary_size.csv", index=False)
to_tex(size, "results/summary_size.tex", "Execution time, speedup and efficiency versus particle count.", "tab:size")
print(size.to_string(index=False))

fig, ax = plt.subplots(figsize=(6, 4))
for impl, (lab, c, m) in STYLE.items():
    s = size[size.impl == impl]
    if len(s):
        ax.plot(s.n, s.total_s, marker=m, color=c, label=f"{lab} (p={s.units.iloc[0]})" if impl in ("omp", "mpi") else lab)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Number of particles N"); ax.set_ylabel("Total time (s)")
ax.set_title("Execution time vs particle count"); ax.grid(True, which="both", alpha=.3); ax.legend()
save(fig, "time_vs_n")

fig, ax = plt.subplots(figsize=(6, 4))
for impl, (lab, c, m) in STYLE.items():
    s = size[(size.impl == impl) & size.speedup.notna()]
    if impl != "seq" and len(s):
        ax.plot(s.n, s.speedup, marker=m, color=c, label=lab)
ax.set_xscale("log"); ax.set_xlabel("Number of particles N"); ax.set_ylabel("Speedup over sequential")
ax.set_title("Speedup vs particle count"); ax.grid(True, which="both", alpha=.3); ax.legend()
save(fig, "speedup_vs_n")

# O(N^2) check: slope of log(time) vs log(N)
with open("results/complexity_fit.txt", "w") as f:
    f.write("Log-log fit of total time vs N (slope ~ 2 expected for direct O(N^2) summation)\n")
    for impl in ["seq", "omp", "mpi", "cuda"]:
        s = size[size.impl == impl]
        if len(s) >= 3:
            k, b = np.polyfit(np.log(s.n), np.log(s.total_s), 1)
            f.write(f"{impl:5s} slope = {k:.3f}   (over N = {int(s.n.min())}..{int(s.n.max())})\n")
print(open("results/complexity_fit.txt").read())

# =====================================================================
# C. strong scaling at the N that has the most (units) configurations
# =====================================================================
cnt = med[med.impl.isin(["omp", "mpi"])].groupby("n")["units"].nunique()
if len(cnt):
    N = int(cnt.idxmax())
    t_seq = seq.get(N, np.nan)
    rows = []
    for impl in ["omp", "mpi"]:
        for _, r in med[(med.impl == impl) & (med.n == N)].sort_values("units").iterrows():
            p = int(r.units)
            S = t_seq / r.total_s
            kf = (1 / S - 1 / p) / (1 - 1 / p) if p > 1 else np.nan
            rows.append(dict(impl=impl, n=N, p=p, total_s=r.total_s, speedup=S, efficiency=S / p, karp_flatt=kf))
    sc = pd.DataFrame(rows)
    sc.to_csv("results/summary_scaling.csv", index=False)
    to_tex(sc, "results/summary_scaling.tex", f"Strong scaling at N={N}.", "tab:scaling")
    print(sc.to_string(index=False))

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for impl in ["omp", "mpi"]:
        s = sc[sc.impl == impl]
        if len(s):
            ax.plot(s.p, s.speedup, marker=STYLE[impl][2], color=STYLE[impl][1], label=STYLE[impl][0])
    pmax = sc.p.max()
    ax.plot([1, pmax], [1, pmax], "--", color="gray", label="Ideal")
    ax.set_xlabel("Threads / processes p"); ax.set_ylabel("Speedup")
    ax.set_title(f"Strong scaling (N={N})"); ax.grid(True, alpha=.3); ax.legend()
    save(fig, "strong_scaling_speedup")

    fig, ax = plt.subplots(figsize=(5.5, 4))
    for impl in ["omp", "mpi"]:
        s = sc[sc.impl == impl]
        if len(s):
            ax.plot(s.p, s.efficiency, marker=STYLE[impl][2], color=STYLE[impl][1], label=STYLE[impl][0])
    ax.axhline(1.0, ls="--", color="gray")
    ax.set_ylim(0, 1.15); ax.set_xlabel("Threads / processes p"); ax.set_ylabel("Parallel efficiency")
    ax.set_title(f"Efficiency (N={N})"); ax.grid(True, alpha=.3); ax.legend()
    save(fig, "strong_scaling_efficiency")

    # CUDA block-size study at the same N
    cu = med[(med.impl == "cuda") & (med.n == N)].sort_values("units")
    if len(cu) > 1:
        cu = cu.assign(speedup=t_seq / cu.total_s).rename(columns={"units": "block_size"})
        cu[["block_size", "total_s", "avg_iter_s", "speedup", "aux_s"]].to_csv("results/summary_cuda_block.csv", index=False)
        to_tex(cu[["block_size", "total_s", "avg_iter_s", "speedup", "aux_s"]], "results/summary_cuda_block.tex",
               f"CUDA execution time versus thread-block size (N={N}); aux = host-device transfer time.", "tab:cudablock")
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.plot(cu.block_size, cu.total_s, marker="D", color="tab:red")
        ax.set_xscale("log", base=2); ax.set_xlabel("Threads per block"); ax.set_ylabel("Total kernel time (s)")
        ax.set_title(f"CUDA block-size study (N={N})"); ax.grid(True, which="both", alpha=.3)
        save(fig, "cuda_block_size")
else:
    print("no OpenMP/MPI scaling data found; skipping strong-scaling plots")
