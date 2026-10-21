#!/usr/bin/env bash
# run_experiments.sh - runs the whole benchmark suite and writes results/results.csv
#
# Experiments (each repeated REPS times; analyze.py uses the median):
#   A. correctness   : seq vs omp / mpi / cuda final states        -> results/correctness.txt
#   B. problem size  : every implementation at fixed parallelism, N = SIZES
#   C. strong scaling: fixed N = SCALE_N, threads/procs = 1,2,4,8,... and CUDA block sizes
#
# Usage:  ./run_experiments.sh            (uses defaults below)
#         ITERS=10 REPS=3 SIZES="1000 5000 10000" ./run_experiments.sh
# Implementations that are not built (no mpicc / nvcc) are skipped automatically.
set -u
cd "$(dirname "$0")"

ITERS=${ITERS:-10}
REPS=${REPS:-3}
SIZES=${SIZES:-"1000 5000 10000 20000 32000 64000"}     # data/star_<N>.bin must exist
SCALE_N=${SCALE_N:-20000}
FIXED_UNITS=${FIXED_UNITS:-4}                            # threads / procs used in experiment B
UNITS_LIST=${UNITS_LIST:-"1 2 4 8"}                      # threads / procs for experiment C
BLOCKS=${BLOCKS:-"32 64 128 256 512 1024"}               # CUDA block sizes for experiment C
CUDA_BLOCK=${CUDA_BLOCK:-256}                            # CUDA block size used in experiment B
MAX_SEQ_N=${MAX_SEQ_N:-32000}                            # skip the slow sequential run above this N (set big to run all)
OUT=results/results.csv
CORR=results/correctness.txt
mkdir -p results data

have() { [ -x "./$1" ]; }
MPIRUN=$(command -v mpirun || command -v mpiexec || true)

# ---- system information for the report's "Experimental Setup" section ----
{
  echo "date: $(date -Is)"
  echo "--- cpu ---";  lscpu 2>/dev/null | grep -E "Model name|Socket|Core|Thread|^CPU\(s\)|MHz|L2|L3"
  echo "--- memory ---"; free -h 2>/dev/null | head -2
  echo "--- compilers ---"; gcc --version | head -1; (mpicc --version 2>/dev/null | head -1) || true; (nvcc --version 2>/dev/null | tail -2) || true
  echo "--- gpu ---"; nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv 2>/dev/null || echo "no nvidia-smi"
} > results/system_info.txt

echo "impl,n,iters,units,total_s,avg_iter_s,aux_s" > "$OUT"

# run one command, keep only RESULT lines (drop the "RESULT," prefix) and append to the CSV
record() { "$@" 2>&1 | tee /dev/stderr | grep '^RESULT,' | sed 's/^RESULT,//' >> "$OUT"; }

# ---------------- A. correctness ----------------
echo "=== A. correctness ===" | tee "$CORR"
CN=${CORR_N:-2000}
if [ ! -f data/star_$CN.bin ]; then python3 prepare_data.py csv ${CSV:-c_0000.csv} --sizes $CN >/dev/null; fi
IN=data/star_$CN.bin
./nbody_seq $IN $ITERS results/ref_seq.bin >/dev/null
have nbody_omp && { ./nbody_omp $IN $ITERS 4 results/out_omp.bin >/dev/null; echo "OpenMP vs sequential (N=$CN, $ITERS iters):" | tee -a "$CORR"; ./compare results/ref_seq.bin results/out_omp.bin 1e-6 | tee -a "$CORR"; }
if have nbody_mpi && [ -n "$MPIRUN" ]; then
  $MPIRUN -np 4 ./nbody_mpi $IN $ITERS results/out_mpi.bin >/dev/null
  echo "MPI(4) vs sequential:" | tee -a "$CORR"; ./compare results/ref_seq.bin results/out_mpi.bin 1e-6 | tee -a "$CORR"
fi
if have nbody_cuda; then
  ./nbody_cuda $IN $ITERS $CUDA_BLOCK results/out_cuda.bin >/dev/null
  echo "CUDA vs sequential (float round-off expected):" | tee -a "$CORR"; ./compare results/ref_seq.bin results/out_cuda.bin 1e-4 | tee -a "$CORR"
fi

# ---------------- B. problem-size scaling ----------------
echo "=== B. problem size ==="
for n in $SIZES; do
  f=data/star_$n.bin
  [ -f "$f" ] || { echo "missing $f - run prepare_data.py first"; continue; }
  for r in $(seq $REPS); do
    [ "$n" -le "$MAX_SEQ_N" ] && record ./nbody_seq $f $ITERS
    have nbody_omp && record ./nbody_omp $f $ITERS $FIXED_UNITS
    [ -n "$MPIRUN" ] && have nbody_mpi && record $MPIRUN -np $FIXED_UNITS ./nbody_mpi $f $ITERS
    have nbody_cuda && record ./nbody_cuda $f $ITERS $CUDA_BLOCK
  done
done

# ---------------- C. strong scaling at fixed N ----------------
echo "=== C. strong scaling (N=$SCALE_N) ==="
f=data/star_$SCALE_N.bin
[ -f "$f" ] || python3 prepare_data.py csv ${CSV:-c_0000.csv} --sizes $SCALE_N >/dev/null
for r in $(seq $REPS); do
  record ./nbody_seq $f $ITERS
  for u in $UNITS_LIST; do
    have nbody_omp && record ./nbody_omp $f $ITERS $u
    [ -n "$MPIRUN" ] && have nbody_mpi && record $MPIRUN -np $u ./nbody_mpi $f $ITERS
  done
  if have nbody_cuda; then for b in $BLOCKS; do record ./nbody_cuda $f $ITERS $b; done; fi
done

echo "Done. Raw results: $OUT   Correctness: $CORR   System: results/system_info.txt"
