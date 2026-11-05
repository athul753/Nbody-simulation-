#!/usr/bin/env bash
# Repeatable studies. Existing results are never overwritten.
set -Eeuo pipefail
cd "$(dirname "$0")"
ITERS=${ITERS:-100}
REPS=${REPS:-5}
SIZES=${SIZES:-"1000 5000 10000 20000 32000 64000"}
SCALE_SIZES=${SCALE_SIZES:-${SCALE_N:-"1000 10000 20000"}}
UNITS_LIST=${UNITS_LIST:-"1 2 4 8 16"}
BLOCKS=${BLOCKS:-"32 64 128 256 512 1024"}
FIXED_UNITS=${FIXED_UNITS:-4}
CUDA_BLOCK=${CUDA_BLOCK:-256}
MAX_SEQ_N=${MAX_SEQ_N:-32000}
CORR_N=${CORR_N:-2000}
CORR_ITERS=${CORR_ITERS:-10}
BIN_DIR=${BIN_DIR:-.}
RUN_DIR=${RUN_DIR:-"results/runs/$(date +%Y%m%d_%H%M%S)_$$"}
for value in $ITERS $REPS $SIZES $SCALE_SIZES $UNITS_LIST $BLOCKS $FIXED_UNITS $CUDA_BLOCK $MAX_SEQ_N $CORR_N $CORR_ITERS; do
    [[ "$value" =~ ^[1-9][0-9]*$ ]] || { echo "Invalid positive integer: $value" >&2; exit 1; }
done
mkdir -p "$(dirname "$RUN_DIR")"
mkdir "$RUN_DIR"
trap 'echo "FAILED at line $LINENO; partial results retained in $RUN_DIR" >&2' ERR
OUT="$RUN_DIR/results.csv"
CORR="$RUN_DIR/correctness.txt"
have() { [[ -x "$BIN_DIR/$1" ]]; }
MPIRUN=$(command -v mpirun || command -v mpiexec || true)
have nbody_seq && have compare || { echo 'Build seq and compare first' >&2; exit 1; }
for impl in omp mpi cuda; do
    if ! have "nbody_$impl"; then echo "SKIP: nbody_$impl not built" | tee -a "$RUN_DIR/skipped.txt"; fi
done
if have nbody_mpi && [[ -z "$MPIRUN" ]]; then echo 'SKIP: MPI launcher unavailable' | tee -a "$RUN_DIR/skipped.txt"; fi
for n in $SIZES $SCALE_SIZES $CORR_N; do
    [[ -f "data/star_$n.bin" ]] || { echo "Missing data/star_$n.bin; prepare datasets first" >&2; exit 1; }
done
{
    date -Is
    uname -a
    lscpu || true
    free -h || true
    gcc --version || true
    mpicc --version || true
    nvcc --version || true
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv || true
} > "$RUN_DIR/system_info.txt" 2>&1
{
    printf 'ITERS=%s\nREPS=%s\nSIZES=%s\nSCALE_SIZES=%s\nUNITS_LIST=%s\nBLOCKS=%s\n' "$ITERS" "$REPS" "$SIZES" "$SCALE_SIZES" "$UNITS_LIST" "$BLOCKS"
    printf 'FIXED_UNITS=%s\nCUDA_BLOCK=%s\nMAX_SEQ_N=%s\nCORR_N=%s\nCORR_ITERS=%s\nBIN_DIR=%s\n' "$FIXED_UNITS" "$CUDA_BLOCK" "$MAX_SEQ_N" "$CORR_N" "$CORR_ITERS" "$BIN_DIR"
    printf 'OMP_PROC_BIND=%s\nOMP_PLACES=%s\n' "${OMP_PROC_BIND:-unset}" "${OMP_PLACES:-unset}"
    git rev-parse HEAD || true
    git diff --stat || true
    sha256sum "$BIN_DIR"/nbody_* common.h nbody_*.c nbody_cuda.cu || true
} > "$RUN_DIR/config.txt" 2>&1
echo 'experiment,rep,impl,n,iters,units,total_s,avg_iter_s,aux_s' > "$OUT"
record() {
    local experiment=$1 rep=$2 result
    shift 2
    printf '%q ' "$@" >> "$RUN_DIR/commands.log"
    printf '\n' >> "$RUN_DIR/commands.log"
    "$@" > "$RUN_DIR/last.log" 2>&1
    cat "$RUN_DIR/last.log" | tee -a "$RUN_DIR/execution.log"
    result=$(grep '^RESULT,' "$RUN_DIR/last.log")
    [[ $(printf '%s\n' "$result" | wc -l) -eq 1 ]] || { echo 'Expected exactly one RESULT line' >&2; return 1; }
    printf '%s,%s,%s\n' "$experiment" "$rep" "${result#RESULT,}" >> "$OUT"
}
echo 'Correctness checks (must all pass before benchmarks)' | tee "$CORR"
input="data/star_$CORR_N.bin"
"$BIN_DIR/nbody_seq" "$input" "$CORR_ITERS" "$RUN_DIR/ref_seq.bin" >> "$CORR"
if have nbody_omp; then
    "$BIN_DIR/nbody_omp" "$input" "$CORR_ITERS" "$FIXED_UNITS" "$RUN_DIR/out_omp.bin" >> "$CORR"
    "$BIN_DIR/compare" "$RUN_DIR/ref_seq.bin" "$RUN_DIR/out_omp.bin" 1e-6 | tee -a "$CORR"
fi
if have nbody_mpi && [[ -n "$MPIRUN" ]]; then
    "$MPIRUN" -np "$FIXED_UNITS" "$BIN_DIR/nbody_mpi" "$input" "$CORR_ITERS" "$RUN_DIR/out_mpi.bin" >> "$CORR"
    "$BIN_DIR/compare" "$RUN_DIR/ref_seq.bin" "$RUN_DIR/out_mpi.bin" 1e-6 | tee -a "$CORR"
fi
if have nbody_cuda; then
    for kernel in naive tiled; do
        env NBODY_CUDA_KERNEL="$kernel" "$BIN_DIR/nbody_cuda" "$input" "$CORR_ITERS" "$CUDA_BLOCK" "$RUN_DIR/out_$kernel.bin" >> "$CORR"
        "$BIN_DIR/compare" "$RUN_DIR/ref_seq.bin" "$RUN_DIR/out_$kernel.bin" 1e-4 | tee -a "$CORR"
    done
fi
for n in $SIZES; do
    for ((r=1; r<=REPS; r++)); do
        input="data/star_$n.bin"
        if (( n <= MAX_SEQ_N )); then record size "$r" "$BIN_DIR/nbody_seq" "$input" "$ITERS"; fi
        if have nbody_omp; then record size "$r" "$BIN_DIR/nbody_omp" "$input" "$ITERS" "$FIXED_UNITS"; fi
        if have nbody_mpi && [[ -n "$MPIRUN" ]]; then record size "$r" "$MPIRUN" -np "$FIXED_UNITS" "$BIN_DIR/nbody_mpi" "$input" "$ITERS"; fi
        if have nbody_cuda; then record size "$r" env NBODY_CUDA_KERNEL=tiled "$BIN_DIR/nbody_cuda" "$input" "$ITERS" "$CUDA_BLOCK"; fi
    done
done
for n in $SCALE_SIZES; do
    input="data/star_$n.bin"
    for ((r=1; r<=REPS; r++)); do
        record scaling "$r" "$BIN_DIR/nbody_seq" "$input" "$ITERS"
        for u in $UNITS_LIST; do
            if have nbody_omp; then record scaling "$r" "$BIN_DIR/nbody_omp" "$input" "$ITERS" "$u"; fi
            if have nbody_mpi && [[ -n "$MPIRUN" ]]; then record scaling "$r" "$MPIRUN" -np "$u" "$BIN_DIR/nbody_mpi" "$input" "$ITERS"; fi
        done
        if have nbody_cuda; then
            for b in $BLOCKS; do record block "$r" env NBODY_CUDA_KERNEL=tiled "$BIN_DIR/nbody_cuda" "$input" "$ITERS" "$b"; done
        fi
    done
done
if have nbody_cuda; then
    for n in $SIZES; do
        for ((r=1; r<=REPS; r++)); do
            kernels='naive tiled'; if (( r % 2 == 0 )); then kernels='tiled naive'; fi
            for kernel in $kernels; do
                record cuda_compare "$r" env NBODY_CUDA_KERNEL="$kernel" "$BIN_DIR/nbody_cuda" "data/star_$n.bin" "$ITERS" "$CUDA_BLOCK"
            done
        done
    done
fi
touch "$RUN_DIR/COMPLETE"
echo "Completed. Analyze with: python3 analyze.py --input $OUT"
