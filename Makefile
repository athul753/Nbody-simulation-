# N-body project build file.
#   make            -> everything the machine can build (seq, omp, mpi, cuda, compare)
#   make seq omp compare
#   make cuda ARCH=sm_75      (Colab T4 = sm_75, RTX 30xx = sm_86, RTX 40xx = sm_89, A100 = sm_80)
#
# -ffp-contract=off keeps CPU results bit-identical between seq/omp/mpi.
# Do NOT use -ffast-math: it reorders the float sums and breaks the correctness check.

CC      ?= gcc
MPICC   ?= mpicc
NVCC    ?= nvcc
ARCH    ?= native
CFLAGS  := -O3 -march=native -ffp-contract=off -std=gnu11 -Wall
LDLIBS  := -lm

TARGETS := nbody_seq nbody_omp compare
ifneq ($(shell which $(MPICC) 2>/dev/null),)
TARGETS += nbody_mpi
endif
ifneq ($(shell which $(NVCC) 2>/dev/null),)
TARGETS += nbody_cuda
endif

all: $(TARGETS)

nbody_seq: nbody_seq.c common.h
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

nbody_omp: nbody_omp.c common.h
	$(CC) $(CFLAGS) -fopenmp -o $@ $< $(LDLIBS)

nbody_mpi: nbody_mpi.c common.h
	$(MPICC) $(CFLAGS) -o $@ $< $(LDLIBS)

nbody_cuda: nbody_cuda.cu common.h
	$(NVCC) -O3 -arch=$(ARCH) -o $@ $<

compare: compare.c common.h
	$(CC) $(CFLAGS) -o $@ $< $(LDLIBS)

seq: nbody_seq
omp: nbody_omp
mpi: nbody_mpi
cuda: nbody_cuda

clean:
	rm -f nbody_seq nbody_omp nbody_mpi nbody_cuda compare

.PHONY: all clean seq omp mpi cuda
