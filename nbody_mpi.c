/*
 * nbody_mpi.c - MPI (distributed-memory) N-body simulation.
 *
 * Usage: mpirun -np <P> ./nbody_mpi <input.bin> <iterations> [output.bin]
 *
 * Decomposition: every rank keeps a full copy of the particle array (the
 * force on a particle needs all other particles) but is *responsible* for
 * updating one contiguous block of ~N/P particles.
 *
 * Per iteration:
 *   1. each rank computes the new velocities of its own block   (O(N * N/P))
 *   2. each rank advances the positions of its own block
 *   3. MPI_Allgatherv (MPI_IN_PLACE) distributes the updated blocks so every
 *      rank again holds the complete, up-to-date system             (O(N) data)
 *
 * Rank 0 reads the input once and broadcasts it; file I/O is excluded from
 * the timed region. Blocks differ by at most one particle (load balanced).
 */
#include "common.h"
#include <mpi.h>

int main(int argc, char **argv)
{
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (argc < 3) {
        if (rank == 0) fprintf(stderr, "Usage: mpirun -np P %s <input.bin> <iterations> [output.bin]\n", argv[0]);
        MPI_Finalize();
        return EXIT_FAILURE;
    }
    const int iters = parse_int(argv[2], "iterations");
    const char *out = (argc > 3) ? argv[3] : NULL;

    MPI_Datatype ptype;
    MPI_Type_contiguous(7, MPI_FLOAT, &ptype);
    MPI_Type_commit(&ptype);

    /* ---- input: rank 0 reads, everybody receives a full copy ---- */
    int n = 0;
    Particle *p = NULL;
    if (rank == 0) p = load_particles(argv[1], &n);
    MPI_Bcast(&n, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (rank != 0) p = (Particle *)malloc((size_t)n * sizeof(Particle));
    MPI_Bcast(p, n, ptype, 0, MPI_COMM_WORLD);

    /* ---- block decomposition ---- */
    int *counts = (int *)malloc(size * sizeof(int));
    int *displs = (int *)malloc(size * sizeof(int));
    const int base = n / size, rem = n % size;
    for (int r = 0, off = 0; r < size; r++) {
        counts[r] = base + (r < rem ? 1 : 0);
        displs[r] = off;
        off += counts[r];
    }
    const int lo = displs[rank], hi = lo + counts[rank];

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();
    double comm_time = 0.0;

    for (int it = 0; it < iters; it++) {
        for (int i = lo; i < hi; i++) {
            float ax, ay, az;
            accel_on(p, n, i, &ax, &ay, &az);
            p[i].vx += DT * ax;
            p[i].vy += DT * ay;
            p[i].vz += DT * az;
        }
        for (int i = lo; i < hi; i++) {
            p[i].x += p[i].vx * DT;
            p[i].y += p[i].vy * DT;
            p[i].z += p[i].vz * DT;
        }
        const double c0 = MPI_Wtime();
        MPI_Allgatherv(MPI_IN_PLACE, 0, ptype, p, counts, displs, ptype, MPI_COMM_WORLD);
        comm_time += MPI_Wtime() - c0;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double total = MPI_Wtime() - t0;

    /* report the slowest rank's communication time (it bounds the iteration) */
    double comm_max;
    MPI_Reduce(&comm_time, &comm_max, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("[mpi] N=%d iterations=%d procs=%d total=%.4f s avg_iter=%.4f s (comm<=%.4f s)\n",
               n, iters, size, total, total / iters, comm_max);
        print_result("mpi", n, iters, size, total, total / iters, comm_max);
        if (out) write_particles(out, p, n);
    }

    free(p); free(counts); free(displs);
    MPI_Type_free(&ptype);
    MPI_Finalize();
    return 0;
}
