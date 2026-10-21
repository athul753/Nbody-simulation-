/*
 * nbody_omp.c - OpenMP (shared-memory) N-body simulation.
 *
 * Usage: ./nbody_omp <input.bin> <iterations> <threads> [output.bin]
 *
 * Parallelisation: the outer i-loop (one particle per iteration) of both the
 * force/velocity update and the position update is split across threads.
 *   - Force loop : each thread reads all positions, writes only v[i]  -> no races.
 *   - Position   : each thread writes only its own r[i]               -> no races.
 * The implicit barrier at the end of each parallel-for guarantees that all
 * velocities are updated before positions move (same semantics as sequential).
 *
 * schedule(static) is used on purpose: every particle costs exactly N
 * interactions, so the work is uniform and static scheduling has the lowest
 * overhead. Set -DSCHED='dynamic,64' at compile time to experiment.
 */
#include "common.h"
#include <omp.h>

#ifndef SCHED
#define SCHED static
#endif

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <input.bin> <iterations> <threads> [output.bin]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int iters   = parse_int(argv[2], "iterations");
    const int threads = parse_int(argv[3], "threads");
    const char *out   = (argc > 4) ? argv[4] : NULL;

    omp_set_num_threads(threads);

    int n;
    Particle *p = load_particles(argv[1], &n);

    const double t0 = omp_get_wtime();
    for (int it = 0; it < iters; it++) {
        #pragma omp parallel for schedule(SCHED)
        for (int i = 0; i < n; i++) {
            float ax, ay, az;
            accel_on(p, n, i, &ax, &ay, &az);
            p[i].vx += DT * ax;
            p[i].vy += DT * ay;
            p[i].vz += DT * az;
        }
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < n; i++) {
            p[i].x += p[i].vx * DT;
            p[i].y += p[i].vy * DT;
            p[i].z += p[i].vz * DT;
        }
    }
    const double total = omp_get_wtime() - t0;

    printf("[omp] N=%d iterations=%d threads=%d total=%.4f s avg_iter=%.4f s\n",
           n, iters, threads, total, total / iters);
    print_result("omp", n, iters, threads, total, total / iters, 0.0);

    if (out) write_particles(out, p, n);
    free(p);
    return 0;
}
