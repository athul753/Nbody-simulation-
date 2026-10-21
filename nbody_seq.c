/*
 * nbody_seq.c - sequential direct-summation N-body simulation (baseline).
 *
 * Usage: ./nbody_seq <input.bin> <iterations> [output.bin]
 *
 * Timing covers only the simulation loop (file I/O excluded) so that all
 * implementations are measured on the same footing.
 */
#include "common.h"

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <input.bin> <iterations> [output.bin]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int iters = parse_int(argv[2], "iterations");
    const char *out = (argc > 3) ? argv[3] : NULL;

    int n;
    Particle *p = load_particles(argv[1], &n);

    const double t0 = wall_time();
    for (int it = 0; it < iters; it++) {
        /* 1. force/velocity update: reads positions only, writes velocities only */
        for (int i = 0; i < n; i++) {
            float ax, ay, az;
            accel_on(p, n, i, &ax, &ay, &az);
            p[i].vx += DT * ax;
            p[i].vy += DT * ay;
            p[i].vz += DT * az;
        }
        /* 2. position update */
        for (int i = 0; i < n; i++) {
            p[i].x += p[i].vx * DT;
            p[i].y += p[i].vy * DT;
            p[i].z += p[i].vz * DT;
        }
    }
    const double total = wall_time() - t0;

    printf("[seq] N=%d iterations=%d total=%.4f s avg_iter=%.4f s\n", n, iters, total, total / iters);
    print_result("seq", n, iters, 1, total, total / iters, 0.0);

    if (out) write_particles(out, p, n);
    free(p);
    return 0;
}
