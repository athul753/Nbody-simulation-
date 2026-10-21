/*
 * compare.c - correctness check between two result files.
 *
 * Usage: ./compare <reference.bin> <test.bin> [tolerance]
 *
 * Reports, over all particles and all 6 phase-space components, the maximum
 * absolute difference and the maximum difference relative to the system's
 * overall scale. PASS if max abs diff <= tolerance (default 1e-4).
 * CPU implementations are expected to match the sequential run (near) exactly;
 * the GPU may differ by float round-off (different FMA / rounding behaviour).
 * Exit code 0 = PASS, 1 = FAIL.
 */
#include "common.h"

int main(int argc, char **argv)
{
    if (argc < 3) {
        fprintf(stderr, "Usage: %s <reference.bin> <test.bin> [tolerance]\n", argv[0]);
        return 2;
    }
    const double tol = (argc > 3) ? atof(argv[3]) : 1e-4;

    int na, nb;
    Particle *a = load_particles(argv[1], &na);
    Particle *b = load_particles(argv[2], &nb);
    if (na != nb) { printf("FAIL: particle counts differ (%d vs %d)\n", na, nb); return 1; }

    double maxabs = 0.0, scale = 0.0, sumsq = 0.0;
    int worst = -1, bad_nan = 0;
    for (int i = 0; i < na; i++) {
        const float A[7] = { a[i].m, a[i].x, a[i].y, a[i].z, a[i].vx, a[i].vy, a[i].vz };
        const float B[7] = { b[i].m, b[i].x, b[i].y, b[i].z, b[i].vx, b[i].vy, b[i].vz };
        for (int k = 0; k < 7; k++) {
            if (isnan(A[k]) || isnan(B[k])) { bad_nan++; continue; }
            double d = fabs((double)A[k] - (double)B[k]);
            sumsq += d * d;
            if (d > maxabs) { maxabs = d; worst = i; }
            if (fabs((double)A[k]) > scale) scale = fabs((double)A[k]);
        }
    }
    const double rms = sqrt(sumsq / (7.0 * na));
    printf("particles=%d  max|diff|=%.3e (particle %d)  rms diff=%.3e  scale=%.3e  max rel=%.3e  NaNs=%d\n",
           na, maxabs, worst, rms, scale, scale > 0 ? maxabs / scale : 0.0, bad_nan);
    const int ok = (bad_nan == 0 && maxabs <= tol);
    printf("%s (tolerance %.1e)\n", ok ? "PASS" : "FAIL", tol);
    free(a); free(b);
    return ok ? 0 : 1;
}
