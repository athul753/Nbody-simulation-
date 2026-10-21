/*
 * common.h - shared definitions for the N-body project.
 *
 * Physics (direct O(N^2) summation, N-body units, G = 1):
 *   a_i = sum_j  m_j * (r_j - r_i) / (|r_j - r_i|^2 + EPS2)^(3/2)
 *   v_i += a_i * DT        (all accelerations use positions from the START of the step)
 *   r_i += v_i * DT        (semi-implicit / symplectic Euler)
 *
 * The inner j-loop is always executed in the same order j = 0..n-1 with the same
 * arithmetic in every implementation, so sequential/OpenMP/MPI/CUDA results
 * should agree to (near) floating-point round-off.
 *
 * Binary file format (native endian):  int32 n, then n * Particle
 * Particle = 7 x float32 : m, x, y, z, vx, vy, vz   (28 bytes, no padding)
 */
#ifndef NBODY_COMMON_H
#define NBODY_COMMON_H

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <limits.h>
#include <errno.h>
#include <math.h>
#include <time.h>

#ifndef EPS2
#define EPS2 1e-4f      /* softening length squared (avoids singularities in close encounters) */
#endif
#ifndef DT
#define DT 0.01f        /* time step */
#endif

typedef struct { float m, x, y, z, vx, vy, vz; } Particle;

#ifdef __cplusplus
static_assert(sizeof(Particle) == 7 * sizeof(float), "Particle must be 28 bytes");
#else
_Static_assert(sizeof(Particle) == 7 * sizeof(float), "Particle must be 28 bytes");
#endif

static inline double wall_time(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static inline int parse_int(const char *s, const char *what)
{
    char *end;
    errno = 0;
    long v = strtol(s, &end, 10);
    if (errno || end == s || *end != '\0' || v <= 0 || v > INT_MAX) {
        fprintf(stderr, "Invalid %s: '%s'\n", what, s);
        exit(EXIT_FAILURE);
    }
    return (int)v;
}

/* Load a binary particle file; the particle count comes from the file header. */
static inline Particle *load_particles(const char *path, int *n_out)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); exit(EXIT_FAILURE); }
    int32_t n;
    if (fread(&n, sizeof n, 1, f) != 1 || n <= 0) {
        fprintf(stderr, "%s: bad header\n", path);
        exit(EXIT_FAILURE);
    }
    Particle *p = (Particle *)malloc((size_t)n * sizeof(Particle));
    if (!p || fread(p, sizeof(Particle), (size_t)n, f) != (size_t)n) {
        fprintf(stderr, "%s: truncated file or out of memory\n", path);
        exit(EXIT_FAILURE);
    }
    fclose(f);
    *n_out = (int)n;
    return p;
}

/* Same, but abort if the file does not contain exactly n particles. */
static inline Particle *load_particles_checked(const char *path, int n)
{
    int nf;
    Particle *p = load_particles(path, &nf);
    if (nf != n) {
        fprintf(stderr, "%s contains %d particles but %d were requested\n", path, nf, n);
        exit(EXIT_FAILURE);
    }
    return p;
}

static inline void write_particles(const char *path, const Particle *p, int n)
{
    FILE *f = fopen(path, "wb");
    if (!f) { perror(path); exit(EXIT_FAILURE); }
    int32_t nn = (int32_t)n;
    fwrite(&nn, sizeof nn, 1, f);
    fwrite(p, sizeof(Particle), (size_t)n, f);
    fclose(f);
}

/* Acceleration on particle i from all n particles (identical on every CPU path). */
static inline void accel_on(const Particle *p, int n, int i, float *ax, float *ay, float *az)
{
    const float xi = p[i].x, yi = p[i].y, zi = p[i].z;
    float fx = 0.0f, fy = 0.0f, fz = 0.0f;
    for (int j = 0; j < n; j++) {
        float dx = p[j].x - xi;
        float dy = p[j].y - yi;
        float dz = p[j].z - zi;
        float d2 = dx * dx + dy * dy + dz * dz + EPS2;
        float inv = 1.0f / sqrtf(d2);
        float inv3 = inv * inv * inv;
        float s = p[j].m * inv3;
        fx += dx * s;
        fy += dy * s;
        fz += dz * s;
    }
    *ax = fx; *ay = fy; *az = fz;
}

/* Machine-readable result line collected by run_experiments.sh:
 * RESULT,impl,n,iters,units,total_s,avg_iter_s,aux_s            */
static inline void print_result(const char *impl, int n, int iters, int units,
                                double total, double avg_iter, double aux)
{
    printf("RESULT,%s,%d,%d,%d,%.6f,%.6f,%.6f\n", impl, n, iters, units, total, avg_iter, aux);
}

#endif /* NBODY_COMMON_H */
