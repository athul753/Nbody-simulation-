/*
 * nbody_cuda.cu - CUDA (GPU) N-body simulation.
 *
 * Usage: ./nbody_cuda <input.bin> <iterations> <block_size> [output.bin]
 *
 * Mapping: one GPU thread per particle i. The inner j-loop is "tiled":
 * each thread block cooperatively loads blockDim.x particles (x,y,z,m) into
 * shared memory, every thread consumes the tile, then the next tile is loaded.
 * This cuts global-memory traffic by a factor of blockDim.x. Tiles are
 * consumed in order j = 0..N-1, i.e. the same summation order as the CPU code.
 *
 * Two kernels per step (force, then position) so that all forces use the
 * positions from the start of the step - identical semantics to the CPU code.
 *
 * Timing: cudaEvent timers around the iteration loop (kernels only). Host<->device
 * transfer time is measured separately and reported as the "aux" column.
 * Nothing is written to disk inside the loop.
 *
 * Build: nvcc -O3 -arch=native -o nbody_cuda nbody_cuda.cu    (or -arch=sm_75 for a Colab T4)
 */
#include "common.h"
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do {                                                   \
    cudaError_t e_ = (call);                                                    \
    if (e_ != cudaSuccess) {                                                    \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,           \
                cudaGetErrorString(e_));                                        \
        exit(EXIT_FAILURE);                                                     \
    }                                                                           \
} while (0)

__global__ void force_kernel(Particle *p, int n)
{
    extern __shared__ float4 tile[];                 /* (x, y, z, m) of one tile */
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    float xi = 0.f, yi = 0.f, zi = 0.f;
    if (i < n) { xi = p[i].x; yi = p[i].y; zi = p[i].z; }
    float fx = 0.f, fy = 0.f, fz = 0.f;

    for (int base = 0; base < n; base += blockDim.x) {
        const int j = base + threadIdx.x;
        if (j < n) tile[threadIdx.x] = make_float4(p[j].x, p[j].y, p[j].z, p[j].m);
        __syncthreads();

        const int lim = min((int)blockDim.x, n - base);
        for (int k = 0; k < lim; k++) {
            const float4 q = tile[k];
            const float dx = q.x - xi;
            const float dy = q.y - yi;
            const float dz = q.z - zi;
            const float d2 = dx * dx + dy * dy + dz * dz + EPS2;
            const float inv = 1.0f / sqrtf(d2);
            const float inv3 = inv * inv * inv;
            const float s = q.w * inv3;
            fx += dx * s; fy += dy * s; fz += dz * s;
        }
        __syncthreads();                             /* tile may be overwritten next round */
    }

    if (i < n) {
        p[i].vx += DT * fx;
        p[i].vy += DT * fy;
        p[i].vz += DT * fz;
    }
}

__global__ void position_kernel(Particle *p, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        p[i].x += p[i].vx * DT;
        p[i].y += p[i].vy * DT;
        p[i].z += p[i].vz * DT;
    }
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <input.bin> <iterations> <block_size> [output.bin]\n", argv[0]);
        return EXIT_FAILURE;
    }
    const int iters = parse_int(argv[2], "iterations");
    const int block = parse_int(argv[3], "block_size");
    const char *out = (argc > 4) ? argv[4] : NULL;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    if (block > prop.maxThreadsPerBlock) {
        fprintf(stderr, "block_size %d exceeds device limit %d\n", block, prop.maxThreadsPerBlock);
        return EXIT_FAILURE;
    }

    int n;
    Particle *h = load_particles(argv[1], &n);
    Particle *d;
    CUDA_CHECK(cudaMalloc(&d, (size_t)n * sizeof(Particle)));

    cudaEvent_t e0, e1;
    CUDA_CHECK(cudaEventCreate(&e0));
    CUDA_CHECK(cudaEventCreate(&e1));

    /* host -> device (timed separately) */
    const double c0 = wall_time();
    CUDA_CHECK(cudaMemcpy(d, h, (size_t)n * sizeof(Particle), cudaMemcpyHostToDevice));
    double xfer = wall_time() - c0;

    const int grid = (n + block - 1) / block;
    const size_t shmem = (size_t)block * sizeof(float4);

    /* warm-up launch so context/JIT start-up is not timed; the warm-up changes the
       velocities, so the initial state is restored from the host copy afterwards */
    force_kernel<<<grid, block, shmem>>>(d, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(d, h, (size_t)n * sizeof(Particle), cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaEventRecord(e0));
    for (int it = 0; it < iters; it++) {
        force_kernel<<<grid, block, shmem>>>(d, n);
        position_kernel<<<grid, block>>>(d, n);
    }
    CUDA_CHECK(cudaEventRecord(e1));
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaEventSynchronize(e1));

    float ms = 0.f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, e0, e1));
    const double total = ms / 1000.0;

    /* device -> host */
    const double c1 = wall_time();
    CUDA_CHECK(cudaMemcpy(h, d, (size_t)n * sizeof(Particle), cudaMemcpyDeviceToHost));
    xfer += wall_time() - c1;

    printf("[cuda] N=%d iterations=%d block=%d grid=%d total=%.4f s avg_iter=%.4f s transfer=%.4f s (%s)\n",
           n, iters, block, grid, total, total / iters, xfer, prop.name);
    print_result("cuda", n, iters, block, total, total / iters, xfer);

    if (out) write_particles(out, h, n);

    cudaFree(d);
    free(h);
    return 0;
}
