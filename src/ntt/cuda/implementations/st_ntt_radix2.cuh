#pragma once

#include "ntt/cuda/cu_util.cuh"
#include "ntt/cuda/cu_ntt_util.cuh"

#define MAX_TWIDDLES 1024
__constant__ void *const_twiddles[MAX_TWIDDLES * sizeof(int)];

template <uint n, uint lN>
__global__ static void stNttRadix2(int *__restrict__ vec, int mod) {
    constexpr uint N2 = (n >> 1);  // nthreads per NTT
    constexpr uint N4 = (n >> 2);

    int dPos = (blockIdx.x * n * blockDim.y) + threadIdx.x + (threadIdx.y << lN);
    int mask = 1, cont = 0;
    int *twiddles = (int *)const_twiddles;
    int reg[2];

    reg[0] = vec[dPos];
    reg[1] = vec[dPos + N2];

    // Size 2
    butterfly(reg, twiddles[threadIdx.x], mod);

    uint wgidx = (N2 * threadIdx.y & (warpSize - 1));  // Group index in warp
    uint gmask = ~(0xffffffff << N2) << wgidx;

    int shfl_reg[4];
    for (uint step = 1; step < lN; step++) {
        uint threadvirtual = (((threadIdx.x & (mask - 1)) + (threadIdx.x >> (cont + 1) << cont)) & (N4 - 1)) + wgidx;
        uint threadvirtual2 = threadvirtual + N4;
        uint swapidx = (threadIdx.x & mask) != 0;

        __syncwarp(gmask);

        shfl_reg[0] = __shfl_sync(gmask, reg[0], threadvirtual);
        shfl_reg[1] = __shfl_sync(gmask, reg[1], threadvirtual);
        shfl_reg[2] = __shfl_sync(gmask, reg[0], threadvirtual2);
        shfl_reg[3] = __shfl_sync(gmask, reg[1], threadvirtual2);

        reg[0] = shfl_reg[swapidx];
        reg[1] = shfl_reg[swapidx + 2];

        butterfly(reg, twiddles[(threadIdx.x >> step) * (1 << step)], mod);

        mask = mask << 1;
        cont++;
    }

    vec[dPos] = reg[0];
    vec[dPos + N2] = reg[1];
}