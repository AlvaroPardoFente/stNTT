#pragma once

#include "ntt/cuda/cu_util.cuh"
#include "ntt/cuda/cu_ntt_util.cuh"
#include "ntt/cuda/implementations/common.cuh"

template <uint n>
__global__ void ntt_stockham_radix4(int *__restrict__ vec, int mod) {
    constexpr uint lN = log2_constexpr(n);
    constexpr uint N2 = (n >> 1);  // número de threads por FFT
    constexpr uint N4 = (n >> 2);
    constexpr uint N8 = (n >> 3);

    const uint tidx = threadIdx.x;
    const uint tidx1 = threadIdx.x + N4;

    int dPos1 = (blockIdx.x * n * blockDim.y) + tidx + (threadIdx.y << lN);  // lN = 5? log2(n)?
    int dPos2 = (blockIdx.x * n * blockDim.y) + tidx1 + (threadIdx.y << lN);
    int mask = 1, cont = 0;
    int *twiddles = (int *)const_twiddles;
    int reg[4];

    reg[0] = vec[dPos1];
    reg[1] = vec[dPos1 + N2];
    reg[2] = vec[dPos2];
    reg[3] = vec[dPos2 + N2];

    butterfly(reg, twiddles[tidx], mod);
    butterfly(&(reg[2]), twiddles[tidx1], mod);

    uint wgidx = (N4 * threadIdx.y & (warpSize - 1));  // Group index in warp
    uint gmask = ~(0xffffffff << N4) << wgidx;

    int shfl_reg[8];
    for (uint step = 1; step < lN; step++) {
        uint threadvirtual = (((tidx & (mask - 1)) + (tidx >> (cont + 1) << cont)) & (N4 - 1)) + wgidx;
        uint threadvirtual1 = (((tidx1 & (mask - 1)) + (tidx1 >> (cont + 1) << cont)) & (N4 - 1)) + wgidx;
        uint swapidx = (tidx & mask) != 0;
        uint swapidx1 = (tidx1 & mask) != 0;

        __syncwarp(gmask);

        shfl_reg[0] = __shfl_sync(gmask, reg[0], threadvirtual);
        shfl_reg[1] = __shfl_sync(gmask, reg[1], threadvirtual);
        shfl_reg[2] = __shfl_sync(gmask, reg[2], threadvirtual);
        shfl_reg[3] = __shfl_sync(gmask, reg[3], threadvirtual);

        shfl_reg[4] = __shfl_sync(gmask, reg[0], threadvirtual1);
        shfl_reg[5] = __shfl_sync(gmask, reg[1], threadvirtual1);
        shfl_reg[6] = __shfl_sync(gmask, reg[2], threadvirtual1);
        shfl_reg[7] = __shfl_sync(gmask, reg[3], threadvirtual1);

        reg[0] = shfl_reg[swapidx];
        reg[1] = shfl_reg[swapidx + 2];
        reg[2] = shfl_reg[swapidx1 + 4];
        reg[3] = shfl_reg[swapidx1 + 6];

#ifdef PRINT_STEPS_CUDA
        printfth(
            "shfl_reg[00]: %02d, shfl_reg[01]: %02d, shfl_reg[02]: %02d, shfl_reg[03]: %02d, shfl_reg[04]: %02d, "
            "shfl_reg[05]: %02d, shfl_reg[06]: %02d, shfl_reg[07]: %02d\n",
            shfl_reg[0],
            shfl_reg[1],
            shfl_reg[2],
            shfl_reg[3],
            shfl_reg[4],
            shfl_reg[5],
            shfl_reg[6],
            shfl_reg[7]);
        printfth(
            "[PRE-BUTTERFLY] reg[00]: %02d, reg[01]: %02d, reg[02]: %02d, reg[03]: %02d, threadvirtual: %02u, "
            "threadvirtual2: %02u, swapidx: %02u\n",
            reg[0],
            reg[1],
            reg[2],
            reg[3],
            threadvirtual,
            threadvirtual1,
            swapidx);
        // printff("\n");
#endif

        butterfly(reg, twiddles[(tidx >> step) * (1 << step)], mod);
        butterfly(&(reg[2]), twiddles[(tidx1 >> step) * (1 << step)], mod);

#ifdef PRINT_STEPS_CUDA
        printfth(
            "reg[0]: %02d, reg[1]: %02d, reg[2]: %02d, reg[3]: %02d, threadvirtual: %02u, threadvirtual1: %02u, "
            "swapidx: %02u, swapidx1: %02u\n",
            reg[0],
            reg[1],
            reg[2],
            reg[3],
            threadvirtual,
            threadvirtual1,
            swapidx,
            swapidx1);
        printff("\n");
#endif

        mask = mask << 1;
        cont++;
    }

    vec[dPos1] = reg[0];
    vec[dPos1 + N2] = reg[1];
    vec[dPos2] = reg[2];
    vec[dPos2 + N2] = reg[3];
}