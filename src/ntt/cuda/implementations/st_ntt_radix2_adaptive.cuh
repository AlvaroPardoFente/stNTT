#pragma once

#include "ntt/cuda/cu_util.cuh"
#include "ntt/cuda/cu_ntt_util.cuh"
#include "ntt/cuda/implementations/common.cuh"

template <uint n>
__global__ void stNttRadix2Adaptive(int *__restrict__ vec, int mod) {
    extern __shared__ int firstShfls[];

    constexpr uint lN = log2_constexpr(n);  // log2(N)
    constexpr uint N2 = (n >> 1);

    constexpr uint lastSharedStep = (lN >= 6) ? (lN - 6) : 0u;  // lN = 6 => n = 64 => Operations can be intra-warp

    constexpr uint numWarps = (N2 + warpSizeConst - 1) / warpSizeConst;  // Number of warps in the NTT
    constexpr uint logNumWarps = log2_constexpr(numWarps);               // log2(numWarps)

    int dPos = (blockIdx.x * n * blockDim.y) + threadIdx.x + (threadIdx.y << lN);

    int *twiddles = (int *)const_twiddles;
    int reg[2];

    reg[0] = vec[dPos];
    reg[1] = vec[dPos + (n >> 1)];

    // First butterfly
    butterfly(reg, twiddles[threadIdx.x], mod);

    // The last virtual index in the shared memory steps is used in the warp shfl steps
    uint idxVirtual = threadIdx.x;

    int mask = 1, cont = 0;

    if constexpr (lastSharedStep > 0) {
        int offset = 0;  // The accumulated offset for each thread's virtual id based on the previous shuffles

        uint idxInWarp = threadIdx.x & (warpSize - 1);                                 // 0-31
        uint widx = ((blockDim.x * threadIdx.y + threadIdx.x) / warpSize) % numWarps;  // Warp index in the block

        // Shared memory steps
        for (uint step = 1; step <= lastSharedStep; step++) {
            uint warpStride = numWarps >> step;
            uint threadStride = warpStride * warpSize;
            uint wmask = (widx / warpStride) & 1;  // 0 for first half, 1 for second half
            uint wreg = !wmask;

            firstShfls[blockDim.x * threadIdx.y + threadIdx.x] = reg[wreg];

            __syncthreads();

            uint woffset = wmask ? threadIdx.x + threadIdx.y * blockDim.x - threadStride
                                 : threadIdx.x + threadIdx.y * blockDim.x + threadStride;
            reg[wreg] = firstShfls[woffset];

            offset += wmask * mask;
            idxVirtual = ((threadIdx.x << step) % blockDim.x) + offset;

            butterfly(reg, twiddles[(idxVirtual >> step) * (1 << step)], mod);

            mask = mask << 1;
            cont++;
        }
    }

    int shfl_reg[4];
    // Shfl steps
    for (uint step = lastSharedStep + 1; step < lN; step++) {
        uint threadvirtual = (((idxVirtual & (mask - 1)) + (idxVirtual >> (cont + 1) << cont)) & ((n >> 2) - 1));
        uint threadvirtual2 = threadvirtual + (n >> 2);
        uint swapidx = (idxVirtual & mask) != 0;

        __syncwarp();

        shfl_reg[0] = __shfl_sync(0xffffffff, reg[0], (threadvirtual >> lastSharedStep));
        shfl_reg[1] = __shfl_sync(0xffffffff, reg[1], (threadvirtual >> lastSharedStep));
        shfl_reg[2] = __shfl_sync(0xffffffff, reg[0], (threadvirtual2 >> lastSharedStep));
        shfl_reg[3] = __shfl_sync(0xffffffff, reg[1], (threadvirtual2 >> lastSharedStep));

        reg[0] = shfl_reg[swapidx];
        reg[1] = shfl_reg[swapidx + 2];

        butterfly(reg, twiddles[(idxVirtual >> step) * (1 << step)], mod);

        mask = mask << 1;
        cont++;
    }

    // dPos is calculated again to account for the interleaving done in the first stage
    dPos = (blockIdx.x * n * blockDim.y) + idxVirtual + (threadIdx.y << lN);

    vec[dPos] = reg[0];
    vec[dPos + (n >> 1)] = reg[1];
}