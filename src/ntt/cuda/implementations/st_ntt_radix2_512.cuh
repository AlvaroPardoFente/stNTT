#pragma once

#include "ntt/cuda/cu_util.cuh"
#include "ntt/cuda/cu_ntt_util.cuh"
#include "ntt/cuda/implementations/common.cuh"

__global__ void stNttRadix2_512(int *__restrict__ vec, int mod) {
    // 2 buffers for the first shfl (w0 sends 2nd half to w1, w1 sends 1st half to w0)
    // One int for each thread
    extern __shared__ int firstShfls[];

    constexpr uint n = 512;                 // Number of elements in the vector
    constexpr uint lN = log2_constexpr(n);  // log2(N)

    uint idxInWarp = threadIdx.x & (warpSize - 1);                                 // 0-31
    uint numWarps = (blockDim.x + warpSize - 1) / warpSize;                        // Number of warps in the NTT
    uint widx = ((blockDim.x * threadIdx.y + threadIdx.x) / warpSize) % numWarps;  // Warp index in the block
    uint logNumWarps = log2_constexpr(numWarps);                                   // log2(numWarps)
    int dPos = (blockIdx.x * n * blockDim.y) + threadIdx.x + (threadIdx.y << lN);

    int *twiddles = (int *)const_twiddles;
    int reg[2];

    reg[0] = vec[dPos];
    reg[1] = vec[dPos + (n >> 1)];

    // First butterfly
    butterfly(reg, twiddles[threadIdx.x], mod);

    uint idxVirtual;

    int mask = 1, cont = 0;
    int offset = 0;
    for (uint step = 1; step < 4; step++) {
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
    uint lastSharedStep = 3;

    int shfl_reg[4];
    for (uint step = 4; step < lN; step++) {
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