#pragma once

#include "cuda/cu_util.cuh"
#include "cuda/ntt/arithmetic.cuh"
#include "cuda/ntt/implementations/common.cuh"

template <uint n>
__global__ void stNttRadix4Adaptive(int *__restrict__ vec, int mod) {
    extern __shared__ int firstShfls[];

    constexpr uint lN = cuda::log2_constexpr(n);  // log2(N)
    constexpr uint N2 = (n >> 1);
    constexpr uint N4 = (n >> 2);

    constexpr uint lastSharedStep = (lN >= 7) ? (lN - 7) : 1u;  // lN = 6 => n = 64 => Operations can be intra-warp

    constexpr uint numWarps = (N2 + warpSizeConst - 1) / warpSizeConst;  // Number of warps in the NTT
    constexpr uint logNumWarps = cuda::log2_constexpr(numWarps);         // log2(numWarps)

    int tidx = threadIdx.x;
    int tidx1 = tidx + N4;

    int dPos = (blockIdx.x * n * blockDim.y) + tidx + (threadIdx.y << lN);
    int dPos1 = (blockIdx.x * n * blockDim.y) + tidx1 + (threadIdx.y << lN);

    int *twiddles = (int *)const_twiddles;
    int reg[4];

    reg[0] = vec[dPos];
    reg[1] = vec[dPos + N2];
    reg[2] = vec[dPos1];
    reg[3] = vec[dPos1 + N2];

    // First butterfly
    cuda::ntt::butterfly(reg, twiddles[tidx], mod);
    cuda::ntt::butterfly(&reg[2], twiddles[tidx1], mod);

    // The last virtual index in the shared memory steps is used in the warp shfl steps
    uint idxVirtual = tidx;
    uint idxVirtual1 = tidx1;

    // The first swap can be performed without shared memory because every op needs values in the other regs
    cuda::swap(reg[1], reg[2]);

    idxVirtual = (tidx << 1) % N2;
    idxVirtual1 = (tidx1 << 1) % N2 + 1;

    cuda::ntt::butterfly(reg, twiddles[(idxVirtual >> 1) * (1 << 1)], mod);
    cuda::ntt::butterfly(&reg[2], twiddles[(idxVirtual1 >> 1) * (1 << 1)], mod);

    int mask = 2, cont = 1;
    if constexpr (lastSharedStep > 0) {
        int offset = 0;  // The accumulated offset for each thread's virtual id based on the previous shuffles

        uint idxInWarp = threadIdx.x & (warpSize - 1);                                 // 0-31
        uint widx = ((blockDim.x * threadIdx.y + threadIdx.x) / warpSize) % numWarps;  // Warp index in the block

        // Shared memory steps
        for (uint step = 2; step <= lastSharedStep; step++) {
            uint warpStride = numWarps >> step;
            uint threadStride = warpStride * warpSize;
            uint wmask = (widx / warpStride) & 1;  // 0 for first half, 1 for second half
            uint wreg = !wmask;

            firstShfls[blockDim.x * threadIdx.y + tidx] = reg[wreg];
            firstShfls[blockDim.x * threadIdx.y + tidx1] = reg[wreg + 2];

            __syncthreads();

            uint woffset =
                wmask ? tidx + threadIdx.y * blockDim.x - threadStride : tidx + threadIdx.y * blockDim.x + threadStride;

            uint woffset1 = wmask ? tidx1 + threadIdx.y * blockDim.x - threadStride
                                  : tidx1 + threadIdx.y * blockDim.x + threadStride;
            reg[wreg] = firstShfls[woffset];
            reg[wreg + 2] = firstShfls[woffset];

            offset += wmask * mask;
            idxVirtual = ((tidx << step) % N2) + offset;
            idxVirtual1 = ((tidx1 << step) % N2) + offset + 1;

            cuda::ntt::butterfly(reg, twiddles[(idxVirtual >> step) * (1 << step)], mod);
            cuda::ntt::butterfly(&reg[2], twiddles[(idxVirtual1 >> step) * (1 << step)], mod);

            mask = mask << 1;
            cont++;
        }
    }

    int shfl_reg[4];
    // Shfl steps
    for (uint step = lastSharedStep + 1; step < lN; step++) {
        uint threadvirtual = (((idxVirtual & (mask - 1)) + (idxVirtual >> (cont + 1) << cont)) & (N4 - 1));
        uint threadvirtual1 = (((idxVirtual1 & (mask - 1)) + (idxVirtual1 >> (cont + 1) << cont)) & (N4 - 1));
        uint swapidx = (idxVirtual & mask) != 0;
        uint swapidx1 = (idxVirtual1 & mask) != 0;

        __syncwarp();
        shfl_reg[0] = __shfl_sync(0xffffffff, reg[0], threadvirtual >> lastSharedStep);
        shfl_reg[1] = __shfl_sync(0xffffffff, reg[1], threadvirtual >> lastSharedStep);
        shfl_reg[2] = __shfl_sync(0xffffffff, reg[2], threadvirtual >> lastSharedStep);
        shfl_reg[3] = __shfl_sync(0xffffffff, reg[3], threadvirtual >> lastSharedStep);

        shfl_reg[4] = __shfl_sync(0xffffffff, reg[0], threadvirtual1 >> lastSharedStep);
        shfl_reg[5] = __shfl_sync(0xffffffff, reg[1], threadvirtual1 >> lastSharedStep);
        shfl_reg[6] = __shfl_sync(0xffffffff, reg[2], threadvirtual1 >> lastSharedStep);
        shfl_reg[7] = __shfl_sync(0xffffffff, reg[3], threadvirtual1 >> lastSharedStep);

        reg[0] = shfl_reg[swapidx];
        reg[1] = shfl_reg[swapidx + 2];
        reg[2] = shfl_reg[swapidx1 + 4];
        reg[3] = shfl_reg[swapidx1 + 6];

        cuda::ntt::butterfly(reg, twiddles[(idxVirtual >> step) * (1 << step)], mod);
        cuda::ntt::butterfly(&reg[2], twiddles[(idxVirtual1 >> step) * (1 << step)], mod);

        mask = mask << 1;
        cont++;
    }

    // dPos is calculated again to account for the interleaving done in the first stage
    dPos = (blockIdx.x * n * blockDim.y) + idxVirtual + (threadIdx.y << lN);
    dPos1 = (blockIdx.x * n * blockDim.y) + idxVirtual1 + (threadIdx.y << lN);

    vec[dPos] = reg[0];
    vec[dPos + N2] = reg[1];
    vec[dPos1] = reg[2];
    vec[dPos + N2] = reg[3];
}