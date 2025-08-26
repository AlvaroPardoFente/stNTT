#pragma once

#include "cuda/cu_util.cuh"
#include "cuda/ntt/arithmetic.cuh"

#define MAX_TWIDDLES 1024
__constant__ void *const_twiddles[MAX_TWIDDLES * sizeof(int)];

__global__ void stNttRadix2_512(int *__restrict__ vec, int mod) {
    // 2 buffers for the first shfls (w0 sends 2nd half to W/2, W/2 sends 1st half to w0)
    // One int for each thread
    extern __shared__ int firstShfls[];

    constexpr uint n = 512;
    constexpr uint lN = cuda::log2_constexpr(n);

    uint idxInWarp = threadIdx.x & (warpSize - 1);           // 0-31
    uint numWarps = (blockDim.x + warpSize - 1) / warpSize;  // Number of warps per NTT (256 / 32 = 8)
    uint widx = ((blockDim.x * threadIdx.y + threadIdx.x) / warpSize) % numWarps;  // 0 for w0, 1 for w1
    int dPos = (blockIdx.x * n * blockDim.y) + threadIdx.x + (threadIdx.y << lN);
    int *twiddles = (int *)const_twiddles;
    int reg[2];

    reg[0] = vec[dPos];
    reg[1] = vec[dPos + (n >> 1)];

    // First butterfly
    cuda::ntt::butterfly(reg, twiddles[threadIdx.x], mod);

    for (uint i = numWarps >> 1; i > 0; i >>= 1) {
        uint wmask = !(widx & (numWarps / 2 - 1));  // 1 for first half, 0 for second half;
        uint wstride = i * warpSize;

        firstShfls[threadIdx.x + threadIdx.y * blockDim.x] = reg[wmask];

        __syncthreads();

        uint woffset =
            wmask ? threadIdx.x + threadIdx.y * blockDim.x + wstride : threadIdx.x + threadIdx.y * blockDim.x - wstride;
        reg[wmask] = firstShfls[woffset];

        // From the first shuffle on, all idx are interleaved and a virtual index is used
        // TODO: Fix for more than one shared mem iteration
        uint idxVirtual = (idxInWarp << 1) + widx;

        cuda::ntt::butterfly(reg, twiddles[(idxVirtual >> 1) << 1], mod);
    }
}