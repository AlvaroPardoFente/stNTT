#include "cuda/cu_util.cuh"
#include "cuda/ntt/arithmetic.cuh"
#include "cuda/ntt/implementations/common.cuh"

template <typename ButterflyConfig = cuda::ntt::Radix2Butterfly>
__global__ void stNttLocalRadix2_4096(int *__restrict__ vec, int mod) {
    extern __shared__ int firstShfls[];

    uint step = 1;
    constexpr uint n = 4096;
    constexpr uint lN = cuda::log2_constexpr(n);  // log2(N)
    constexpr uint N2 = (n >> 1);

    constexpr uint lastSharedStep = (lN >= 6) ? (lN - 6) : 0u;  // lN = 6 => n = 64 => Operations can be intra-warp

    constexpr uint numWarps = (N2 + warpSizeConst - 1) / warpSizeConst;  // Number of warps in the NTT
    constexpr uint logNumWarps = cuda::log2_constexpr(numWarps);         // log2(numWarps)

    uint idxInGroup = (blockIdx.x & 1) * blockDim.x + threadIdx.x;

    int *twiddles = (int *)const_twiddles;
    int reg[2];

    // All warp arithmetic is needed here to calculate the first offsets if step > 0
    uint idxInWarp = threadIdx.x & (warpSize - 1);  // 0-31
    uint widx = ((blockIdx.x * blockDim.x + (blockDim.x * threadIdx.y + threadIdx.x)) / warpSize) %
                numWarps;  // Warp index in the NTT

    uint offset = 0;
    for (uint i = 1; i <= step; i++) {
        if (widx / (1 << (logNumWarps - i)))
            offset += 1 << (i - 1);
    }
    // uint offset = (widx >> (logNumWarps - step)) & ((1u << step) - 1);

    uint warpStride = numWarps >> step;
#define threadStrideRadix2_4096 (warpStride * warpSizeConst)
    uint wmask = (widx / warpStride) & 1;  // 0 for first half, 1 for second half

    int dPos = (((blockIdx.x >> 1) * n) + idxInGroup) << 1;

    // The last virtual index in the shared memory steps is used in the warp shfl steps
    uint idxVirtual = ((idxInGroup << step) & (N2 - 1)) + offset;

    reg[0] = vec[dPos + wmask * (1 + (int)threadStrideRadix2_4096 * 2 * (-1))];
    reg[1] = vec[dPos + wmask + !wmask * (int)threadStrideRadix2_4096 * 2];

    // First butterfly
    cuda::ntt::butterfly<ButterflyConfig>(reg, twiddles[(idxVirtual >> step) * (1 << step)], mod);

    int mask = 2, cont = 1;

    if constexpr (lastSharedStep > 0) {
        // Shared memory steps
        for (step++; step <= lastSharedStep; step++) {
            warpStride = numWarps >> step;
            wmask = (widx / warpStride) & 1;  // 0 for first half, 1 for second half

            firstShfls[blockDim.x * threadIdx.y + threadIdx.x] = reg[!wmask];

            __syncthreads();

            uint woffset = wmask ? threadIdx.x + threadIdx.y * blockDim.x - threadStrideRadix2_4096
                                 : threadIdx.x + threadIdx.y * blockDim.x + threadStrideRadix2_4096;
            reg[!wmask] = firstShfls[woffset];

            offset += wmask * mask;
            idxVirtual = ((idxInGroup << step) & (N2 - 1)) + offset;

            cuda::ntt::butterfly<ButterflyConfig>(reg, twiddles[(idxVirtual >> step) * (1 << step)], mod);

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

        __syncwarp(0xffffffff);

        shfl_reg[0] = __shfl_sync(0xffffffff, reg[0], (threadvirtual >> lastSharedStep));
        shfl_reg[1] = __shfl_sync(0xffffffff, reg[1], (threadvirtual >> lastSharedStep));
        shfl_reg[2] = __shfl_sync(0xffffffff, reg[0], (threadvirtual2 >> lastSharedStep));
        shfl_reg[3] = __shfl_sync(0xffffffff, reg[1], (threadvirtual2 >> lastSharedStep));

        reg[0] = shfl_reg[swapidx];
        reg[1] = shfl_reg[swapidx + 2];

        cuda::ntt::butterfly<ButterflyConfig>(reg, twiddles[(idxVirtual >> step) * (1 << step)], mod);

        mask = mask << 1;
        cont++;
    }

    // dPos is calculated again to account for the interleaving done in the first stage
    // dPos = ((blockIdx.x >> 1) * n) + ((idxVirtual >> step) << (step + 1)) + (idxVirtual & ((1 << step) - 1));
    dPos = ((blockIdx.x >> 1) * n) + idxVirtual;

    vec[dPos] = reg[0];
    vec[dPos + (n >> 1)] = reg[1];
}