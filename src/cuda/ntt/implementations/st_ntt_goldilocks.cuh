#pragma once

#include "cuda/cu_util.cuh"
#include "cuda/ntt/arithmetic.cuh"
#include "cuda/ntt/implementations/common.cuh"

constexpr ulong mod = 18446744069414584321;
constexpr ulong root = 6636018329409361715;

template <ulong n, typename ButterflyConfig = cuda::ntt::Radix2Butterfly>
__global__ void stNttGoldilocksGlobal_first(int *__restrict__ vec, int *__restrict__ out, int mod) {
    extern __shared__ int firstShfls[];

    constexpr uint lN = cuda::log2_constexpr(n);  // log2(N)
    constexpr uint N2 = (n >> 1);

    constexpr uint numWarps = (N2 + warpSizeConst - 1) / warpSizeConst;  // Number of warps in the NTT
    constexpr uint logNumWarps = cuda::log2_constexpr(numWarps);         // log2(numWarps)
    const uint numBlocks =
        n >> (1 + cuda::log2_uint(blockDim.x));  // (n / 2) / blockDim (it is assumed all are powers of 2)

    uint idxVirtual = (blockIdx.x & (numBlocks - 1)) * blockDim.x + threadIdx.x;

    uint dPos = ((static_cast<uint>(blockIdx.x) >> cuda::log2_uint(numBlocks)) * n) + idxVirtual;

    long reg[2];

    reg[0] = vec[dPos];
    reg[1] = vec[dPos + (n >> 1)];

    // First butterfly
    cuda::ntt::butterfly<ButterflyConfig>(reg, twiddle<n>(idxVirtual), mod);

    dPos = ((blockIdx.x >> cuda::log2_uint(numBlocks)) * n) + (idxVirtual << 1);
    out[dPos] = reg[0];
    out[dPos + 1] = reg[1];
}

template <ulong n, typename ButterflyConfig = cuda::ntt::Radix2Butterfly>
__global__ void stNttGoldiloksGlobal_rest(int *__restrict__ vec, int *__restrict__ out, int mod, uint step) {
    constexpr uint nPerBlock = n > 2048 ? 2048 : n;
    constexpr uint lN = cuda::log2_constexpr(n);  // log2(N)
    constexpr uint N2 = (n >> 1);

    constexpr uint numWarps = (N2 + warpSizeConst - 1) / warpSizeConst;  // Number of warps in the NTT
    constexpr uint logNumWarps = cuda::log2_constexpr(numWarps);         // log2(numWarps)
    const uint numBlocks =
        n >> (1 + cuda::log2_uint(blockDim.x));  // (n / 2) / blockDim (it is assumed all are powers of 2)

    long reg[2];

    uint idxInGroup = (blockIdx.x & (numBlocks - 1)) * blockDim.x + threadIdx.x;

    uint widx = ((blockIdx.x * blockDim.x + (blockDim.x * threadIdx.y + threadIdx.x)) / warpSize) %
                numWarps;  // Warp index in the NTT

    uint offset = 0;
    for (uint i = 1; i <= step; i++) {
        if ((widx / (numWarps >> i)) % 2)
            offset += 1 << (i - 1);
    }

    uint warpStride = numWarps >> step;
#define threadStrideGoldilocksGlobal (warpStride * warpSizeConst)
    uint wmask = (widx / warpStride) & 1;  // 0 for first half, 1 for second half

    uint dPos = ((blockIdx.x >> cuda::log2_uint(numBlocks)) * n) + (idxInGroup << 1);

    // The last virtual index in the shared memory steps is used in the warp shfl steps
    uint idxVirtual = ((idxInGroup << step) & (N2 - 1)) + offset;

    reg[0] = vec[dPos + wmask * (1 + (int)threadStrideGoldilocksGlobal * 2 * (-1))];
    reg[1] = vec[dPos + wmask + !wmask * (int)threadStrideGoldilocksGlobal * 2];

    // First butterfly
    cuda::ntt::butterfly<ButterflyConfig>(reg, twiddle<n>((idxVirtual >> step) * (1 << step)), mod);

    if (false && step == lN - cuda::log2_uint(nPerBlock) - 1) {  // Final (for debugging)
        dPos = ((blockIdx.x >> cuda::log2_uint(numBlocks)) * n) + ((idxVirtual >> step) << (step + 1)) +
               (idxVirtual & ((1 << step) - 1));
        out[dPos] = reg[0];
        out[dPos + (1 << step)] = reg[1];
    } else {
        // Between kernels
        out[dPos] = reg[0];
        out[dPos + 1] = reg[1];
    }
}

template <ulong n, typename ButterflyConfig = cuda::ntt::Radix2Butterfly>
__global__ void stNttGoldilocksLocal(int *__restrict__ vec, int *__restrict__ out, int mod, uint step) {
    extern __shared__ int firstShfls[];

    constexpr uint lN = cuda::log2_constexpr(n);  // log2(N)
    constexpr uint N2 = (n >> 1);

    constexpr uint lastSharedStep = (lN >= 6) ? (lN - 6) : 0u;  // lN = 6 => n = 64 => Operations can be intra-warp

    constexpr uint numWarps = (N2 + warpSizeConst - 1) / warpSizeConst;  // Number of warps in the NTT
    constexpr uint logNumWarps = cuda::log2_constexpr(numWarps);         // log2(numWarps)
    const uint numBlocks =
        n >> (1 + cuda::log2_uint(blockDim.x));  // (n / 2) / blockDim (it is assumed all are powers of 2)

    uint idxInGroup = (blockIdx.x & (numBlocks - 1)) * blockDim.x + threadIdx.x;

    long reg[2];

    // All warp arithmetic is needed here to calculate the first offsets if step > 0
    uint widx = ((blockIdx.x * blockDim.x + (blockDim.x * threadIdx.y + threadIdx.x)) / warpSize) %
                numWarps;  // Warp index in the NTT

    uint offset = 0;
    for (uint i = 1; i <= step; i++) {
        if ((widx / (numWarps >> i)) % 2)
            offset += 1 << (i - 1);
    }
    // uint offset = (widx >> (logNumWarps - step)) & ((1u << step) - 1);

    uint warpStride = numWarps >> step;
#define threadStrideGoldilocksLocal (warpStride * warpSizeConst)
    uint wmask = (widx / warpStride) & 1;  // 0 for first half, 1 for second half

    uint dPos = ((static_cast<uint>(blockIdx.x) >> cuda::log2_uint(numBlocks)) * n) + (idxInGroup << 1);

    // The last virtual index in the shared memory steps is used in the warp shfl steps
    uint idxVirtual = ((idxInGroup << step) & (N2 - 1)) + offset;

    reg[0] = vec[dPos + wmask * (1 + (int)threadStrideGoldilocksLocal * 2 * (-1))];
    reg[1] = vec[dPos + wmask + !wmask * (int)threadStrideGoldilocksLocal * 2];

    // First butterfly
    cuda::ntt::butterfly<ButterflyConfig>(reg, twiddle<n>((idxVirtual >> step) * (1 << step)), mod);

    // Formula for debugging: it stores all data unshuffled
    // dPos = ((blockIdx.x >> log2_uint(numBlocks)) * n) + ((idxVirtual >> step) << (step + 1)) +
    //        (idxVirtual & ((1 << step) - 1));
    // vec[dPos] = reg[0];
    // vec[dPos + (1 << step)] = reg[1];

    step++;
    uint mask = 1 << (step - 1), cont = step - 1;

    if constexpr (lastSharedStep > 0) {
        // Shared memory steps
        for (; step <= lastSharedStep; step++) {
            warpStride = numWarps >> step;
            wmask = (widx / warpStride) & 1;  // 0 for first half, 1 for second half

            firstShfls[blockDim.x * threadIdx.y + threadIdx.x] = reg[!wmask];

            __syncthreads();

            uint woffset = wmask ? threadIdx.x + threadIdx.y * blockDim.x - threadStrideGoldilocksLocal
                                 : threadIdx.x + threadIdx.y * blockDim.x + threadStrideGoldilocksLocal;
            reg[!wmask] = firstShfls[woffset];

            offset += wmask * mask;
            idxVirtual = ((idxInGroup << step) & (N2 - 1)) + offset;

            cuda::ntt::butterfly<ButterflyConfig>(reg, twiddle<n>((idxVirtual >> step) * (1 << step)), mod);

            mask = mask << 1;
            cont++;
        }
    }

    int shfl_reg[4];
    // Shfl steps
    for (; step < lN; step++) {
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

        cuda::ntt::butterfly<ButterflyConfig>(reg, twiddle<n>((idxVirtual >> step) * (1 << step)), mod);

        mask = mask << 1;
        cont++;
    }

    // dPos is calculated again to account for interleaving
    dPos = ((static_cast<uint>(blockIdx.x) >> cuda::log2_uint(numBlocks)) * n) + idxVirtual;

    out[dPos] = reg[0];
    out[dPos + (n >> 1)] = reg[1];
}

template <ulong n, typename ButterflyConfig = cuda::ntt::Radix2Butterfly>
__host__ void sttNttGoldilocks(
    cuda::Buffer<int> &vec,
    cuda::Buffer<int> &doubleBuffer,
    int batches,
    int mod,
    dim3 dimGrid,
    dim3 dimBlock,
    uint sharedMem) {
    constexpr uint lN = cuda::log2_constexpr(n);
    const uint nPerBlock = n > dimBlock.x * 2 ? dimBlock.x * 2 : n;
    const uint logNPerBlock = cuda::log2_uint(nPerBlock);
    // At blockDim.x = 1024, nPerBlock is 2048 and last step is lN - 11 - 1 (-1 because the local kernel will do one
    // global access)
    const uint lastGlobalStep = (lN >= logNPerBlock) ? (lN - logNPerBlock) - 1 : 0u;

    // TODO: Redo all swapping logic (possibly take two cuda::Buffers args instead of creating one)
    stNttGlobalRadix2_first<n, ButterflyConfig><<<dimGrid, dimBlock>>>(vec.data(), doubleBuffer.data(), mod);
    std::swap(vec, doubleBuffer);

    CCErr(cudaDeviceSynchronize());
    for (uint step = 1; step <= lastGlobalStep; step++) {
        stNttGlobalRadix2_rest<n, ButterflyConfig><<<dimGrid, dimBlock>>>(vec.data(), doubleBuffer.data(), mod, step);
        CCErr(cudaDeviceSynchronize());
        std::swap(vec, doubleBuffer);
    }
    stNttLocalRadix2<n, ButterflyConfig>
        <<<dimGrid, dimBlock, sharedMem>>>(vec.data(), doubleBuffer.data(), mod, lastGlobalStep + 1);
    std::swap(vec, doubleBuffer);
}