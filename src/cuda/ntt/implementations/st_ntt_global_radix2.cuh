#pragma once

#include "cuda/cu_util.cuh"
#include "cuda/ntt/arithmetic.cuh"
#include "cuda/ntt/implementations/common.cuh"
#include "cuda/ntt/implementations/st_ntt_local_radix2.cuh"

template <uint n, typename ButterflyConfig = cuda::ntt::Radix2Butterfly>
__global__ void stNttGlobalRadix2_rest(int *__restrict__ vec, int *__restrict__ out, int mod, uint step) {
    constexpr uint nPerBlock = n > 2048 ? 2048 : n;
    constexpr uint lN = cuda::log2_constexpr(n);  // log2(N)
    constexpr uint N2 = (n >> 1);

    constexpr uint numWarps = (N2 + warpSizeConst - 1) / warpSizeConst;  // Number of warps in the NTT
    constexpr uint logNumWarps = cuda::log2_constexpr(numWarps);         // log2(numWarps)
    const uint numBlocks =
        n >> (1 + cuda::log2_uint(blockDim.x));  // (n / 2) / blockDim (it is assumed all are powers of 2)

    int reg[2];

    uint idxInGroup = (blockIdx.x & (numBlocks - 1)) * blockDim.x + threadIdx.x;

    uint widx = ((blockIdx.x * blockDim.x + (blockDim.x * threadIdx.y + threadIdx.x)) / warpSize) %
                numWarps;  // Warp index in the NTT

    uint offset = 0;
    for (uint i = 1; i <= step; i++) {
        if ((widx / (numWarps >> i)) % 2)
            offset += 1 << (i - 1);
    }

    uint warpStride = numWarps >> step;
#define threadStrideRadix2Global (warpStride * warpSizeConst)
    uint wmask = (widx / warpStride) & 1;  // 0 for first half, 1 for second half

    int dPos = ((blockIdx.x >> cuda::log2_uint(numBlocks)) * n) + (idxInGroup << 1);

    // The last virtual index in the shared memory steps is used in the warp shfl steps
    uint idxVirtual = ((idxInGroup << step) & (N2 - 1)) + offset;

    reg[0] = vec[dPos + wmask * (1 + (int)threadStrideRadix2Global * 2 * (-1))];
    reg[1] = vec[dPos + wmask + !wmask * (int)threadStrideRadix2Global * 2];

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

template <uint n, typename ButterflyConfig = cuda::ntt::Radix2Butterfly>
__global__ void stNttGlobalRadix2_first(int *__restrict__ vec, int *__restrict__ out, int mod) {
    extern __shared__ int firstShfls[];

    constexpr uint lN = cuda::log2_constexpr(n);  // log2(N)
    constexpr uint N2 = (n >> 1);

    constexpr uint numWarps = (N2 + warpSizeConst - 1) / warpSizeConst;  // Number of warps in the NTT
    constexpr uint logNumWarps = cuda::log2_constexpr(numWarps);         // log2(numWarps)
    const uint numBlocks =
        n >> (1 + cuda::log2_uint(blockDim.x));  // (n / 2) / blockDim (it is assumed all are powers of 2)

    uint idxVirtual = (blockIdx.x & (numBlocks - 1)) * blockDim.x + threadIdx.x;

    int dPos = ((blockIdx.x >> cuda::log2_uint(numBlocks)) * n) + idxVirtual;

    int reg[2];

    reg[0] = vec[dPos];
    reg[1] = vec[dPos + (n >> 1)];

    // First butterfly
    cuda::ntt::butterfly<ButterflyConfig>(reg, twiddle<n>(idxVirtual), mod);

    dPos = ((blockIdx.x >> cuda::log2_uint(numBlocks)) * n) + (idxVirtual << 1);
    out[dPos] = reg[0];
    out[dPos + 1] = reg[1];
}

template <uint n, typename ButterflyConfig = cuda::ntt::Radix2Butterfly>
__host__ void sttNttGlobalRadix2(
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