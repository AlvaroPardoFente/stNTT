#pragma once

#include "cuda/cu_util.cuh"
#include "cuda/ntt/arithmetic.cuh"
#include "cuda/ntt/implementations/common.cuh"
#include "cuda/ntt/implementations/st_ntt_local_radix2_4096.cuh"

template <typename ButterflyConfig = cuda::ntt::Radix2Butterfly>
__global__ void stNttGlobalRadix2_4096(int *__restrict__ vec, int mod) {
    extern __shared__ int firstShfls[];

    constexpr uint n = 4096;
    constexpr uint nPerBlock = n > 2048 ? 2048 : n;
    constexpr uint lN = cuda::log2_constexpr(n);  // log2(N)
    constexpr uint N2 = (n >> 1);

    constexpr uint numWarps = (N2 + warpSizeConst - 1) / warpSizeConst;  // Number of warps in the NTT
    constexpr uint logNumWarps = cuda::log2_constexpr(numWarps);         // log2(numWarps)

    uint idxVirtual = (blockIdx.x & 1) * blockDim.x + threadIdx.x;

    int dPos = ((blockIdx.x >> 1) * n) + idxVirtual;

    int *twiddles = (int *)const_twiddles;
    int reg[2];

    reg[0] = vec[dPos];
    reg[1] = vec[dPos + (n >> 1)];

    // First butterfly
    cuda::ntt::butterfly<ButterflyConfig>(reg, twiddles[idxVirtual], mod);

    dPos <<= 1;
    vec[dPos] = reg[0];
    vec[dPos + 1] = reg[1];
}

template <typename ButterflyConfig = cuda::ntt::Radix2Butterfly>
void sttNttRadix2_4096(int *__restrict__ vec, int mod, dim3 dimGrid, dim3 dimBlock, uint sharedMem) {
    stNttGlobalRadix2_4096<ButterflyConfig><<<dimGrid, dimBlock>>>(vec, mod);
    CCErr(cudaDeviceSynchronize());
    stNttLocalRadix2_4096<ButterflyConfig><<<dimGrid, dimBlock, sharedMem>>>(vec, mod);
}