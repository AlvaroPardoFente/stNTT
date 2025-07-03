#pragma once

#include "ntt/cuda/cu_util.cuh"
#include "ntt/cuda/cu_ntt_util.cuh"
#include "ntt/cuda/implementations/common.cuh"
#include "ntt/cuda/implementations/st_ntt_local_radix2_4096.cuh"

__global__ void stNttGlobalRadix2_4096(int *__restrict__ vec, int mod) {
    extern __shared__ int firstShfls[];

    constexpr uint n = 4096;
    constexpr uint nPerBlock = n > 2048 ? 2048 : n;
    constexpr uint lN = log2_constexpr(n);  // log2(N)
    constexpr uint N2 = (n >> 1);

    constexpr uint numWarps = (N2 + warpSizeConst - 1) / warpSizeConst;  // Number of warps in the NTT
    constexpr uint logNumWarps = log2_constexpr(numWarps);               // log2(numWarps)

    uint idxVirtual = (blockIdx.x & 1) * blockDim.x + threadIdx.x;

    int dPos = ((blockIdx.x >> 1) * n) + idxVirtual;

    // int *twiddles = (int *)const_twiddles;
    int reg[2];

    reg[0] = vec[dPos];
    reg[1] = vec[dPos + (n >> 1)];

    // First butterfly
    // butterfly(reg, twiddles[idxVirtual], mod);

    vec[dPos] = reg[0];
    vec[dPos + (n >> 1)] = reg[1];
}

void sttNttRadix2_4096(int *__restrict__ vec, int mod, dim3 dimGrid, dim3 dimBlock, uint sharedMem) {
    stNttGlobalRadix2_4096<<<dimGrid, dimBlock>>>(vec, mod);
    CCErr(cudaDeviceSynchronize());
    stNttLocalRadix2_4096<<<dimGrid, dimBlock, sharedMem>>>(vec, mod);
}