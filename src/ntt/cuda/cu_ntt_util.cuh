#pragma once

#include "ntt/cuda/cu_util.cuh"

__forceinline__ __device__ int modulo(int x, int mod);
__device__ __forceinline__ unsigned int log2_uint(unsigned int x);
__host__ __device__ constexpr uint log2_constexpr(uint x);

__forceinline__ __device__ void butterfly(int *data, int twiddle, int mod);
__forceinline__ __device__ void butterflyRadix2x2(int2 *data, int twiddle, int mod);
__forceinline__ __device__ void butterflyRadix4(int2 *data, int2 twiddle, int mod);

// IMPLEMENTATION

__forceinline__ __device__ int modulo(int x, int mod) {
    int result = x % mod;
    if (result >= 0)
        return result;
    else
        return result + mod;
}

__forceinline__ __device__ unsigned int log2_uint(unsigned int x) {
    return 31 - __clz(x);
}

__host__ __device__ __forceinline__ uint log2(uint x) {
    uint res = 0;
    while (x >>= 1)
        ++res;
    return res;
}

__host__ __device__ constexpr uint log2_constexpr(uint x) {
    return (x <= 1) ? 0 : 1 + log2_constexpr(x >> 1);
}

__device__ __forceinline__ void butterfly(int *data, int twiddle, int mod) {
    int t;

    t = modulo(data[0] - data[1], mod);

    data[0] = modulo(data[0] + data[1], mod);
    data[1] = modulo(t * twiddle, mod);
}

__device__ __forceinline__ void butterflyRadix2x2(int2 *data, int twiddle, int mod) {
    int2 t;

    t.x = modulo(data[0].x - data[1].x, mod);
    t.y = modulo(data[0].y - data[1].y, mod);

    data[0].x = modulo(data[0].x + data[1].x, mod);
    data[1].x = modulo(t.x * twiddle, mod);
    data[0].y = modulo(data[0].y + data[1].y, mod);
    data[1].y = modulo(t.y * twiddle, mod);
}

__device__ __forceinline__ void butterflyRadix4(int2 *data, int2 twiddle, int mod) {
    int2 t;

    t.x = modulo(data[0].x - data[1].x, mod);
    t.y = modulo(data[0].y - data[1].y, mod);

    data[0].x = modulo(data[0].x + data[1].x, mod);
    data[1].x = modulo(t.x * twiddle.x, mod);
    data[0].y = modulo(data[0].y + data[1].y, mod);
    data[1].y = modulo(t.y * twiddle.y, mod);
}

namespace cuda {
constexpr uint defaultBlockSize = 1024;

using KernelArgs = std::tuple<dim3, dim3, uint>;
__forceinline__ KernelArgs getNttKernelArgs(uint n, uint radix, uint batches, uint blockSize = cuda::defaultBlockSize) {
    dim3 dimGrid{((n / radix) * batches + blockSize - 1) / blockSize};
    dim3 dimBlock{std::min(n / radix, blockSize), std::max(std::min(blockSize / (n / radix), batches), 1u)};

    uint stepsInWarp = log2(warpSizeConst * radix);
    uint sharedMem = (log2(n) > stepsInWarp) ? std::min((n / 2), blockSize) * dimBlock.y * sizeof(int) : 0;

    return {dimGrid, dimBlock, sharedMem};
}
}  // namespace cuda