#pragma once

namespace cuda {
__device__ __forceinline__ int modulo(int x, int mod);
__host__ __device__ __forceinline__ unsigned int log2_uint(unsigned int x);
__host__ __device__ constexpr uint log2_constexpr(uint x);

// IMPLEMENTATION

__device__ __forceinline__ int modulo(int x, int mod) {
    int result = x % mod;
    if (result >= 0)
        return result;
    else
        return result + mod;
}

__host__ __device__ __forceinline__ unsigned int log2_uint(unsigned int x) {
#if defined(__CUDA_ARCH__)
    return 31 - __clz(x);
#else
    return 31 - __builtin_clz(x);
#endif
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

}  // namespace cuda