#pragma once

#include <cstdint>
#include "cuda/ntt/arithmetic.cuh"

namespace cuda::goldilocks {

using Field = uint64_t;
// Field = a + b*phi | phi = 2^224 % MOD == 2^32
// constexpr uint phi = 1 << 224;
// struct Field {
//     uint a;
//     uint b;
// }

constexpr Field MOD = 0xffffffff00000001ull;  // goldilocks prime = 2^64 - 2^32 + 1
constexpr Field EPS = 0xffffffffull;          // epsilon = 2^32 - 1

__device__ Field add(Field a, Field b) {
    Field s = a + b;
    s -= (Field)(s >= MOD) * MOD;
    return s;
}

__device__ Field sub(Field a, Field b) {
    Field s = a - b;
    if (b > a)
        s += MOD;
    return s;
}

__device__ Field mul(Field a, Field b) {}

template <typename ButterflyConfig = cuda::ntt::Radix2Butterfly>
__device__ Field butterfly(Field* data, Field twiddle) {}
}  // namespace cuda::goldilocks