#pragma once

#include "cuda/math.cuh"

namespace cuda::ntt {

struct Radix2Butterfly {};
struct EmptyButterfly {};
template <typename ButterflyConfig = Radix2Butterfly>
__device__ __forceinline__ void butterfly(int *data, int twiddle, int mod);
__device__ __forceinline__ void butterflyRadix2x2(int2 *data, int twiddle, int mod);
__device__ __forceinline__ void butterflyRadix4(int2 *data, int2 twiddle, int mod);

// IMPLEMENTATION

template <>
__device__ __forceinline__ void butterfly<Radix2Butterfly>(int *data, int twiddle, int mod) {
    int t;

    t = modulo(data[0] - data[1], mod);

    data[0] = modulo(data[0] + data[1], mod);
    data[1] = modulo(t * twiddle, mod);
}

template <>
__device__ __forceinline__ void butterfly<EmptyButterfly>(int *data, int twiddle, int mod) {}

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
}  // namespace cuda::ntt