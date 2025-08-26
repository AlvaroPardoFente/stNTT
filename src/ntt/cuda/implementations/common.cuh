#pragma once

#include <ntt/cuda/cu_util.cuh>
#include <iostream>

#define MAX_TWIDDLES 65536
__constant__ int const_twiddles[MAX_TWIDDLES / sizeof(int)];
__device__ int* global_twiddles = nullptr;

struct GlobalTwiddles {};
struct MixedTwiddles {};
template <typename TwiddleMode = MixedTwiddles>
int initTwiddles(int n, int root, int mod);

template <>
int initTwiddles<MixedTwiddles>(int n, int root, int mod) {
    // Guard to avoid leaking. TODO: Look for a different solution. This will always leak the last buffer
    int* dev_ptr = nullptr;
    CCErr(cudaMemcpyFromSymbol(&dev_ptr, global_twiddles, sizeof(dev_ptr)));
    if (dev_ptr) {
        CCErr(cudaFree(dev_ptr));  // free the actual allocation
    }
    // write nullptr back into the device symbol
    int* nullp = nullptr;
    CCErr(cudaMemcpyToSymbol(global_twiddles, &nullp, sizeof(nullp)));

    int numTwiddles = n / 2;
    int twiddleBytes = numTwiddles * sizeof(int);
    // Should be a power of 2
    int ratio = twiddleBytes / MAX_TWIDDLES;

    // Size of twiddles fits constant memory
    if (twiddleBytes <= MAX_TWIDDLES) {
        int* hostTwiddles;
        CCErr(cudaMallocHost(&hostTwiddles, twiddleBytes));
        int temp = 1;
        for (int i = 0; i < n / 2; ++i) {
            hostTwiddles[i] = temp;
            temp = (temp * root) % mod;
        }
        CCErr(cudaMemcpyToSymbol(const_twiddles, hostTwiddles, twiddleBytes));
        CCErr(cudaFreeHost(hostTwiddles));

    } else {  // Size of twiddles does not fit constant memory

        // ratio == 2: All even twiddles go to const memory
        int* hostConstTwiddles;
        int* hostGlobalTwiddles;
        int* deviceGlobalTwiddles;
        CCErr(cudaMallocHost(&hostConstTwiddles, MAX_TWIDDLES));
        CCErr(cudaMallocHost(&hostGlobalTwiddles, twiddleBytes - MAX_TWIDDLES));
        CCErr(cudaMalloc(&deviceGlobalTwiddles, twiddleBytes - MAX_TWIDDLES));
        CCErr(cudaMemcpyToSymbol(global_twiddles, &deviceGlobalTwiddles, sizeof(int*)));

        int temp = 1, constIdx = 0, globalIdx = 0;
        for (int i = 0; i < n / 2; ++i) {
            if (i % ratio == 0)
                hostConstTwiddles[constIdx++] = temp;
            else
                hostGlobalTwiddles[globalIdx++] = temp;
            temp = (temp * root) % mod;
        }

        CCErr(cudaMemcpyToSymbol(const_twiddles, hostConstTwiddles, MAX_TWIDDLES));
        CCErr(
            cudaMemcpy(deviceGlobalTwiddles, hostGlobalTwiddles, twiddleBytes - MAX_TWIDDLES, cudaMemcpyHostToDevice));

        CCErr(cudaFreeHost(hostConstTwiddles));
        CCErr(cudaFreeHost(hostGlobalTwiddles));
    }

    return ratio;
}

// All twiddles always go to global memory
template <>
int initTwiddles<GlobalTwiddles>(int n, int root, int mod) {
    // Guard to avoid leaking. TODO: Look for a different solution. This will always leak the last buffer
    int* dev_ptr = nullptr;
    CCErr(cudaMemcpyFromSymbol(&dev_ptr, global_twiddles, sizeof(dev_ptr)));
    if (dev_ptr) {
        CCErr(cudaFree(dev_ptr));  // free the actual allocation
    }
    // write nullptr back into the device symbol
    int* nullp = nullptr;
    CCErr(cudaMemcpyToSymbol(global_twiddles, &nullp, sizeof(nullp)));

    int numTwiddles = n / 2;
    int twiddleBytes = numTwiddles * sizeof(int);

    int* hostGlobalTwiddles;
    int* deviceGlobalTwiddles;
    CCErr(cudaMallocHost(&hostGlobalTwiddles, twiddleBytes));
    CCErr(cudaMalloc(&deviceGlobalTwiddles, twiddleBytes));
    CCErr(cudaMemcpyToSymbol(global_twiddles, &deviceGlobalTwiddles, sizeof(int*)));

    int temp = 1;
    for (int i = 0; i < n / 2; ++i) {
        hostGlobalTwiddles[i] = temp;
        temp = (temp * root) % mod;
    }

    CCErr(cudaMemcpy(deviceGlobalTwiddles, hostGlobalTwiddles, twiddleBytes, cudaMemcpyHostToDevice));

    CCErr(cudaFreeHost(hostGlobalTwiddles));

    return 0;
}

// Get the twiddle factor at index i
template <uint n, typename TwiddleMode = MixedTwiddles>
__device__ int twiddle(uint i) {
    if constexpr (std::is_same_v<TwiddleMode, MixedTwiddles>) {
        // sizeof(n) being a power of 2 is assumed
        // TODO: make sizeof dependant on the actual twiddle/number type
        constexpr uint ratio = ((n / 2) * sizeof(int)) / MAX_TWIDDLES;
        constexpr uint logRatio = log2_constexpr(ratio);

        if constexpr (ratio <= 1)
            return const_twiddles[i];
        else {
            if ((i & (ratio - 1)) == 0)
                return const_twiddles[i >> logRatio];
            else {
                uint constMemSkips = (i >> logRatio) + 1;
                return global_twiddles[i - constMemSkips];
                return 0;
            }
        }
    } else if constexpr (std::is_same_v<TwiddleMode, GlobalTwiddles>) {
        return global_twiddles[i];
    }
}