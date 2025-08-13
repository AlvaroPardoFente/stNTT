#pragma once

#include "ntt/cuda/ntt_impl.h"
#include "ntt/cuda/cu_util.cuh"
#include "ntt/cuda/cu_ntt_util.cuh"

#include "ntt/cuda/implementations/st_ntt_radix2.cuh"
#include "ntt/cuda/implementations/st_ntt_radix4.cuh"
#include "ntt/cuda/implementations/st_ntt_radix2_128.cuh"
#include "ntt/cuda/implementations/st_ntt_radix2_512.cuh"
#include "ntt/cuda/implementations/st_ntt_radix2_adaptive.cuh"
#include "ntt/cuda/implementations/st_ntt_radix4_adaptive.cuh"
// #include "ntt/cuda/implementations/st_ntt_global_radix2_4096.cuh"
#include "ntt/cuda/implementations/st_ntt_global_radix2.cuh"
#include "ntt/cuda/implementations/st_ntt_goldilocks.cuh"

#include <cuda_profiler_api.h>

#include <optional>
#include <functional>
#include <map>
#include <span>

namespace cuda {
constexpr std::array<uint, 6> defaultBatchesNums = {1, 2, 5, 10, 100, 1000};

__host__ struct NttArgs {
    uint n{};
    uint batches{};
    uint root{};
    uint mod{};
    uint radix{};

    uint blockSize{};
    dim3 dimGrid{};
    dim3 dimBlock{};
    uint sharedMem{};
    // True if global memory and double buffering are needed
    bool isGlobal{};

    std::span<int> vec;

    // This is needed to keep the args in the stats
    NttArgs() = default;

    NttArgs(uint n, uint batches, uint root, uint mod, uint radix, uint blockSize = cuda::defaultBlockSize)
        : n(n), batches(batches), root(root), mod(mod), radix(radix), blockSize(blockSize) {
        initKernelArgs();
        if (n / radix > dimBlock.x)
            isGlobal = true;
    }

    NttArgs(
        std::span<int> vec,
        uint n,
        uint batches,
        uint root,
        uint mod,
        uint radix,
        uint blockSize = cuda::defaultBlockSize)
        : NttArgs(n, batches, root, mod, radix, blockSize) {
        this->vec = vec;
    }

    void initKernelArgs() {
        std::tie(dimGrid, dimBlock, sharedMem) = cuda::getNttKernelArgs(n, radix, batches, blockSize);
    }
};

// struct KernelKey {
//     uint n;
//     uint batches;
//     ntt::KernelId impl_id;

//     auto operator<=>(const KernelKey &) const = default;
// };

// using KernelLauncher = std::function<void(int *, int)>;
// using KernelMap = std::map<KernelKey, KernelLauncher>;

// void registerAllKernels(KernelMap &kernels, std::span<const uint> batchesNums);

// float stNtt(KernelMap &kernels, std::span<int> vec, uint n, uint batches, ntt::KernelId id, int root, int mod);
__forceinline__ float
stNtt(std::function<void(int* vec)> kernel, std::span<int> vec, uint n, uint batches, int root, int mod) {
    float gpuTime;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cuda::Buffer vecGPU(vec);

    initTwiddles(n, root, mod);

    cudaProfilerStart();
    cudaEventRecord(start);
    kernel(vecGPU.data());
    CCErr(cudaGetLastError());
    CCErr(cudaEventRecord(end));

    CCErr(cudaProfilerStop());
    CCErr(cudaEventSynchronize(end));
    CCErr(cudaEventElapsedTime(&gpuTime, start, end));

    CCErr(cudaDeviceSynchronize());

    vecGPU.store(vec);

    return gpuTime;
}

template <size_t n>
auto chooseKernel(const cuda::NttArgs& args) {
    if constexpr (n < (1 << 6)) {
        return [args](int* vec) { stNttRadix2<n><<<args.dimGrid, args.dimBlock, args.sharedMem>>>(vec, args.mod); };
    } else if constexpr (n < (1 << 12)) {
        return [args](int* vec) {
            stNttRadix2Adaptive<n><<<args.dimGrid, args.dimBlock, args.sharedMem>>>(vec, args.mod);
        };
    } else {
        return [args](cuda::Buffer<int>& vec, cuda::Buffer<int>& doubleBuffer) {
            sttNttGlobalRadix2<n, Radix2Butterfly>(
                vec,
                doubleBuffer,
                args.batches,
                args.mod,
                args.dimGrid,
                args.dimBlock,
                args.sharedMem);
        };
    }
}

template <typename T>
concept LocalKernel = requires(T kernel, int* ptr) { kernel(ptr); };

template <typename T>
concept GlobalKernel = requires(T kernel, int* ptr1, int* ptr2) { kernel(ptr1, ptr2); };

template <typename T>
concept DoubleBufferKernel = requires(T kernel, cuda::Buffer<int>& in, cuda::Buffer<int>& out) { kernel(in, out); };

template <typename KernelFunc>
    requires LocalKernel<KernelFunc> || GlobalKernel<KernelFunc> || DoubleBufferKernel<KernelFunc>
__forceinline__ float ntt(KernelFunc kernel, cuda::NttArgs args) {
    float gpuTime;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cuda::Buffer vecGPU(args.vec);
    cuda::Buffer<int> doubleBuffer;
    if (args.isGlobal || GlobalKernel<KernelFunc>)
        doubleBuffer.alloc(args.vec.size());

    initTwiddles(args.n, args.root, args.mod);

    cudaProfilerStart();
    cudaEventRecord(start);
    if constexpr (LocalKernel<KernelFunc>)
        kernel(vecGPU.data());
    else if constexpr (GlobalKernel<KernelFunc>)
        kernel(vecGPU.data(), doubleBuffer.data());
    else if constexpr (DoubleBufferKernel<KernelFunc>)
        kernel(vecGPU, doubleBuffer);
    CCErr(cudaGetLastError());
    CCErr(cudaEventRecord(end));

    CCErr(cudaProfilerStop());
    CCErr(cudaEventSynchronize(end));
    CCErr(cudaEventElapsedTime(&gpuTime, start, end));

    CCErr(cudaDeviceSynchronize());

    vecGPU.store(args.vec);

    return gpuTime;
}
}  // namespace cuda