#pragma once

#include "cuda/ntt/ntt_impl.h"
#include "cuda/cu_util.cuh"

#include "cuda/ntt/implementations/st_ntt_radix2.cuh"
#include "cuda/ntt/implementations/st_ntt_radix4.cuh"
#include "cuda/ntt/implementations/st_ntt_radix2_128.cuh"
#include "cuda/ntt/implementations/st_ntt_radix2_512.cuh"
#include "cuda/ntt/implementations/st_ntt_radix2_adaptive.cuh"
#include "cuda/ntt/implementations/st_ntt_radix4_adaptive.cuh"
// #include "cuda/ntt/implementations/st_ntt_global_radix2_4096.cuh"
#include "cuda/ntt/implementations/st_ntt_global_radix2.cuh"
#include "cuda/ntt/implementations/st_ntt_goldilocks.cuh"

#include <cuda_profiler_api.h>

#include <optional>
#include <functional>
#include <map>
#include <tuple>
#include <span>

namespace cuda::ntt {
constexpr uint defaultBlockSize = 1024;
constexpr std::array<uint, 4> defaultBatchesNums = {1, 2, 5, 10};
constexpr std::array<uint, 5> defaultMaxBlockSizes = {64, 128, 256, 512, 1024};

using KernelArgs = std::tuple<dim3, dim3, uint>;
__host__ __forceinline__ KernelArgs
getNttKernelArgs(uint n, uint radix, uint batches, uint blockSize = cuda::ntt::defaultBlockSize) {
    dim3 dimGrid{((n / radix) * batches + blockSize - 1) / blockSize};
    dim3 dimBlock{std::min(n / radix, blockSize), std::max(std::min(blockSize / (n / radix), batches), 1u)};

    uint stepsInWarp = log2(warpSizeConst * radix);
    uint sharedMem = (log2(n) > stepsInWarp) ? std::min((n / 2), blockSize) * dimBlock.y * sizeof(int) : 0;

    return {dimGrid, dimBlock, sharedMem};
}

struct NttArgs {
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

    NttArgs(uint n, uint batches, uint root, uint mod, uint radix, uint blockSize = cuda::ntt::defaultBlockSize)
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
        uint blockSize = cuda::ntt::defaultBlockSize)
        : NttArgs(n, batches, root, mod, radix, blockSize) {
        this->vec = vec;
    }

    void initKernelArgs() {
        std::tie(dimGrid, dimBlock, sharedMem) = getNttKernelArgs(n, radix, batches, blockSize);
    }
};

template <size_t n>
auto chooseKernel(const NttArgs& args) {
    if constexpr (n < (1 << 6)) {
        return [args](int* vec) { stNttRadix2<n><<<args.dimGrid, args.dimBlock, args.sharedMem>>>(vec, args.mod); };
    } else if constexpr (n < (1 << 12)) {
        return [args](int* vec) {
            stNttRadix2Adaptive<n><<<args.dimGrid, args.dimBlock, args.sharedMem>>>(vec, args.mod);
        };
    } else {
        return [args](cuda::Buffer<int>& vec, cuda::Buffer<int>& doubleBuffer) {
            sttNttGlobalRadix2<n, cuda::ntt::Radix2Butterfly>(
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
__forceinline__ float stNtt(KernelFunc kernel, NttArgs args) {
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

template <size_t n>
float autoNtt(NttArgs args) {
    float gpuTime;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cuda::Buffer vecGPU(args.vec);
    cuda::Buffer<int> doubleBuffer;
    if (args.isGlobal)
        doubleBuffer.alloc(args.vec.size());

    initTwiddles<MixedTwiddles>(args.n, args.root, args.mod);

    cudaProfilerStart();
    cudaEventRecord(start);

    if (!args.isGlobal) {
        if constexpr (n < (1 << 6)) {
            stNttRadix2<n><<<args.dimGrid, args.dimBlock, args.sharedMem>>>(vecGPU.data(), args.mod);
        } else {
            stNttRadix2Adaptive<n><<<args.dimGrid, args.dimBlock, args.sharedMem>>>(vecGPU.data(), args.mod);
        }
    } else {
        sttNttGlobalRadix2<n, cuda::ntt::Radix2Butterfly>(
            vecGPU,
            doubleBuffer,
            args.batches,
            args.mod,
            args.dimGrid,
            args.dimBlock,
            args.sharedMem);
    }

    CCErr(cudaGetLastError());
    CCErr(cudaEventRecord(end));

    CCErr(cudaProfilerStop());
    CCErr(cudaEventSynchronize(end));
    CCErr(cudaEventElapsedTime(&gpuTime, start, end));

    CCErr(cudaDeviceSynchronize());

    vecGPU.store(args.vec);

    CCErr(cudaEventDestroy(start));
    CCErr(cudaEventDestroy(end));

    return gpuTime;
}
}  // namespace cuda