#include "ntt/cuda/ntt_kernel.h"
#include "ntt/cuda/cu_util.cuh"
#include "ntt/cuda/cu_ntt_util.cuh"

#include "ntt/cuda/implementations/st_ntt_radix2.cuh"
#include "ntt/cuda/implementations/st_ntt_radix4.cuh"
#include "ntt/cuda/implementations/st_ntt_radix2_128.cuh"
#include "ntt/cuda/implementations/st_ntt_radix2_512.cuh"
#include "ntt/cuda/implementations/st_ntt_radix2_adaptive.cuh"
#include "ntt/cuda/implementations/st_ntt_radix4_adaptive.cuh"

#include <cuda_profiler_api.h>
#include <algorithm>

namespace cuda {

using Kernel = void (*)(int *vec, int mod);

template <uint n, uint radix>
void registerNttKernel(KernelMap &kernels, Kernel kernel, uint batches, ntt::KernelId impl_id) {
    auto [dimGrid, dimBlock, sharedMem] = getNttKernelArgs(n, radix, batches);
    kernels[{n, batches, impl_id}] = [kernel, dimGrid, dimBlock, sharedMem](int *vec, int mod) {
        kernel<<<dimGrid, dimBlock, sharedMem>>>(vec, mod);
    };
}

#define REGISTER_NTT_KERNELS(kernels, batches, kernel_template, radix, impl_id, ...)                   \
    do {                                                                                               \
        auto sizes = std::integer_sequence<uint, __VA_ARGS__>{};                                       \
        ([&]<uint... Sizes>(std::integer_sequence<uint, Sizes...>) {                                   \
            (registerNttKernel<Sizes, radix>(kernels, kernel_template<Sizes>, batches, impl_id), ...); \
        })(sizes);                                                                                     \
    } while (0)

void registerAllKernels(KernelMap &kernels, std::span<const uint> batchesNums) {
    for (auto batches : batchesNums) {
        // clang-format off
        REGISTER_NTT_KERNELS(kernels, batches, stNttRadix2, 2, ntt::KernelId::stNttRadix2, 2, 4, 8, 16, 32, 64);
        REGISTER_NTT_KERNELS(kernels, batches, stNttRadix2Adaptive, 2, ntt::KernelId::stNttRadix2Adaptive, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048);
        REGISTER_NTT_KERNELS(kernels, batches, stNttRadix2_128, 2, ntt::KernelId::stNttRadix2_128, 128);
        REGISTER_NTT_KERNELS(kernels, batches, stNttRadix2_512, 2, ntt::KernelId::stNttRadix2_512, 512);
        REGISTER_NTT_KERNELS(kernels, batches, stNttRadix4, 4, ntt::KernelId::stNttRadix4, 4, 8, 16, 32, 64, 128);
        // clang-format on
    }
}

float stNtt(KernelMap &kernels, std::span<int> vec, uint n, uint batches, ntt::KernelId id, int root, int mod) {
    KernelKey key{n, batches, id};

    float gpuTime;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cuda::Buffer vecGPU(vec);

    int *host_twiddles;
    CCErr(cudaMallocHost(&host_twiddles, n * sizeof(int)));
    int temp = 1;
    for (uint i = 0; i < n / 2; ++i) {
        host_twiddles[i] = temp;
        temp = (temp * root) % mod;
    }

    CCErr(cudaMemcpyToSymbol(const_twiddles, host_twiddles, n / 2 * sizeof(int)));

    auto &kernel = kernels[key];

    cudaProfilerStart();
    cudaEventRecord(start);
    kernel(vecGPU.data(), mod);
    CCErr(cudaGetLastError());
    CCErr(cudaEventRecord(end));

    CCErr(cudaProfilerStop());
    CCErr(cudaEventSynchronize(end));
    CCErr(cudaEventElapsedTime(&gpuTime, start, end));

    CCErr(cudaDeviceSynchronize());

    vecGPU.store(vec);
    CCErr(cudaFreeHost(host_twiddles));

    return gpuTime;
}

float stNtt(std::function<void(int *vec)> kernel, std::span<int> vec, uint n, uint batches, int root, int mod) {
    float gpuTime;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cuda::Buffer vecGPU(vec);

    int *host_twiddles;
    CCErr(cudaMallocHost(&host_twiddles, n * sizeof(int)));
    int temp = 1;
    for (uint i = 0; i < n / 2; ++i) {
        host_twiddles[i] = temp;
        temp = (temp * root) % mod;
    }

    CCErr(cudaMemcpyToSymbol(const_twiddles, host_twiddles, n / 2 * sizeof(int)));

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
    CCErr(cudaFreeHost(host_twiddles));

    return gpuTime;
}
}  // namespace cuda