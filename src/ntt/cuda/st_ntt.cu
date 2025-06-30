#include "st_ntt.h"

#include "ntt/cuda/implementations/st_ntt_radix2.cuh"
#include "ntt/cuda/implementations/st_ntt_radix2_128.cuh"
#include "ntt/cuda/implementations/st_ntt_radix2_512.cuh"
#include "ntt/cuda/implementations/st_ntt_radix2_adaptive.cuh"
#include "ntt/cuda/implementations/st_ntt_radix4_adaptive.cuh"

#include <cuda_profiler_api.h>

#include <map>

using nttkernel = void (*)(int *vec, int mod);

std::map<uint, nttkernel> radix2{
    {2, stNttRadix2<2>},
    {4, stNttRadix2<4>},
    {8, stNttRadix2<8>},
    {16, stNttRadix2<16>},
    {32, stNttRadix2<32>},
    {64, stNttRadix2Adaptive<64>},
    {128, stNttRadix2_128},
    {256, stNttRadix2Adaptive<256>},
    {512, stNttRadix4Adaptive<512>},
    {1024, stNttRadix2Adaptive<1024>},
    {2048, stNttRadix2Adaptive<2048>},
};

float stNtt(std::span<int> vec, int size, int root, int mod, int batches, Radix radix) {
    float gpuTime;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    cuda::Buffer vecGPU(vec);

    int *host_twiddles;
    CCErr(cudaMallocHost(&host_twiddles, size * sizeof(int)));
    int temp = 1;
    for (int i = 0; i < size / 2; ++i) {
        host_twiddles[i] = temp;
        temp = (temp * root) % mod;
    }

    CCErr(cudaMemcpyToSymbol(const_twiddles, host_twiddles, size / 2 * sizeof(int)));

    // Higher blockSize, bigger bottleneck on register usage per SM
    constexpr int blockSize = 1024;
    int n = size;
    int n2 = n >> 1;
    int n4 = n >> 2;
    int n8 = n >> 3;
    int b2 = batches >> 1;
    int lN = std::bit_width(static_cast<uint>(n)) - 1;
    dim3 dimBlock;
    dim3 dimGrid;
    nttkernel kernel;

    // radix2
    kernel = radix2.at(n);
    dimBlock = dim3(n2, std::min(blockSize / n2, batches));
    dimGrid = dim3((n2 * batches + blockSize - 1) / blockSize);
    // int sharedMem = lN > 6 ? dimBlock.x * dimBlock.y * sizeof(int) : 0; // radix 2
    int sharedMem = lN > 7 ? dimBlock.x * dimBlock.y * sizeof(int) * 2 : 0;  // radix 4

    cudaProfilerStart();
    cudaEventRecord(start);
    kernel<<<dimGrid, dimBlock, sharedMem>>>(vecGPU.data(), mod);
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