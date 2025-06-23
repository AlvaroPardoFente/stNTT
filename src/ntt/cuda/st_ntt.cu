#include "st_ntt.h"

#include "ntt/cuda/implementations/st_ntt_radix2.cuh"

#include <cuda_profiler_api.h>

#include <map>

using nttkernel = void (*)(int *vec, int mod);

std::map<uint, nttkernel> radix2{
    {2, stNttRadix2<2>},
    {4, stNttRadix2<4>},
    {8, stNttRadix2<8>},
    {16, stNttRadix2<16>},
    {32, stNttRadix2<32>},
    {64, stNttRadix2<64>},
    // {128, stNttRadix2<128, 7>},
};

float stNtt(std::span<int> vec, int size, int root, int mod, int batches, Radix radix) {
    int memsize = vec.size() * sizeof(int);
    int *vecGPU;

    float gpuTime;
    cudaEvent_t start, end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);

    CCErr(cudaMalloc(&vecGPU, memsize));
    CCErr(cudaMemcpy(vecGPU, vec.data(), memsize, cudaMemcpyHostToDevice));

    int *host_twiddles;
    CCErr(cudaMallocHost(&host_twiddles, size * sizeof(int)));
    int temp = 1;
    for (size_t i = 0; i < size / 2; ++i) {
        host_twiddles[i] = temp;
        temp = (temp * root) % mod;
    }

    CCErr(cudaMemcpyToSymbol(const_twiddles, host_twiddles, size / 2 * sizeof(int)));

    // Higher blockSize, bigger bottleneck on register usage per SM
    constexpr int blockSize = 512;
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

    cudaProfilerStart();
    cudaEventRecord(start);
    kernel<<<dimGrid, dimBlock>>>(vecGPU, mod);
    CCErr(cudaGetLastError());
    cudaEventRecord(end);

    cudaProfilerStop();
    cudaEventSynchronize(end);
    cudaEventElapsedTime(&gpuTime, start, end);

    CCErr(cudaDeviceSynchronize());

    CCErr(cudaMemcpy(vec.data(), vecGPU, memsize, cudaMemcpyDeviceToHost));
    CCErr(cudaFree(vecGPU));
    CCErr(cudaFreeHost(host_twiddles));

    return gpuTime;
}