#include "ntt/ntt_cpu.h"
#include "ntt/ntt_util.h"
#include "ntt/cuda/ntt_kernel.cuh"
#include "ntt/cuda/cu_ntt_util.cuh"
#include "util/io.h"
#include "util/rng.h"

#include <iostream>
#include <vector>
#include <random>

int minMod = 11;
size_t vecSize = 32;
size_t batches = 1;

int main() {
    // auto kernels = cuda::KernelMap();
    // cuda::registerAllKernels(kernels, std::span(cuda::defaultBatchesNums));

    const char *vecSizeEnv = std::getenv("VEC_SIZE");
    const char *batchesEnv = std::getenv("NUM_BATCHES");

    if (vecSizeEnv != nullptr)
        vecSize = std::stoul(vecSizeEnv);
    if (batchesEnv != nullptr)
        batches = std::stoi(batchesEnv);

    util::Rng rng(util::Rng::defaultSeed);
    std::vector<int> vec = rng.get_vector(vecSize * batches);
    // std::vector<int> vec(vecSize * batches);
    // std::iota(vec.begin(), vec.begin() + vecSize, 0);
    // for (int i = 1; i < batches; i++)
    //     std::copy(vec.begin(), vec.begin() + vecSize, vec.begin() + i * vecSize);
    std::vector<int> cpuRes = vec;
    std::vector<int> gpuRes = vec;
    double gpuTime;

    auto [root, mod] = findParams(vecSize, minMod);

    // CPU
    nttStockham(cpuRes, vecSize, root, mod, batches);

    // GPU
    try {
        cuda::NttArgs args(gpuRes, vecSize, batches, root, mod, 2);
        // auto k = [args](int *vec) {
        //     stNttRadix2Adaptive<1 << 10><<<args.dimGrid, args.dimBlock, args.sharedMem>>>(vec, args.mod);
        // };
        // auto k = [args](cuda::Buffer<int> &vec, cuda::Buffer<int> &doubleBuffer) {
        //     sttNttGlobalRadix2<1 << 27, Radix2Butterfly>(
        //         vec,
        //         doubleBuffer,
        //         args.batches,
        //         args.mod,
        //         args.dimGrid,
        //         args.dimBlock,
        //         args.sharedMem);
        // };
        auto k = cuda::chooseKernel<1 << 25>(args);
        gpuTime = cuda::ntt(k, args);
    } catch (const std::out_of_range &e) {
        std::cerr << "Error: " << e.what() << "\n";
        std::cerr << "This may be due to an unsupported vector size or root/mod combination.\n";
        return 1;
    }
    // gpuTime += 1;

    if (cpuRes != gpuRes) {
        std::cout << "ERROR: cpu result != gpu result\n";
        std::cout << "INPUT:\n" << vec << "\n";
        std::cout << "CPU:\n" << cpuRes << "\n";
        std::cout << "GPU:\n" << gpuRes << "\n";
        reportDifferences(cpuRes, gpuRes);
    } else
        std::cout << "SUCCESS\n";

    std::cout << "root: " << root << "\n";
    std::cout << "mod: " << mod << "\n";

    return 0;
}