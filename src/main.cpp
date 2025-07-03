#include "ntt/ntt_cpu.h"
#include "ntt/ntt_util.h"
#include "ntt/cuda/ntt_kernel.h"
#include "util/io.h"
#include "util/rng.h"

#include <iostream>
#include <vector>
#include <random>

int minMod = 11;
size_t vecSize = 32;
size_t batches = 1;

int main() {
    auto kernels = cuda::KernelMap();
    cuda::registerAllKernels(kernels, std::span(cuda::defaultBatchesNums));

    const char *vecSizeEnv = std::getenv("VEC_SIZE");
    const char *batchesEnv = std::getenv("NUM_BATCHES");

    if (vecSizeEnv != nullptr)
        vecSize = std::stoul(vecSizeEnv);
    if (batchesEnv != nullptr)
        batches = std::stoi(batchesEnv);

    util::Rng rng(util::Rng::defaultSeed);
    std::vector<int> vec = rng.get_vector(vecSize * batches);
    std::vector<int> cpuRes = vec;
    std::vector<int> gpuRes = vec;
    double gpuTime;

    auto [root, mod] = findParams(vecSize, minMod);

    // CPU
    nttStockham(cpuRes.data(), vecSize, root, mod, batches);

    // GPU
    try {
        gpuTime = cuda::stNtt(kernels, gpuRes, vecSize, batches, ntt::KernelId::stNttRadix2, root, mod);
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
    } else
        std::cout << "SUCCESS\n";

    std::cout << "root: " << root << "\n";
    std::cout << "mod: " << mod << "\n";

    return 0;
}