#include "doctest.h"

#include "ntt/cuda/ntt_impl.h"
#include "ntt/cuda/ntt_kernel.h"
#include "ntt/cuda/cu_ntt_util.cuh"
#include "ntt/ntt_cpu.h"
#include "ntt/ntt_util.h"
#include "util/io.h"
#include "util/rng.h"

#include "ntt/cuda/implementations/st_ntt_global_radix2_4096.cuh"

#include <vector>
#include <array>
#include <iostream>
#include <algorithm>

TEST_CASE("stNttRadix2Adaptive for N[2:2048]") {
    auto kernels = cuda::KernelMap();
    cuda::registerAllKernels(kernels, std::span(cuda::defaultBatchesNums));

    int minMod = 11;
    uint maxVecSize = 2048;

    util::Rng rng(util::Rng::defaultSeed);

    std::vector<int> vec = rng.get_vector(maxVecSize * cuda::defaultBatchesNums.back());

    for (auto [key, kernel] : kernels) {
        auto [n, batches, id] = key;
        auto [root, mod] = findParams(n, minMod);
        std::vector<int> cpuRes(vec.begin(), vec.begin() + n * batches);
        std::vector<int> gpuRes(vec.begin(), vec.begin() + n * batches);

        nttStockham(cpuRes.data(), n, root, mod, batches);
        cuda::stNtt(kernels, gpuRes, n, batches, id, root, mod);

        CHECK_MESSAGE(cpuRes == gpuRes, std::string("n=" + std::to_string(n) + ", batches=" + std::to_string(batches)));
    }
}

TEST_CASE("radix2_global_4096") {
    constexpr uint n = 4096;
    constexpr int minMod = 11;
    auto [root, mod] = findParams(n, minMod);

    util::Rng rng(util::Rng::defaultSeed);
    std::vector<int> vec = rng.get_vector(n * 1);

    for (const auto batches : {1}) {
        std::vector<int> cpuRes(vec.begin(), vec.begin() + n * batches);
        std::vector<int> gpuRes(vec.begin(), vec.begin() + n * batches);
        auto [dimGrid, dimBlock, sharedMem] = cuda::getNttKernelArgs(n, 2, batches);

        nttStockham(cpuRes.data(), n, root, mod, batches);
        auto a = [mod, dimGrid, dimBlock, sharedMem](int *vec) {
            sttNttRadix2_4096(vec, mod, dimGrid, dimBlock, sharedMem);
        };
        cuda::stNtt(a, gpuRes, n, batches, root, mod);
        CHECK_MESSAGE(cpuRes == gpuRes, std::string("n=" + std::to_string(n) + ", batches=" + std::to_string(batches)));
    }
}

// TEST_CASE("All implementations") {
//     util::Rng rng(util::Rng::defaultSeed);
//     std::vector<uint> batchesNums = {1, 2, 5, 10, 100, 1000};
//     std::vector<int> vec = rng.get_vector(2048 * *std::max_element(batchesNums.begin(), batchesNums.end()));

//     for (auto&)
// }
