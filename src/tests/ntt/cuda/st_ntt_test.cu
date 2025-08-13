#include "doctest.h"

#include "ntt/cuda/ntt_impl.h"
#include "ntt/cuda/ntt_kernel.cuh"
#include "ntt/cuda/cu_ntt_util.cuh"
#include "ntt/ntt_cpu.h"
#include "ntt/ntt_util.h"
#include "util/io.h"
#include "util/rng.h"
#include "util/index_sequences.h"

#include <vector>
#include <array>
#include <iostream>
#include <algorithm>

template <size_t I>
void runNttIteration(std::span<int> vec, size_t batches) {
    constexpr size_t n = 1UL << I;

    auto [root, mod] = findParams(n, 11);

    std::vector<int> cpuVec{vec.subspan<0, n>().begin(), vec.subspan<0, n>().end()};
    nttStockham(cpuVec, I, root, mod, batches);

    std::vector<int> gpuVec{vec.subspan<0, n>().begin(), vec.subspan<0, n>().end()};
    cuda::NttArgs args(gpuVec, n, batches, root, mod, 2);

    // auto k = cuda::chooseKernel<n>(args);
    auto k = [args](int *vec) {
        stNttRadix2Adaptive<n><<<args.dimGrid, args.dimBlock, args.sharedMem>>>(vec, args.mod);
    };

    cuda::ntt(k, args);

    CHECK(cpuVec == gpuVec);
}
template <size_t... I>
void runAll(std::span<int> vec, size_t batches, std::index_sequence<I...>) {
    ([&]() { runNttIteration<I>(vec, batches); }(), ...);
}

TEST_CASE("stNttRadix2Adaptive for N[2:2048]") {
    int minMod = 11;
    auto nRange = make_index_range<2, 12>();
    util::Rng rng(util::Rng::defaultSeed);
    std::vector<int> vec = rng.get_vector((1 << 11) * cuda::defaultBatchesNums.back());

    for (const auto batch : cuda::defaultBatchesNums) {
        runAll(vec, batch, nRange);
    }
}

// TEST_CASE("radix2_global_4096") {
//     constexpr uint n = 4096;
//     constexpr int minMod = 11;
//     auto [root, mod] = findParams(n, minMod);

//     util::Rng rng(util::Rng::defaultSeed);
//     std::vector<int> vec = rng.get_vector(n * cuda::defaultBatchesNums.back());

//     for (const auto batches : cuda::defaultBatchesNums) {
//         std::vector<int> cpuRes(vec.begin(), vec.begin() + n * batches);
//         std::vector<int> gpuRes(vec.begin(), vec.begin() + n * batches);
//         auto [dimGrid, dimBlock, sharedMem] = cuda::getNttKernelArgs(n, 2, batches);

//         nttStockham(cpuRes.data(), n, root, mod, batches);
//         auto a = [mod, dimGrid, dimBlock, sharedMem](int *vec) {
//             sttNttRadix2_4096(vec, mod, dimGrid, dimBlock, sharedMem);
//         };
//         cuda::stNtt(a, gpuRes, n, batches, root, mod);
//         CHECK_MESSAGE(cpuRes == gpuRes, std::string("n=" + std::to_string(n) + ", batches=" +
//         std::to_string(batches)));
//     }
// }

// TEST_CASE("All implementations") {
//     util::Rng rng(util::Rng::defaultSeed);
//     std::vector<uint> batchesNums = {1, 2, 5, 10, 100, 1000};
//     std::vector<int> vec = rng.get_vector(2048 * *std::max_element(batchesNums.begin(), batchesNums.end()));

//     for (auto&)
// }

TEST_CASE("findParams works for all tested Ns") {
    size_t minMod = 11;
    size_t root = 0, mod = 0;

    for (size_t n = 1; n < 1UL << 31; n *= 2) {
        auto [newRoot, newMod] = findParams(n, minMod);

        CHECK(newMod >= mod);
        std::cout << n << ": " << newMod << ", " << newRoot << "\n";
        mod = newMod;
        root = newRoot;
    }
}