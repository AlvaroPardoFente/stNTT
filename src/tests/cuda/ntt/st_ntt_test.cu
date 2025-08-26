#include "doctest.h"

#include "cuda/ntt/ntt_impl.h"
#include "cuda/ntt/ntt_kernel.cuh"
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
void runNttIteration(std::span<int> vec, size_t batches, uint maxBlockSize) {
    constexpr size_t n = 1UL << I;

    auto [root, mod] = findParams(n, 11);

    auto refSpan = vec.subspan(0, n * batches);

    std::vector<int> cpuVec{refSpan.begin(), refSpan.end()};
    nttStockham(cpuVec, n, root, mod, batches);

    std::vector<int> gpuVec{refSpan.begin(), refSpan.end()};
    cuda::ntt::NttArgs args(gpuVec, n, batches, root, mod, 2, maxBlockSize);

    // auto k = cuda::chooseKernel<n>(args);
    // auto k = [args](int *vec) {
    //     stNttRadix2Adaptive<n><<<args.dimGrid, args.dimBlock, args.sharedMem>>>(vec, args.mod);
    // };

    // cuda::ntt(k, args);
    cuda::ntt::autoNtt<n>(args);
    CHECK_MESSAGE(
        cpuVec == gpuVec,
        "N: 2^",
        std::to_string(I),
        " | NUM_BATCHES: ",
        batches,
        " | BLOCK_SIZE: ",
        args.blockSize);
}
template <size_t... I>
void runAll(std::span<int> vec, size_t batches, uint maxBlockSize, std::index_sequence<I...>) {
    ([&]() { runNttIteration<I>(vec, batches, maxBlockSize); }(), ...);
}

TEST_CASE("stNttRadix2 for N[2^1:2^6]") {
    int minMod = 11;
    auto nRange = make_index_range<2, 7>();
    util::Rng rng(util::Rng::defaultSeed);
    std::vector<int> vec = rng.get_vector((1 << 6) * cuda::ntt::defaultBatchesNums.back());

    for (const auto batch : cuda::ntt::defaultBatchesNums) {
        for (const auto maxBlockSize : cuda::ntt::defaultMaxBlockSizes)
            runAll(vec, batch, maxBlockSize, nRange);
    }
}

TEST_CASE("stNttRadix2Adaptive for N[2^7:2^11]") {
    int minMod = 11;
    auto nRange = make_index_range<7, 12>();
    util::Rng rng(util::Rng::defaultSeed);
    std::vector<int> vec = rng.get_vector((1 << 11) * cuda::ntt::defaultBatchesNums.back());

    for (const auto batch : cuda::ntt::defaultBatchesNums) {
        for (const auto maxBlockSize : cuda::ntt::defaultMaxBlockSizes)
            runAll(vec, batch, maxBlockSize, nRange);
    }
}

TEST_CASE("stNttGlobalRadix2 for N[2^12:2^20]") {
    int minMod = 11;
    auto nRange = make_index_range<12, 21>();
    util::Rng rng(util::Rng::defaultSeed);
    std::vector<int> vec = rng.get_vector((1 << 20) * cuda::ntt::defaultBatchesNums.back());

    for (const auto batch : cuda::ntt::defaultBatchesNums) {
        for (const auto maxBlockSize : cuda::ntt::defaultMaxBlockSizes)
            runAll(vec, batch, maxBlockSize, nRange);
    }
}

TEST_CASE("stNttGlobalRadix2 for N[2^21:2^27]" * doctest::skip()) {
    int minMod = 11;
    auto nRange = make_index_range<21, 27>();
    util::Rng rng(util::Rng::defaultSeed);
    std::vector<int> vec = rng.get_vector((1 << 26) * cuda::ntt::defaultBatchesNums.back());

    // This test is too slow to add both batches and block sizes
    for (const auto batch : cuda::ntt::defaultBatchesNums) {
        runAll(vec, batch, 1024, nRange);
    }
}

TEST_CASE("findParams works for all tested Ns" * doctest::skip()) {
    size_t minMod = 11;
    size_t root = 0, mod = 0;

    for (size_t n = 1; n < 1UL << 31; n *= 2) {
        auto [newRoot, newMod] = findParams(n, minMod);

        CHECK(newMod >= mod);
        // std::cout << n << ": " << newMod << ", " << newRoot << "\n";
        mod = newMod;
        root = newRoot;
    }
}