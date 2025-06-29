#include "doctest.h"

#include "ntt/ntt_cpu.h"
#include "ntt/ntt_util.h"
#include "ntt/cuda/st_ntt.h"
#include "util/io.h"
#include "util/rng.h"

#include <vector>
#include <array>
#include <iostream>

TEST_CASE("stNttRadix2Adaptive for N[2:2048]") {
    int minMod = 11;
    std::array<size_t, 11> vecSizes = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048};
    std::array<size_t, 6> batchesNums = {1, 2, 5, 10, 100, 1000};
    util::Rng rng(util::Rng::defaultSeed);

    std::vector<int> vec = rng.get_vector(vecSizes.back() * batchesNums.back());

    for (auto vecSize : vecSizes) {
        auto [root, mod] = findParams(vecSize, minMod);
        for (auto batches : batchesNums) {
            std::vector<int> cpuRes(vec.begin(), vec.begin() + vecSize * batches);
            std::vector<int> gpuRes(vec.begin(), vec.begin() + vecSize * batches);

            nttStockham(cpuRes.data(), vecSize, root, mod, batches);
            stNtt(gpuRes, vecSize, root, mod, batches, Radix::Radix2);

            CHECK(cpuRes == gpuRes);
        }
    }
}
