#include "ntt/ntt_cpu.h"
#include "ntt/ntt_util.h"
#include "ntt/cuda/st_ntt.h"
#include "util/io.h"

#include <iostream>
#include <vector>
#include <random>

const int seed = 42;
std::mt19937 rng(seed);
std::uniform_int_distribution<int> dist(0, 9);

int minMod = 11;
size_t vecSize = 32;
size_t batches = 1;

int main() {
    std::vector<int> vec;
    std::vector<int> cpuRes;
    std::vector<int> gpuRes;
    double gpuTime;

    const char *vecSizeEnv = std::getenv("VEC_SIZE");
    const char *batchesEnv = std::getenv("NUM_BATCHES");

    if (vecSizeEnv != nullptr)
        vecSize = std::stoul(vecSizeEnv);
    if (batchesEnv != nullptr)
        batches = std::stoi(batchesEnv);

    // Default vector init
    for (size_t i = 0; i < vecSize * batches; i++)
        vec.push_back(dist(rng));
    cpuRes = vec;
    gpuRes = vec;

    auto [root, mod] = findParams(vecSize, minMod);

    // CPU
    nttStockham(cpuRes.data(), vecSize, root, mod, batches);

    // GPU
    gpuTime = stNtt(gpuRes, vecSize, root, mod, batches, Radix::Radix2);

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