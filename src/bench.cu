#include "ntt/ntt_cpu.h"
#include "ntt/ntt_util.h"
#include "ntt/cuda/ntt_kernel.cuh"
#include "ntt/cuda/cu_ntt_util.cuh"
#include "util/io.h"
#include "util/rng.h"
#include "util/timer.h"
#include "util/index_sequences.h"

struct BenchStats {
    cuda::NttArgs args;
    Timer::Duration gpuTime;
    // Time it takes to find root and mod
    Timer::Duration findParamsTime;
    // Time it takes to create the args struct and the lambda
    Timer::Duration initArgsTime;
    // Time it takes to init GPU resources, run the kernel and clean up
    Timer::Duration kernelWithInitTime;
    Timer::Duration totalTime;

    friend std::ostream& operator<<(std::ostream& os, const BenchStats& p);

    static std::ostream& csvHeader(std::ostream& os);
    friend std::ostream& toCsv(std::ostream& os, const BenchStats& p);
};

template <size_t I>
BenchStats runNttIteration(std::span<int> vec, uint maxBlockSize) {
    Timer totalTimer;
    BenchStats stats;

    constexpr size_t n = 1UL << I;
    size_t batches = 1;

    Timer timer;
    auto [root, mod] = findParams(n, 11);
    stats.findParamsTime = timer.stop();

    timer.start();
    cuda::NttArgs args(vec.subspan(0, n * batches), n, batches, root, mod, 2, maxBlockSize);
    stats.args = args;

    // auto k = cuda::chooseKernel<n>(args);

    stats.initArgsTime = timer.stop();

    timer.start();
    // double gpuTime = cuda::ntt(k, args);
    double gpuTime = cuda::autoNtt<n>(args);
    stats.kernelWithInitTime = timer.stop();
    stats.gpuTime = Timer::Duration(gpuTime);

    stats.totalTime = totalTimer.stop();

    return stats;
}

template <size_t... I>
auto runAll(std::span<int> vec, size_t reps, uint maxBlockSize, std::index_sequence<I...>) {
    std::vector<BenchStats> result{};
    result.reserve(sizeof...(I) * reps);
    // Warmup
    (
        [&]() {
            for (size_t r = 0; r < 10; r++) {
                runNttIteration<I>(vec, maxBlockSize);
            }
        }(),
        ...);

    size_t idx = 0;
    (
        [&]() {
            for (size_t r = 0; r < reps; r++) {
                result.push_back(runNttIteration<I>(vec, maxBlockSize));
            }
        }(),
        ...);
    return result;
}

int main() {
    util::Rng rng(util::Rng::defaultSeed);
    std::vector<int> vec = rng.get_vector(pow(2, 27));
    std::vector<int> cpuRes = vec;
    std::vector<int> gpuRes = vec;
    double gpuTime;

    auto nRange = make_index_range<2, 27>();

    std::vector<BenchStats> stats;
    for (const auto maxBlockSize : cuda::defaultMaxBlockSizes) {
        auto blockStats = runAll(gpuRes, 20, maxBlockSize, nRange);
        stats.insert(stats.end(), blockStats.begin(), blockStats.end());
    }

    bool printCsv = true;
    if (printCsv) {
        BenchStats::csvHeader(std::cout);
        for (const auto& stat : stats)
            toCsv(std::cout, stat);
    } else
        for (const auto& stat : stats) {
            std::cout << stat << "\n";
        }
}

std::ostream& operator<<(std::ostream& os, const BenchStats& p) {
    os << "N: " << p.args.n << " (2^" << log2_uint(p.args.n) << ")\n";
    os << "Block size:\t\t" << p.args.dimBlock.x * p.args.dimBlock.y * p.args.dimBlock.z << "\n";
    os << "Kernel time:\t\t" << p.gpuTime << "\n";
    os << "findParams time:\t" << p.findParamsTime << "\n";
    os << "initArgs time:\t\t" << p.initArgsTime << "\n";
    os << "Kernel+transfer time:\t" << p.kernelWithInitTime << "\n";
    os << "Total time:\t\t" << p.totalTime << "\n";
    // os << "Vector address: " << p.args.vec.data() << "\n";
    // os << "Vector size(bytes): " << p.args.vec.size() << "(" << p.args.vec.size_bytes() << ")\n";
    // os << "Batches: " << p.args.batches << "\n";
    // os << "Root: " << p.args.root << "\n";
    // os << "Mod: " << p.args.mod << "\n";
    // os << "Radix: " << p.args.radix << "\n";
    // os << "Block size: " << p.args.blockSize << "\n";
    // os << "dimGrid: (" << p.args.dimGrid.x << ", " << p.args.dimGrid.y << ", " << p.args.dimGrid.z << ")\n";
    // os << "dimBlock: (" << p.args.dimBlock.x << ", " << p.args.dimBlock.y << ", " << p.args.dimBlock.z << ")\n";
    // os << "Shared memory: " << p.args.sharedMem << "\n";
    return os;
}

static constexpr char sep = ',';
std::ostream& BenchStats::csvHeader(std::ostream& os) {
    // clang-format off
    return os << "N"
        << sep << "Block size"
        << sep << "Kernel time"
        << sep << "findParams time"
        << sep << "initArgs time"
        << sep << "Kernel+transfers time"
        << sep << "Total time"
        << "\n";
    // clang-format on
}
// std::ostream& toCsv(std::ostream& os, const BenchStats& p) {
//     // clang-format off
//     return os << p.args.n
//         << sep << p.args.dimBlock.x * p.args.dimBlock.y * p.args.dimBlock.z
//         << sep << p.gpuTime
//         << sep << p.findParamsTime
//         << sep << p.initArgsTime
//         << sep << p.kernelWithInitTime
//         << sep << p.totalTime
//         << "\n";
//     // clang-format on
// }

std::ostream& toCsv(std::ostream& os, const BenchStats& p) {
    // clang-format off
    return os << p.args.n
        << sep << p.args.dimBlock.x * p.args.dimBlock.y * p.args.dimBlock.z
        << sep << p.gpuTime.count()
        << sep << p.findParamsTime.count()
        << sep << p.initArgsTime.count()
        << sep << p.kernelWithInitTime.count()
        << sep << p.totalTime.count()
        << "\n";
    // clang-format on
}
