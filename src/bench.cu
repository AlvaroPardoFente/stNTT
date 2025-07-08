#include "ntt/ntt_cpu.h"
#include "ntt/ntt_util.h"
#include "ntt/cuda/ntt_kernel.cuh"
#include "ntt/cuda/cu_ntt_util.cuh"
#include "util/io.h"
#include "util/rng.h"
#include "util/timer.h"

template <std::size_t Offset, typename Seq>
struct offset_index_sequence_impl;

template <std::size_t Offset, std::size_t... Is>
struct offset_index_sequence_impl<Offset, std::index_sequence<Is...>> {
    using type = std::index_sequence<(Is + Offset)...>;
};

template <std::size_t Offset, std::size_t N>
using make_index_sequence_from = typename offset_index_sequence_impl<Offset, std::make_index_sequence<N>>::type;

template <size_t... I>
constexpr auto generateGlobalRadix2Array(std::index_sequence<I...>) {
    return std::array{sttNttGlobalRadix2<(1u << I)>...};
}

constexpr auto globalRadix2Funcs = generateGlobalRadix2Array(std::make_integer_sequence<size_t, 10>());

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
};

std::ostream& operator<<(std::ostream& os, const BenchStats& p) {
    os << "N: " << p.args.n << " (2^" << log2_uint(p.args.n) << ")\n";
    os << "GPU time:\t\t" << p.gpuTime << "\n";
    os << "findParams time:\t" << p.findParamsTime << "\n";
    os << "initArgs time:\t\t" << p.initArgsTime << "\n";
    os << "Full kernel time:\t" << p.kernelWithInitTime << "\n";
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

template <size_t I>
BenchStats runNttIteration(std::span<int> vec) {
    Timer totalTimer;
    BenchStats stats;

    constexpr size_t n = 1UL << I;
    size_t batches = 1;

    Timer timer;
    auto [root, mod] = findParams(n, 11);
    stats.findParamsTime = timer.stop();

    timer.start();
    cuda::NttArgs args(vec, n, batches, root, mod, 2);
    stats.args = args;

    auto k = [args](cuda::Buffer<int>& vec, cuda::Buffer<int>& doubleBuffer) {
        sttNttGlobalRadix2<n, EmptyButterfly>(
            vec,
            doubleBuffer,
            args.batches,
            args.mod,
            args.dimGrid,
            args.dimBlock,
            args.sharedMem);
    };
    stats.initArgsTime = timer.stop();

    timer.start();
    double gpuTime = cuda::ntt(k, args);
    stats.kernelWithInitTime = timer.stop();
    stats.gpuTime = Timer::Duration(gpuTime);

    stats.totalTime = totalTimer.stop();

    return stats;
}

template <size_t... I>
auto runAll(std::span<int> vec, std::index_sequence<I...>) {
    return std::array<BenchStats, sizeof...(I)>{runNttIteration<I>(vec)...};
}

int main() {
    util::Rng rng(util::Rng::defaultSeed);
    std::vector<int> vec = rng.get_vector(pow(2, 20));
    std::vector<int> cpuRes = vec;
    std::vector<int> gpuRes = vec;
    double gpuTime;

    auto stats = runAll(gpuRes, make_index_sequence_from<12, 15>());

    for (const auto& stat : stats) {
        std::cout << stat << "\n";
    }
}