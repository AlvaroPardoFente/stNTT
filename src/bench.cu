#include "ntt/ntt_cpu.h"
#include "ntt/ntt_util.h"
#include "ntt/cuda/ntt_kernel.cuh"
#include "ntt/cuda/cu_ntt_util.cuh"
#include "util/io.h"
#include "util/rng.h"

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

template <size_t I>
void runNttIteration(std::span<int> vec) {
    constexpr size_t n = 1 << I;
    size_t batches = 1;

    auto [root, mod] = findParams(n, 11);
    cuda::NttArgs args(vec, n, batches, root, mod, 2);

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

    cuda::ntt(k, args);
    std::cout << "Ran ntt with N = " << I << "\n";
}

template <size_t... I>
void runAll(std::span<int> vec, std::index_sequence<I...>) {
    (..., runNttIteration<I>(vec));
}

int main() {
    util::Rng rng(util::Rng::defaultSeed);
    std::vector<int> vec = rng.get_vector(pow(2, 20));
    std::vector<int> cpuRes = vec;
    std::vector<int> gpuRes = vec;
    double gpuTime;

    runAll(gpuRes, make_index_sequence_from<12, 9>());
}