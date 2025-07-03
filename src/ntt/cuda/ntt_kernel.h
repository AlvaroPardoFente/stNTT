#pragma once

#include "ntt/cuda/ntt_impl.h"

#include <optional>
#include <functional>
#include <map>
#include <span>

namespace cuda {
constexpr std::array<uint, 6> defaultBatchesNums = {1, 2, 5, 10, 100, 1000};

struct KernelKey {
    uint n;
    uint batches;
    ntt::KernelId impl_id;

    auto operator<=>(const KernelKey &) const = default;
};

using KernelLauncher = std::function<void(int *, int)>;
using KernelMap = std::map<KernelKey, KernelLauncher>;

void registerAllKernels(KernelMap &kernels, std::span<const uint> batchesNums);

float stNtt(KernelMap &kernels, std::span<int> vec, uint n, uint batches, ntt::KernelId id, int root, int mod);
float stNtt(std::function<void(int *)> kernel, std::span<int> vec, uint n, uint batches, int root, int mod);
}  // namespace cuda