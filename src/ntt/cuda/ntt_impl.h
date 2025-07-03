#pragma once

namespace ntt {
// All cuda ntt implementation variants
enum class KernelId : unsigned {
    stNttRadix2 = 1,
    stNttRadix2Adaptive = 2,
    stNttRadix2_128 = 3,
    stNttRadix2_512 = 4,
    stNttRadix4 = 5,
    stNttRadixAdaptive = 6,
};
}  // namespace ntt