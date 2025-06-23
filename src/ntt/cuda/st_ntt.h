#pragma once

#include <span>

enum class Radix {
    Radix2,    // Explicitly radix 2 for n <= 64, radix2_128 for n == 128
    Radix4,    // Explicitly radix 4
    Radix8,    // Adaptive radix with case 8
    Minradix,  // Adaptive radix with minimum possible radix to keep one NTT per warp
};

float stNtt(std::span<int> vec, int size, int root, int mod, int batches, Radix radix);