#pragma once

#include <vector>
#include <span>
#include <tuple>
#include <optional>
#include <ranges>
#include <bit>
#include <algorithm>

std::tuple<size_t, size_t> findParams(size_t size, size_t min_mod);
size_t findModulus(size_t veclength, size_t min);
size_t findPrimitiveRoot(size_t degree, size_t totient, size_t mod);
size_t findGenerator(size_t totient, size_t mod);
bool isPrimitiveRoot(size_t val, size_t degree, size_t mod);

std::vector<size_t> uniquePrimeFactors(size_t n);
template <typename T>
bool isPrime(T n);
size_t sqrt(size_t n);
template <std::integral I>
I pow(I base, I exponent, I mod);
int modulo(int x, int mod);

// Performs in-place element-wise multiplication of two sequences
// s0.size() must be less than or equal to s1.size()
// TODO: Use proper mod multiplication instead of modulo
template <typename T>
void elemWiseMul(std::span<T> s0, std::span<T> s1, size_t mod) {
    size_t n = s0.size();
    for (size_t i = 0; i < n; ++i)
        s0[i] = modulo(s0[i] * s1[i], mod);
}

template <typename T>
size_t bitReverse(T x, size_t bits) {
    size_t y = 0;
    for (size_t i = 0; i < bits; ++i) {
        y = (y << 1) | (x & 1);
        x >>= 1;
    }
    return y;
}

// Reorder a vector based on the bit reverse value of it's indices
template <typename T>
std::vector<T> &bitReverseVector(std::vector<T> &vec, size_t levels) {
    size_t n = vec.size();

    for (size_t i = 0; i < n; ++i) {
        size_t j = bitReverse(i, levels);
        if (j > i)
            std::swap(vec[j], vec[i]);
    }

    return vec;
}

template <typename T>
std::vector<T> &bitReverseVector(std::vector<T> &vec) {
    size_t levels = std::bit_width(vec.size()) - 1;
    return bitReverseVector(vec, levels);
}

template <typename T>
inline bool isPrime(T n) {
    auto range = std::views::iota(2UL, sqrt(n) + 1UL);
    return std::ranges::all_of(range, [&n](T x) { return n % x != 0; });
}

// Basic int modular power
template <std::integral I>
I pow(I base, I exponent, I mod) {
    if (exponent < 0) {
        // Compute modular multiplicative inverse of base modulo mod
        I inverse = 1, b = base, m = mod - 2;  // Using Fermat's Little Theorem: base^(mod-2) = base^(-1) (mod mod)
        while (m > 0) {
            if (m % 2 == 1)
                inverse = (inverse * b) % mod;
            b = (b * b) % mod;
            m /= 2;
        }
        base = inverse;
        exponent = -exponent;
    }

    I result = 1;
    base %= mod;
    while (exponent > 0) {
        if (exponent % 2 == 1)
            result = (result * base) % mod;
        base = (base * base) % mod;
        exponent /= 2;
    }
    return result;
}