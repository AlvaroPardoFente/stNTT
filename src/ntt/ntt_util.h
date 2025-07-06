#pragma once

#include <vector>
#include <span>
#include <tuple>
#include <optional>

std::tuple<int, int> findParams(size_t size, int min_mod);
int findModulus(int veclength, int min);
int findPrimitiveRoot(int degree, int totient, int mod);
int findGenerator(int totient, int mod);
bool isPrimitiveRoot(size_t val, size_t degree, size_t mod);

std::vector<int> uniquePrimeFactors(int n);
bool isPrime(int n);
int sqrt(int n);
template <std::integral I>
I pow(I base, I exponent, I mod);
int modulo(int x, int mod);

// Performs in-place element-wise multiplication of two sequences
// s0.size() must be less than or equal to s1.size()
// TODO: Use proper mod multiplication instead of modulo
template <typename T>
void elemWiseMul(std::span<T> s0, std::span<T> s1, int mod) {
    size_t n = s0.size();
    for (size_t i = 0; i < n; ++i)
        s0[i] = modulo(s0[i] * s1[i], mod);
}

template <typename T>
int bitReverse(T x, size_t bits) {
    int y = 0;
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