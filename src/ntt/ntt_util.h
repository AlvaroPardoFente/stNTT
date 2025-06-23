#pragma once

#include <vector>
#include <span>
#include <tuple>
#include <optional>

std::tuple<int, int> findParams(size_t size, int min_mod);
int findModulus(int veclength, int min);
int findPrimitiveRoot(int degree, int totient, int mod);
int findGenerator(int totient, int mod);
bool isPrimitiveRoot(int val, int degree, int mod);

std::vector<int> uniquePrimeFactors(int n);
bool isPrime(int n);
int sqrt(int n);
int pow(int base, int exponent, int mod);
int modulo(int x, int mod);

// Performs in-place element-wise multiplication of two sequences
// s0.size() must be less than or equal to s1.size()
// TODO: Use proper mod multiplication instead of modulo
template <typename T>
void elem_wise_mul(std::span<T> s0, std::span<T> s1, int mod) {
    size_t n = s0.size();
    for (size_t i = 0; i < n; ++i)
        s0[i] = modulo(s0[i] * s1[i], mod);
}

template <typename T>
int bit_reverse(T x, size_t bits) {
    int y = 0;
    for (size_t i = 0; i < bits; ++i) {
        y = (y << 1) | (x & 1);
        x >>= 1;
    }
    return y;
}

// Reorder a vector based on the bit reverse value of it's indices
template <typename T>
std::vector<T> &bit_reverse_vector(std::vector<T> &vec, size_t levels) {
    size_t n = vec.size();

    for (size_t i = 0; i < n; ++i) {
        size_t j = bit_reverse(i, levels);
        if (j > i)
            std::swap(vec[j], vec[i]);
    }

    return vec;
}

template <typename T>
std::vector<T> &bit_reverse_vector(std::vector<T> &vec) {
    size_t levels = std::bit_width(vec.size()) - 1;
    return bit_reverse_vector(vec, levels);
}