#pragma once

#include <vector>
#include <optional>

int find_modulus(int veclength, int min);
int find_primitive_root(int degree, int totient, int mod);
int find_generator(int totient, int mod);
bool is_primitive_root(int val, int degree, int mod);

std::vector<int> unique_prime_factors(int n);
bool is_prime(int n);
int sqrt(int n);
int pow(int base, int exponent, int mod);
int modulo(int x, int mod);

template <typename T>
int bit_reverse(T x, size_t bits);
template <typename T>
std::vector<T> &bit_reverse_vector(std::vector<T> &vec, std::optional<size_t> levels = std::nullopt);