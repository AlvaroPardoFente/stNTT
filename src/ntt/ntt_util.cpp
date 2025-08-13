#include "ntt/ntt_util.h"

#include <ranges>
#include <algorithm>
#include <stdexcept>
#include <concepts>
#include <limits>
#include <iostream>

std::tuple<size_t, size_t> findParams(size_t size, size_t min_mod) {
    size_t mod = findModulus(size, min_mod);
    size_t root = findPrimitiveRoot(size, mod - 1, mod);
    return std::make_tuple(root, mod);
}

size_t findModulus(size_t veclen, size_t min) {
    // ???
    size_t start = (min - 1UL + veclen - 1UL) / veclen;
    for (size_t k = std::max(start, 1UL);; ++k) {
        if (k > std::numeric_limits<size_t>::max() / veclen)
            throw std::overflow_error("Search range exhausted (overflow)");
        size_t n = k * veclen + 1UL;
        if (isPrime(n))
            return n;
    }
    throw std::runtime_error("Could not find prime N (unreachable)");
}

size_t findPrimitiveRoot(size_t degree, size_t totient, size_t mod) {
    size_t gen = findGenerator(totient, mod);
    size_t root = pow(gen, totient / degree, mod);

    return root;
}

size_t findGenerator(size_t totient, size_t mod) {
    auto range = std::views::iota(1UL, mod);
    auto res = std::ranges::find_if(range, [totient, mod](size_t i) { return isPrimitiveRoot(i, totient, mod); });
    if (res == range.end())
        throw std::runtime_error("No generator exists");
    return *res;
}

bool isPrimitiveRoot(size_t val, size_t degree, size_t mod) {
    std::vector<size_t> prime_factors = uniquePrimeFactors(degree);
    return pow(val, degree, mod) == 1UL && std::ranges::all_of(prime_factors, [val, degree, mod](size_t p) {
               return pow(val, degree / p, mod) != 1UL;
           });
}

std::vector<size_t> uniquePrimeFactors(size_t n) {
    std::vector<size_t> result;
    size_t end = sqrt(n);

    for (size_t i = 2; i <= end; i++) {
        if (n % i == 0) {
            n /= i;
            result.push_back(i);
            while (n % i == 0)
                n /= i;
            end = sqrt(n);
        }
    }

    if (n > 1)
        result.push_back(n);

    return result;
}

// Int sqrt that returns floor(sqrt(n))
size_t sqrt(size_t n) {
    size_t i = 1;
    size_t result = 0;

    while (i <= n / i)
        i *= 2;

    for (; i > 0; i /= 2)
        if ((result + i) * (result + i) <= n)
            result += i;

    return result;
}

// True modulus operation (ideally this would not be used, operations
// that include the modulo would be used instead)
int modulo(int x, int mod) {
    int result = x % mod;
    if (result >= 0)
        return result;
    else
        return result + mod;
}