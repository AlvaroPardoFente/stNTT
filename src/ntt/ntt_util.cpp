#include "ntt/ntt_util.h"

#include <ranges>
#include <algorithm>
#include <stdexcept>
#include <concepts>

std::tuple<int, int> findParams(size_t size, int min_mod) {
    int mod = findModulus(size, min_mod);
    int root = findPrimitiveRoot(size, mod - 1, mod);
    return std::make_tuple(root, mod);
}

int findModulus(int veclen, int min) {
    // ???
    int start = (min - 1 + veclen - 1) / veclen;
    for (int k = std::max(start, 1);; ++k) {
        int n = k * veclen + 1;
        if (isPrime(n))
            return n;
    }
    throw std::runtime_error("Could not find prime N (unreachable)");
}

int findPrimitiveRoot(int degree, int totient, int mod) {
    int gen = findGenerator(totient, mod);
    int root = pow(gen, totient / degree, mod);

    return root;
}

int findGenerator(int totient, int mod) {
    auto range = std::views::iota(1, mod);
    auto res = std::ranges::find_if(range, [totient, mod](int i) { return isPrimitiveRoot(i, totient, mod); });
    if (res == range.end())
        throw std::runtime_error("No generator exists");
    return *res;
}

bool isPrimitiveRoot(size_t val, size_t degree, size_t mod) {
    std::vector<int> prime_factors = uniquePrimeFactors(degree);
    return pow(val, degree, mod) == 1 &&
           std::ranges::all_of(prime_factors, [val, degree, mod](int p) { return pow(val, degree / p, mod) != 1; });
}

std::vector<int> uniquePrimeFactors(int n) {
    std::vector<int> result;
    int end = sqrt(n);

    for (int i = 2; i <= end; i++) {
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

bool isPrime(int n) {
    auto range = std::views::iota(2, sqrt(n) + 1);
    return std::ranges::all_of(range, [&n](int x) { return n % x != 0; });
}

// Int sqrt that returns floor(sqrt(n))
int sqrt(int n) {
    int i = 1;
    int result = 0;

    while (i * i < n)
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