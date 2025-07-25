#include "ntt/ntt_cpu.h"

#include "ntt/ntt_util.h"

#include <stdexcept>
#include <algorithm>
#include <numeric>
#include "ntt_cpu.h"

// In-place implementation of the cooley-tukey NTT (NO->BO)
void nttCt(std::vector<int> &vec, int root, int mod) {
    size_t n = vec.size();
    int levels = std::bit_width(n) - 1;
    if (1 << levels != n)
        throw std::invalid_argument("vec.size() must be a power of 2");

    // Holds w(root)^0, w^1, w^2 ... w^(n/2)
    // It is turned to bit order because of the twiddle factor ordering
    // https://arxiv.org/pdf/2211.13546 pg. 27
    std::vector<int> pow_table;
    int temp = 1;
    for (size_t i = 0; i < n / 2; ++i) {
        pow_table.push_back(temp);
        temp = (temp * root) % mod;
    }
    bitReverseVector(pow_table);

    for (int size = n; size >= 2; size /= 2) {
        int halfsize = size / 2;
        int k = 0;
        for (int i = 0; i < n; i += size) {
            for (int j = i; j < i + halfsize; ++j) {
                int l = j + halfsize;
                int left = vec[j];
                int right = vec[l] * pow_table[k];
                vec[j] = modulo(left + right, mod);
                vec[l] = modulo(left - right, mod);
            }
            k++;
        }
    }
}

// In place implementation of the ntt using the DIF stockham algorithm (NO->NO)
// https://doi.org/10.1109/TCSII.2019.2917621 pg. 2
void nttStockham(int *vec, int size, int root, int mod, std::optional<int> nsteps) {
    size_t n = size;
    int steps = std::bit_width(n) - 1;
    if (1 << steps != n)
        throw std::invalid_argument("size must be a power of 2");

    std::vector<int> y_vector(n, 0);
    int *y = y_vector.data();
    bool swap = false;

    // Holds w(root)^0, w^1, w^2 ... w^(n/2)
    std::vector<int> pow_table;
    int temp = 1;
    for (size_t i = 0; i < n / 2; ++i) {
        pow_table.push_back(temp);
        temp = (temp * root) % mod;
    }

    if (nsteps)
        steps = *nsteps;
    for (int i = 0; i < steps; i++) {
        int J = pow(2, i, mod);
        for (int s = 0; s < n / (2 * J); s++) {
            int w = pow_table[J * s];
            for (int j = 0; j < J; ++j) {
                int left = vec[s * J + j];
                int right = vec[s * J + j + n / 2];
                y[2 * s * J + j] = modulo(left + right, mod);
                y[(2 * s + 1) * J + j] = modulo(w * modulo(left - right, mod), mod);
            }
        }
        std::swap(vec, y);
        swap = !swap;
    }

    if (swap) {
        std::swap(vec, y);
        std::copy(y, y + n, vec);
    }
}

void nttStockham(std::span<int> vec, int size, int root, int mod, int batches, std::optional<int> nsteps) {
    for (size_t i = 0; i < batches; ++i)
        nttStockham(vec.data() + i * size, size, root, mod, nsteps);
}

void nttStockhamIdx(int *vec, int size, int root, int mod, std::optional<int> nsteps) {
    size_t n = size;
    int steps = std::bit_width(n) - 1;
    if (1 << steps != n)
        throw std::invalid_argument("size must be a power of 2");

    std::vector<int> y_vector(n, 0);
    int *y = y_vector.data();
    bool swap = false;

    if (nsteps)
        steps = *nsteps;
    for (int i = 0; i < steps; i++) {
        int J = pow(2, i, mod);
        for (int s = 0; s < n / (2 * J); s++) {
            for (int j = 0; j < J; ++j) {
                int left = vec[s * J + j];
                int right = vec[s * J + j + n / 2];
                y[2 * s * J + j] = left;
                y[(2 * s + 1) * J + j] = right;
            }
        }
        std::swap(vec, y);
        swap = !swap;
    }

    if (swap) {
        std::swap(vec, y);
        std::copy(y, y + n, vec);
    }
}

void nttStockhamIdx(std::span<int> vec, int size, int root, int mod, int batches, std::optional<int> nsteps) {
    for (size_t i = 0; i < batches; ++i)
        nttStockhamIdx(vec.data() + i * size, size, root, mod, nsteps);
}

// In-place implementation of the Gentleman-Sande NTT (BO->NO)
void inttGs(std::vector<int> &vec, int root, int mod) {
    unsigned int n = vec.size();
    size_t levels = std::bit_width(n) - 1;
    if (1 << levels != n)
        throw std::invalid_argument("vec.size() must be a power of 2");

    // https://arxiv.org/pdf/2211.13546 pg. 28
    std::vector<int> pow_table;
    for (int i = 0; i < n / 2; ++i)
    // TODO: Faster implementation
    {
        pow_table.push_back(pow(root, -i, mod));
    }
    bitReverseVector(pow_table);

    for (int size = 2; size <= n; size *= 2) {
        int halfsize = size / 2;
        int tablestep = n / size;
        int k = 0;
        for (int i = 0; i < n; i += size) {
            for (int j = i; j < i + halfsize; ++j) {
                int l = j + halfsize;
                int left = vec[j];
                int right = vec[l];
                vec[j] = modulo((left + right), mod);
                vec[l] = modulo((left - right) * pow_table[k], mod);
            }
            k++;
        }
    }

    int scaler = pow(static_cast<int>(n), -1, mod);  // Could be fetched early from the pow_table
    for (int &x : vec)
        x = (x * scaler) % mod;
}

// Performs the Positive-Wrapped Convolution using NTT. vec0 and vec1 are modified and the result is stored in vec0
void pwcNtt(std::vector<int> &vec0, std::vector<int> &vec1, int root, int mod) {
    nttCt(vec0, root, mod);
    nttCt(vec1, root, mod);

    elemWiseMul(std::span{vec0}, std::span{vec1}, mod);

    inttGs(vec0, root, mod);
}

// Wrapper over pwcNtt that performs the circular convolution of two vectors, finding the appropiate mod and root
// Doesn't modify the original vectors
// The chosen modulus ensures that the result is the polynomial multiplication of the vectors (no element of the result
// will wrap around the modulus)
std::tuple<std::vector<int>, int, int> pwc_convolution(std::span<const int> s0, std::span<const int> s1) {
    // len(vec0) == len(vec1) > 0, all(vec0, vec1) > 0
    int len = s0.size();
    int max_val = std::max(*std::ranges::max_element(s0), *std::ranges::max_element(s1));
    // Max output value is max_val^2 * len(vec) + 1 : M = m^2 * n + 1
    int min_mod = max_val * max_val * len + 1;
    auto [root, mod] = findParams(len, min_mod);

    std::vector<int> vec0(s0.begin(), s0.end());
    std::vector<int> vec1(s1.begin(), s1.end());
    pwcNtt(vec0, vec1, root, mod);

    return std::make_tuple(vec0, root, mod);
}