#pragma once

#include <vector>
#include <span>
#include <optional>

void nttCt(std::vector<int> &vec, int root, int mod);
void nttStockham(int *vec, int size, int root, int mod, std::optional<int> nsteps = std::nullopt);
void nttStockham(std::span<int> vec, int size, int root, int mod, int batch, std::optional<int> nsteps = std::nullopt);
// A version of stockham that only shuffles indices and does not perform butterfly operations
void nttStockhamIdx(int *vec, int size, int root, int mod, std::optional<int> nsteps = std::nullopt);
void nttStockhamIdx(
    std::span<int> vec,
    int size,
    int root,
    int mod,
    int batches,
    std::optional<int> nsteps = std::nullopt);
void pwcNtt(std::vector<int> &vec0, std::vector<int> &vec1, int root, int mod);
std::tuple<std::vector<int>, int, int> pwc_convolution(std::span<const int> s0, std::span<const int> s1);
