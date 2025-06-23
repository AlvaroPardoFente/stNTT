#pragma once

#include <vector>
#include <span>

void nttCt(std::vector<int> &vec, int root, int mod);
void nttStockham(int *vec, int size, int root, int mod);
void nttStockham(int *vec, int size, int root, int mod, int batch);
void pwcNtt(std::vector<int> &vec0, std::vector<int> &vec1, int root, int mod);
std::tuple<std::vector<int>, int, int> pwc_convolution(std::span<const int> s0, std::span<const int> s1);
