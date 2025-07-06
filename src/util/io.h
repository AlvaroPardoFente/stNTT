#pragma once

#include <iostream>
#include <ostream>
#include <vector>

template <class T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v) {
    os << "[";
    for (typename std::vector<T>::const_iterator it = v.begin(); it != v.end(); ++it)
        os << *it << (std::next(it) != v.end() ? ", " : "");
    os << "]";
    return os;
}

template <class T>
void reportDifferences(const std::vector<T> &cpu, const std::vector<T> &gpu) {
    size_t n = std::min(cpu.size(), gpu.size());
    for (size_t i = 0; i < n; ++i) {
        if (cpu[i] != gpu[i]) {
            std::cout << "Index " << i << ": CPU=" << cpu[i] << ", GPU=" << gpu[i] << "\n";
        }
    }
    if (cpu.size() != gpu.size())
        std::cout << "Vectors differ in size (" << cpu.size() << " vs " << gpu.size() << ")\n";
}