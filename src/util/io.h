#pragma once

#include <ostream>
#include <vector>

template <class T>
std::ostream &operator<<(std::ostream &os, const std::vector<T> &v)
{
    os << "[";
    for (typename std::vector<T>::const_iterator it = v.begin(); it != v.end(); ++it)
        os << *it << (std::next(it) != v.end() ? ", " : "");
    os << "]";
    return os;
}