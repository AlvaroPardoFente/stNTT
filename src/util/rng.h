#pragma once

#include <random>
#include <vector>

namespace util {
// A random int generator
class Rng {
public:
    const static int defaultSeed = 42;

    Rng() {
        std::random_device rd;
        engine.seed(rd());
    };
    Rng(int seed) {
        engine.seed(seed);
    }

    int get() {
        return dist(engine);
    }

    std::vector<int> get_vector(size_t size) {
        std::vector<int> vec(size);
        for (auto& e : vec)
            e = dist(engine);

        return vec;
    }

private:
    std::mt19937 engine;
    // For now, the range is hard coded
    std::uniform_int_distribution<int> dist{0, 9};
};
}  // namespace util