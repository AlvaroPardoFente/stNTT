#pragma once

#include <chrono>
#include <iomanip>

class Timer {
public:
    using Clock = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<double, std::milli>;

    template <bool start = true>
    Timer() {
        if constexpr (start)
            this->start();
    }

    void start() {
        start_ = Clock::now();
    }

    Duration stop() {
        auto end = Clock::now();
        return (end - start_);
    }

    Duration restart() {
        auto now = Clock::now();
        Duration elapsed = now - start_;
        start_ = now;
        return elapsed;
    }

private:
    Clock::time_point start_;
};

template <typename T = double, typename Period = std::milli>
std::ostream& operator<<(std::ostream& os, const std::chrono::duration<T, Period>& d) {
    if constexpr (std::is_same_v<Period, std::chrono::seconds>)
        return os << std::fixed << std::setprecision(2) << d.count() << " s";
    else if constexpr (std::is_same_v<Period, std::milli>)
        return os << std::fixed << std::setprecision(2) << d.count() << " ms";
    else if constexpr (std::is_same_v<Period, std::micro>)
        return os << std::fixed << std::setprecision(2) << d.count() << " µs";
    else if constexpr (std::is_same_v<Period, std::nano>)
        return os << std::fixed << std::setprecision(2) << d.count() << " ns";
    else
        return os << std::fixed << std::setprecision(2) << d.count() << " ?";
}