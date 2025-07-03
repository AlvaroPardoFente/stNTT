#pragma once

#include <cstdio>
#include <span>
#include <stdexcept>
#include <cuda_runtime.h>

constexpr uint warpSizeConst = 32;

// Prints the error, description, file and line if a CUDA error occurs.
#define CCErr(val) checkPrintErr((val), #val, __FILE__, __LINE__)

// Prints the extended description of a CUDA error given its code.
template <typename T>
void checkPrintErr(T result, char const *const func, const char *const file, int const line) {
    if (result) {
        fprintf(
            stderr,
            "CUDA error at %s:%d: %s, %d: %s\n",
            file,
            line,
            func,
            static_cast<unsigned int>(result),
            cudaGetErrorString(static_cast<cudaError_t>(result)));
        exit(EXIT_FAILURE);
    }
}

// Printf synchronized ordered by threadIdx
#define printft(...)                                                  \
    for (int threadID = 0; threadID < warpSize; threadID++) {         \
        int global_thread_id = blockIdx.x * blockDim.x + threadIdx.x; \
        if (threadID == global_thread_id) {                           \
            printf(__VA_ARGS__);                                      \
        }                                                             \
        __syncthreads();                                              \
    }

// Printf synchronized ordered by threadIdx, including threadId header
#define printfth(...)                                                                            \
    for (int threadID = 0; threadID < gridDim.x * blockDim.x * blockDim.y; threadID++) {         \
        int global_thread_id = blockIdx.x * blockDim.x + threadIdx.y * blockDim.x + threadIdx.x; \
        if (threadID == global_thread_id) {                                                      \
            printf("(%02d)[%02d,%02d] ", blockIdx.x, threadIdx.x, threadIdx.y);                  \
            printf(__VA_ARGS__);                                                                 \
        }                                                                                        \
        __syncthreads();                                                                         \
    }

// Printf for the first thread in the block
#define printff(...)         \
    if (threadIdx.x == 0) {  \
        printf(__VA_ARGS__); \
    }                        \
    __syncthreads();

namespace cuda {
template <typename T>
__device__ __forceinline__ void swap(T &a, T &b) {
    T temp = a;
    a = b;
    b = temp;
}

// GPU buffer wrapper for simple cases
template <typename T>
struct Buffer {
    Buffer(std::span<T> span) {
        this->load(span);
    }

    ~Buffer() {
        if (this->data() != nullptr) {
            CCErr(cudaFree(this->data()));
        }
    }

    void alloc(size_t size) {
        if (size < 1ul)
            throw std::invalid_argument("Size mut be grater than zero");
        this->size_ = size;
        CCErr(cudaMalloc(&data_, this->bytes()));
    }

    void load(std::span<T> span) {
        this->alloc(span.size());
        CCErr(cudaMemcpy(this->data(), span.data(), this->bytes(), cudaMemcpyHostToDevice));
    }

    void store(std::span<T> span) {
        if (span.size_bytes() < this->bytes())
            throw std::out_of_range("Host buffer is smaller than device buffer");
        CCErr(cudaMemcpy(span.data(), this->data(), this->bytes(), cudaMemcpyDeviceToHost));
    }

    T *data() {
        return data_;
    }

    size_t size() {
        return size_;
    }

    size_t bytes() {
        return this->size() * sizeof(T);
    }

private:
    T *data_{};
    size_t size_{};
};
}  // namespace cuda