#pragma once

#include <cstdio>

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
    for (int threadID = 0; threadID < blockDim.x * blockDim.y; threadID++) {                     \
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
