#pragma once

#include "ntt/cuda/cu_util.cuh"
#include "ntt/cuda/cu_ntt_util.cuh"
#include "ntt/cuda/implementations/common.cuh"

__global__ void stNttRadix2_128(int *__restrict__ vec, int mod) {
    // 2 buffers for the first shfl (w0 sends 2nd half to w1, w1 sends 1st half to w0)
    // One int for each thread
    extern __shared__ int firstShfl[];

    constexpr uint n = 128;  // Number of elements in the vector
    constexpr uint lN = 7;   // log2(N)

    uint idxInWarp = threadIdx.x & (warpSize - 1);  // 0-31
    uint widx = (threadIdx.x / warpSize) % 2;       // 0 for w0, 1 for w1
    uint wmask = !(widx & 1);                       // 1 for w0, 0 for w1;
    int dPos = (blockIdx.x * n * blockDim.y) + threadIdx.x + (threadIdx.y << lN);
    int *twiddles = (int *)const_twiddles;
    int reg[2];

    reg[0] = vec[dPos];
    reg[1] = vec[dPos + (n >> 1)];

    // First butterfly
    butterfly(reg, twiddles[threadIdx.x], mod);

    // First shfl (shared memory)
    firstShfl[threadIdx.x + threadIdx.y * blockDim.x] = reg[wmask];

    __syncthreads();

    uint woffset =
        wmask ? threadIdx.x + threadIdx.y * blockDim.x + warpSize : threadIdx.x + threadIdx.y * blockDim.x - warpSize;
    reg[wmask] = firstShfl[woffset];

    // From the first shuffle on, all idx are interleaved and a virtual index is used
    uint idxVirtual = (idxInWarp << 1) + widx;

    // Second butterfly
    butterfly(reg, twiddles[(idxVirtual >> 1) << 1], mod);

#ifdef PRINT_STEPS_CUDA
    printfth(
        "[AFTER-BUTTERFLY(01)] tidx[%02u]: %02u, vidx: %02u, reg[%02u]: %02d, reg[%02u]: %02d, bIdx: %02u\n",
        0,
        threadIdx.x,
        idxVirtual,
        0,
        reg[0],
        1,
        reg[1],
        (idxVirtual >> 1) * (1 << 1));
    printff("\n");
#endif

    // No group in warp or gmask since all threads in the warp are used

    int mask = 2, cont = 1;
    int shfl_reg[4];
    for (uint step = 2; step < lN; step++) {
        uint threadvirtual = (((idxVirtual & (mask - 1)) + (idxVirtual >> (cont + 1) << cont)) & ((n >> 2) - 1));
        uint threadvirtual2 = threadvirtual + (n >> 2);
        uint swapidx = (idxVirtual & mask) != 0;

        __syncwarp();

        shfl_reg[0] = __shfl_sync(0xffffffff, reg[0], (threadvirtual >> 1));
        shfl_reg[1] = __shfl_sync(0xffffffff, reg[1], (threadvirtual >> 1));
        shfl_reg[2] = __shfl_sync(0xffffffff, reg[0], (threadvirtual2 >> 1));
        shfl_reg[3] = __shfl_sync(0xffffffff, reg[1], (threadvirtual2 >> 1));

        reg[0] = shfl_reg[swapidx];
        reg[1] = shfl_reg[swapidx + 2];

#ifdef PRINT_STEPS_CUDA
        printfth(
            "[AFTER-SHFL(%02u)] vidx: %02u, reg[%02u]: %02d, reg[%02u]: %02d, threadvirtual: %02u, threadvirtual2: "
            "%02u, threadreal(w): %02u, threadreal2(w): %02u, swapidx: %02u\n",
            step,
            idxVirtual,
            0,
            reg[0],
            1,
            reg[1],
            threadvirtual,
            threadvirtual2,
            (threadvirtual >> 1),
            (threadvirtual2 >> 1),
            swapidx);
        printff("\n");
#endif

        butterfly(reg, twiddles[(idxVirtual >> step) * (1 << step)], mod);

#ifdef PRINT_STEPS_CUDA
        printfth(
            "[AFTER-BUTTERFLY(%02u)] reg[%02u]: %02d, reg[%02u]: %02d, threadvirtual: %02u, threadvirtual2: %02u, "
            "threadreal: %02u, threadreal2: %02u, swapidx: %02u, bIdx: %02u\n",
            step,
            0,
            reg[0],
            1,
            reg[1],
            threadvirtual,
            threadvirtual2,
            (threadvirtual >> 1) + (widx * 32),
            (threadvirtual2 >> 1) + (widx * 32),
            swapidx,
            (idxVirtual >> 1) << 1);
        printff("\n");
#endif

        mask = mask << 1;
        cont++;
    }

    // dPos is calculated again to account for the interleaving done in the first stage
    dPos = (blockIdx.x * n * blockDim.y) + idxVirtual + (threadIdx.y << lN);

    vec[dPos] = reg[0];
    vec[dPos + (n >> 1)] = reg[1];
}