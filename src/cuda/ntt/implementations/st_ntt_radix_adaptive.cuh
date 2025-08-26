#pragma once

#include "cuda/cu_util.cuh"
#include "cuda/ntt/arithmetic.cuh"
#include "cuda/ntt/implementations/common.cuh"

template <uint n, uint radix>
__global__ void stNttRadixAdaptive(int *__restrict__ vec, int mod) {
    constexpr uint lN = cuda::log2_constexpr(n);
    constexpr uint radixDiv2 = radix >> 1;
    constexpr uint radixIdx = cuda::log2_constexpr(radix);

    // All simulated thread indices this thread will use (radix 2 -> 1, radix 4 -> 2...)
    uint tidx[radixDiv2];
    for (uint i = 0; i < radixDiv2; i++)
        tidx[i] = threadIdx.x + (i * (n >> radixIdx));

    // Positions in global memory for all the simulated threads
    uint dPos[radixDiv2];
    for (uint i = 0; i < radixDiv2; i++)
        dPos[i] = (blockIdx.x * n * blockDim.y) + tidx[i] + (threadIdx.y << lN);

    int mask = 1, cont = 0;
    int *twiddles = (int *)const_twiddles;

    // 2 registers for each simulated thread
    int reg[radix];
    for (uint i = 0; i < radix; i += 2) {
        reg[i] = vec[dPos[i >> 1]];
        reg[i + 1] = vec[dPos[i >> 1] + (n >> 1)];  // N[1] == N / 2
    }

    // TODO: Check if other implementations are faster for radix > 2
    for (uint i = 0; i < radixDiv2; i++)
        cuda::ntt::butterfly(&(reg[i * 2]), twiddles[tidx[i]], mod);

    uint wgidx = ((n >> radixIdx) * threadIdx.y & (warpSize - 1));  // Group index in warp
    uint gmask = ~(0xffffffff << (n >> radixIdx)) << wgidx;

    // Two shfl registers and one swapidx for each reg
    int shfl_reg[radix << 1];
    uint swapidx[radixDiv2];

    uint threadvirtual[radix];
    for (uint step = 1; step < lN; step++) {
        for (uint i = 0; i < radix; i += 2) {
            // All thread indexes for shfl are stored relative to the group, not the warp
            threadvirtual[i] =
                (((tidx[i >> 1] & (mask - 1)) + (tidx[i >> 1] >> (cont + 1) << cont)) &
                 ((n >> 2) - 1));                                  // N[2] == N / 4
            threadvirtual[i + 1] = (threadvirtual[i] + (n >> 2));  // & (N[radixIdx] - 1) == modulo nthreads
        }

        for (uint i = 0; i < radixDiv2; i++)
            swapidx[i] = (tidx[i] & mask) != 0;

        __syncwarp(gmask);

        for (uint i = 0; i < radixDiv2; i++) {
            // Index of the register in the shfl_reg array
            uint regIdx = i * 4;
            // Index of the pairs of [0, 1] registers in the remote reg. The threadvirtual value is used to
            // index the specific register in the array for the radix > 4 cases (where each thread holds more than 2
            // pairs)
            uint pairIdxInThread = 2 * (threadvirtual[(i * 2)] / (n >> radixIdx));
            uint pairIdxInThread2 = 2 * (threadvirtual[(i * 2 + 1)] / (n >> radixIdx));

            shfl_reg[regIdx] =
                __shfl_sync(gmask, reg[(pairIdxInThread) % radix], threadvirtual[(i * 2)] % (n >> radixIdx) + wgidx);
            shfl_reg[regIdx + 1] = __shfl_sync(
                gmask,
                reg[(pairIdxInThread + 1) % radix],
                threadvirtual[(i * 2)] % (n >> radixIdx) + wgidx);
            shfl_reg[regIdx + 2] = __shfl_sync(
                gmask,
                reg[(pairIdxInThread2) % radix],
                threadvirtual[(i * 2 + 1)] % (n >> radixIdx) + wgidx);
            shfl_reg[regIdx + 3] = __shfl_sync(
                gmask,
                reg[(pairIdxInThread2 + 1) % radix],
                threadvirtual[(i * 2 + 1)] % (n >> radixIdx) + wgidx);
        }

        for (uint i = 0; i < radix; i++)
            reg[i] = shfl_reg[swapidx[i >> 1] + (i << 1)];

#ifdef PRINT_STEPS_CUDA
        for (uint i = 0; i < radixDiv2; i++) {
            printfth(
                "[AFTER-SHFL(%02u)] reg[%02u]: %02d, reg[%02u]: %02d, threadvirtual: %02u, threadvirtual2: %02u, "
                "swapidx: %02u\n",
                step,
                i * 2,
                reg[i * 2],
                i * 2 + 1,
                reg[i * 2 + 1],
                threadvirtual[i * 2],
                threadvirtual[i * 2 + 1],
                swapidx[i]);
        }
        printff("\n");
#endif

        // TODO: Check if other implementations are faster for radix > 2
        for (uint i = 0; i < radixDiv2; i++)
            cuda::ntt::butterfly(&(reg[i << 1]), twiddles[(tidx[i] >> step) * (1 << step)], mod);

#ifdef PRINT_STEPS_CUDA
        for (uint i = 0; i < radixDiv2; i++) {
            printfth(
                "[AFTER-BUTTERFLY(%02u)] reg[%02u]: %02d, reg[%02u]: %02d, threadvirtual: %02u, threadvirtual2: %02u, "
                "swapidx: %02u, bIdx: %02u\n",
                step,
                i * 2,
                reg[i * 2],
                i * 2 + 1,
                reg[i * 2 + 1],
                threadvirtual[i * 2],
                threadvirtual[i * 2 + 1],
                swapidx[i],
                (tidx[i] >> step) * (1 << step));
        }
        printff("\n");
#endif

        mask = mask << 1;
        cont++;
    }

    for (uint i = 0; i < radix; i += 2) {
        vec[dPos[i >> 1]] = reg[i];
        vec[dPos[i >> 1] + (n >> 1)] = reg[i + 1];  // N[1] == N / 2
    }
}