#pragma once

#define MAX_TWIDDLES 1024
__constant__ void *const_twiddles[MAX_TWIDDLES * sizeof(int)];

constexpr uint warpSizeConst = 32;