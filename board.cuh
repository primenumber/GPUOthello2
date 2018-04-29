#pragma once
#include "types.hpp"

constexpr ull emptymask = 0x007E7E7E7E7E7E00ull;

__host__ __device__ ull flip(ull player, ull opponent, int pos);
__host__ __device__ ull mobility(ull player, ull opponent);
__host__ __device__ int mobility_count(ull player, ull opponent);
inline __host__ __device__ int final_score(ull player, ull opponent) {
#ifdef __CUDA_ARCH__
  int pcnt = __popcll(player);
  int ocnt = __popcll(opponent);
#else
  int pcnt = __builtin_popcountll(player);
  int ocnt = __builtin_popcountll(opponent);
#endif
  if (pcnt == ocnt) return 0;
  if (pcnt > ocnt) return 64 - 2*ocnt;
  return 2*pcnt - 64;
}
__host__ __device__ int stones_count(ull player, ull opponent);
