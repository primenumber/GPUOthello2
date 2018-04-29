#include "board.cuh"

__constant__ ull mask1[4] = {
  0x0080808080808080ULL,
  0x7f00000000000000ULL,
  0x0102040810204000ULL,
  0x0040201008040201ULL
};
constexpr ull mask1_host[4] = {
  0x0080808080808080ULL,
  0x7f00000000000000ULL,
  0x0102040810204000ULL,
  0x0040201008040201ULL
};
__constant__ ull mask2[4] = {
  0x0101010101010100ULL,
  0x00000000000000feULL,
  0x0002040810204080ULL,
  0x8040201008040200ULL
};
constexpr ull mask2_host[4] = {
  0x0101010101010100ULL,
  0x00000000000000feULL,
  0x0002040810204080ULL,
  0x8040201008040200ULL
};

__host__ __device__ ull flip_impl(ull player, ull opponent, int pos, int simd_index) {
  ull om = opponent;
  if (simd_index) om &= 0x7E7E7E7E7E7E7E7EULL;
#ifdef __CUDA_ARCH__
  ull mask = mask1[simd_index] >> (63 - pos);
  ull outflank = (0x8000000000000000ULL >> __clzll(~om & mask)) & player;
#else
  ull mask = mask1_host[simd_index] >> (63 - pos);
  ull outflank = (0x8000000000000000ULL >> __builtin_clzll(~om & mask)) & player;
#endif
  ull flipped = (-outflank << 1) & mask;
#ifdef __CUDA_ARCH__
  mask = mask2[simd_index] << pos;
#else
  mask = mask2_host[simd_index] << pos;
#endif
  outflank = mask & ((om | ~mask) + 1) & player;
  flipped |= (outflank - (outflank != 0)) & mask;
  return flipped;
}

__host__ __device__ ull flip(ull player, ull opponent, int pos) {
  return flip_impl(player, opponent, pos, 0)
    | flip_impl(player, opponent, pos, 1)
    | flip_impl(player, opponent, pos, 2)
    | flip_impl(player, opponent, pos, 3);
}

__host__ __device__ ull mobility_impl(ull player, ull opponent, int simd_index) {
  ull PP, mOO, MM, flip_l, flip_r, pre_l, pre_r, shift2;
  ull shift1[4] = { 1, 7, 9, 8 };
  ull mflipH[4] = { 0x7e7e7e7e7e7e7e7eULL, 0x7e7e7e7e7e7e7e7eULL, 0x7e7e7e7e7e7e7e7eULL, 0xffffffffffffffffULL };

  PP = player;
  mOO = opponent & mflipH[simd_index];
  flip_l  = mOO & (PP << shift1[simd_index]);       flip_r  = mOO & (PP >> shift1[simd_index]);
  flip_l |= mOO & (flip_l << shift1[simd_index]);   flip_r |= mOO & (flip_r >> shift1[simd_index]);
  pre_l   = mOO & (mOO << shift1[simd_index]);      pre_r   = pre_l >> shift1[simd_index];
  shift2 = shift1[simd_index] + shift1[simd_index];
  flip_l |= pre_l & (flip_l << shift2);             flip_r |= pre_r & (flip_r >> shift2);
  flip_l |= pre_l & (flip_l << shift2);             flip_r |= pre_r & (flip_r >> shift2);
  MM = flip_l << shift1[simd_index];                MM |= flip_r >> shift1[simd_index];
  return MM & ~(player|opponent);
}

__host__ __device__ ull mobility(ull player, ull opponent) {
  return emptymask &
    (mobility_impl(player, opponent, 0)
    | mobility_impl(player, opponent, 1)
    | mobility_impl(player, opponent, 2)
    | mobility_impl(player, opponent, 3));
}

__host__ __device__ int mobility_count(ull player, ull opponent) {
#ifdef __CUDA_ARCH__
  return __popcll(mobility(player, opponent));
#else
  return __builtin_popcountll(mobility(player, opponent));
#endif
}

__host__ __device__ int stones_count(ull player, ull opponent) {
#ifdef __CUDA_ARCH__
  return __popcll(player | opponent);
#else
  return __builtin_popcountll(player | opponent);
#endif
}
