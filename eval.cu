#include "eval.cuh"
#include <bitset>
#include <fstream>
#include "eval_host.hpp"

__host__ int pow3(int x) {
  if (x == 0) return 1;
  return 3 * pow3(x-1);
}

__host__ Evaluator::Evaluator(const std::string &features_file_name, const std::string &values_file_name) {
  std::ifstream features_stream(features_file_name);
  features_stream >> features_count;
  cudaMallocManaged((void**)&features, sizeof(ull) * features_count);
  for (size_t i = 0; i < features_count; ++i) {
    ull feature = 0;
    for (size_t j = 0; j < 8; ++j) {
      std::bitset<8> bs;
      features_stream >> bs;
      feature |= bs.to_ullong() << (j*8);
    }
    features[i] = feature;
  }

  std::ifstream values_stream(values_file_name);
  cudaMallocManaged((void**)&values, sizeof(float*) * features_count);
  int max_bits_count = 0;
  for (size_t i = 0; i < features_count; ++i) {
    int bits_count = __builtin_popcountll(features[i]);
    max_bits_count = std::max(max_bits_count, bits_count);
    int length = pow3(bits_count);
    cudaMallocManaged((void**)&values[i], sizeof(float) * length);
    values_stream.read((char*)values[i], sizeof(float) * length);
  }

  cudaMallocManaged((void**)&base3_table, sizeof(int) * (1 << max_bits_count));
  for (int i = 0; i < (1 << max_bits_count); ++i) {
    int sum = 0;
    for (int j = 0; j < max_bits_count; ++j) {
      if ((i >> j) & 1) {
        sum += pow3(j);
      }
    }
    base3_table[i] = sum;
  }
}

__host__ __device__ ull parallel_bit_extract(ull x, ull mask) {
#ifdef __CUDA_ARCH__
  ull res = 0;
  int cnt = 0;
  ull bit = 0;
  for (; mask; mask &= mask+bit) {
    bit = mask & -mask;
    res |= (x & mask & ~(mask + bit)) >> (__popcll(bit-1) - cnt);
    cnt += __popcll(mask & ~(mask + bit));
  }
  return res;
#else
  return pext(x, mask);
#endif
}

__host__ __device__ uint64_t flipVertical(uint64_t x) {
  x = (x >> 32) | (x << 32);
  x = ((x >> 16) & UINT64_C(0x0000FFFF0000FFFF)) | ((x << 16) & UINT64_C(0xFFFF0000FFFF0000));
  x = ((x >>  8) & UINT64_C(0x00FF00FF00FF00FF)) | ((x <<  8) & UINT64_C(0xFF00FF00FF00FF00));
  return x;
}

// inline __device__ uint64_t mirrorHorizontal(uint64_t x) {
//   x = ((x >>  4) & UINT64_C(0x0F0F0F0F0F0F0F0F)) | ((x <<  4) & UINT64_C(0xF0F0F0F0F0F0F0F0));
//   x = ((x >>  2) & UINT64_C(0x3333333333333333)) | ((x <<  2) & UINT64_C(0xCCCCCCCCCCCCCCCC));
//   x = ((x >>  1) & UINT64_C(0x5555555555555555)) | ((x <<  1) & UINT64_C(0xAAAAAAAAAAAAAAAA));
//   return x;
// }

__host__ __device__ uint64_t delta_swap(uint64_t bits, uint64_t mask, int delta) {
  uint64_t tmp = mask & (bits ^ (bits << delta));
  return bits ^ tmp ^ (tmp >> delta);
}

__host__ __device__ uint64_t flipDiag(uint64_t bits) {
  uint64_t mask1 = UINT64_C(0x5500550055005500);
  uint64_t mask2 = UINT64_C(0x3333000033330000);
  uint64_t mask3 = UINT64_C(0x0f0f0f0f00000000);
  bits = delta_swap(bits, mask3, 28);
  bits = delta_swap(bits, mask2, 14);
  return delta_swap(bits, mask1, 7);
}

// inline __device__ uint64_t flipDiagA8H1(uint64_t bits) {
//   uint64_t mask1 = UINT64_C(0xaa00aa00aa00aa00);
//   uint64_t mask2 = UINT64_C(0xcccc0000cccc0000);
//   uint64_t mask3 = UINT64_C(0xf0f0f0f000000000);
//   bits = delta_swap(bits, mask3, 36);
//   bits = delta_swap(bits, mask2, 18);
//   return delta_swap(bits, mask1, 9);
// }

__host__ __device__ ull rot90(const ull x) {
  return flipVertical(flipDiag(x));
}

__host__ __device__ int Evaluator::get_index(const ull me, const ull op, const ull feature) {
#ifdef __CUDA_ARCH__
  return __ldg(base3_table + parallel_bit_extract(me, feature))
    + 2 * __ldg(base3_table + parallel_bit_extract(op, feature));
#else
  return base3_table[parallel_bit_extract(me, feature)]
    + 2 * base3_table[parallel_bit_extract(op, feature)];
#endif
}

__host__ __device__ float Evaluator::eval(ull me, ull op) {
  ull me_r = me, op_r = op;
  float score = 0.0f;
  for (int i = 0; i < 4; ++i) {
    for (size_t j = 0; j < features_count; ++j) {
#ifdef __CUDA_ARCH__
      const float * const values_j = (const float *)__ldg((ull*)(values + j));
      score += __ldg(values_j + get_index(me_r, op_r, features[j]));
      score += __ldg(values_j + get_index(flipDiag(me_r), flipDiag(op_r), features[j]));
#else
      const float * const values_j = values[j];
      score += values_j[get_index(me_r, op_r, features[j])];
      score += values_j[get_index(flipDiag(me_r), flipDiag(op_r), features[j])];
#endif
      me_r = rot90(me_r);
      op_r = rot90(op_r);
    }
  }
  return score;
}
