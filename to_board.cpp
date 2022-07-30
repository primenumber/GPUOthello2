#include "to_board.hpp"

#include <cstdint>

#include "x86intrin.h"

// divide 16 integers under 27 by 9
__m128i div9_epu8(__m128i x) {
  __m128i hi = _mm_and_si128(_mm_maddubs_epi16(x, _mm_set1_epi16(0x1d00)),
                             _mm_set1_epi16(0xFF00));
  __m128i lo = _mm_srli_epi16(_mm_maddubs_epi16(x, _mm_set1_epi16(0x001d)), 8);
  return _mm_or_si128(hi, lo);
}

std::pair<ull, ull> toBoard(const char* const str) {
  __m128i data = _mm_loadu_si128((__m128i*)str);
  data = _mm_sub_epi8(data, _mm_set1_epi8(33));  // substruct offset
  __m128i b =
      _mm_and_si128(_mm_srli_epi16(data, 2), _mm_set1_epi8(0x08));  // a3 black
  __m128i w =
      _mm_and_si128(_mm_srli_epi16(data, 3), _mm_set1_epi8(0x08));  // a3 white
  data = _mm_and_si128(data, _mm_set1_epi8(0x1F));
  __m128i a2 = div9_epu8(data);
  b |= _mm_and_si128(_mm_slli_epi16(a2, 2), _mm_set1_epi8(0x04));  // a2 black
  w |= _mm_and_si128(_mm_slli_epi16(a2, 1), _mm_set1_epi8(0x04));  // a2 white
  data = _mm_sub_epi8(data, _mm_mullo_epi16(a2, _mm_set1_epi16(9)));
  __m128i table1 = _mm_setr_epi8(0x0, 0x1, 0x0, 0x2, 0x3, 0x2, 0x0, 0x1, 0x0, 0,
                                 0, 0, 0, 0, 0, 0);
  __m128i table2 = _mm_setr_epi8(0x0, 0x0, 0x1, 0x0, 0x0, 0x1, 0x2, 0x2, 0x3, 0,
                                 0, 0, 0, 0, 0, 0);
  b |= _mm_shuffle_epi8(table1, data);
  w |= _mm_shuffle_epi8(table2, data);
  b = _mm_maddubs_epi16(b, _mm_set1_epi16(0x1001));
  w = _mm_maddubs_epi16(w, _mm_set1_epi16(0x1001));
  __m128i res = _mm_packus_epi16(b, w);
  ull player = _mm_cvtsi128_si64x(res);
  ull opponent = _mm_extract_epi64(res, 1);
  return std::make_pair(player, opponent);
}

__m128i delta_swap_epi16(__m128i x, __m128i mask, int delta) {
  auto t = mask & (x ^ _mm_srli_epi16(x, delta));
  return t ^ _mm_slli_epi16(t, delta) ^ x;
}

void fromBoard(const Board& board, char* const str) {
  auto v = _mm_set_epi64x(board.second, board.first);
  const auto sfl = _mm_setr_epi8(0x0, 0x8, 0x1, 0x9, 0x2, 0xa, 0x3, 0xb, 0x4,
                                 0xc, 0x5, 0xd, 0x6, 0xe, 0x7, 0xf);
  v = _mm_shuffle_epi8(v, sfl);
  v = delta_swap_epi16(v, _mm_set1_epi16(0xf0), 4);
  const auto b3tbl =
      _mm_setr_epi8(0, 1, 3, 4, 9, 10, 12, 13, 32, 33, 35, 36, 41, 42, 44, 45);
  const auto p3b = _mm_shuffle_epi8(b3tbl, v & _mm_set1_epi8(0x0f));
  const auto o3b = _mm_slli_epi16(
      _mm_shuffle_epi8(b3tbl, _mm_srli_epi16(v, 4) & _mm_set1_epi8(0x0f)), 1);
  const auto ans = _mm_add_epi8(_mm_add_epi8(p3b, o3b), _mm_set1_epi8(33));
  _mm_storeu_si128(reinterpret_cast<__m128i*>(str), ans);
  str[16] = '\0';
}
