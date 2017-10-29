#include "to_board.hpp"
#include "x86intrin.h"

// divide 16 integers under 27 by 9
__m128i div9_epu8(__m128i x) {
  __m128i hi = _mm_and_si128(_mm_maddubs_epi16(x, _mm_set1_epi16(0x1d00)), _mm_set1_epi16(0xFF00));
  __m128i lo = _mm_srli_epi16(_mm_maddubs_epi16(x, _mm_set1_epi16(0x001d)), 8);
  return _mm_or_si128(hi, lo);
}

std::pair<ull, ull> toBoard(const char * const str) {
  __m128i data = _mm_loadu_si128((__m128i*)str);
  data = _mm_sub_epi8(data, _mm_set1_epi8(33)); // substruct offset
  __m128i b = _mm_and_si128(_mm_srli_epi16(data, 2), _mm_set1_epi8(0x08)); // a3 black
  __m128i w = _mm_and_si128(_mm_srli_epi16(data, 3), _mm_set1_epi8(0x08)); // a3 white
  data = _mm_and_si128(data, _mm_set1_epi8(0x1F));
  __m128i a2 = div9_epu8(data);
  b |= _mm_and_si128(_mm_slli_epi16(a2, 2), _mm_set1_epi8(0x04)); // a2 black
  w |= _mm_and_si128(_mm_slli_epi16(a2, 1), _mm_set1_epi8(0x04)); // a2 white
  data = _mm_sub_epi8(data, _mm_mullo_epi16(a2, _mm_set1_epi16(9)));
  __m128i table1 = _mm_setr_epi8(0x0, 0x1, 0x0, 0x2, 0x3, 0x2, 0x0, 0x1, 0x0, 0,0,0,0,0,0,0);
  __m128i table2 = _mm_setr_epi8(0x0, 0x0, 0x1, 0x0, 0x0, 0x1, 0x2, 0x2, 0x3, 0,0,0,0,0,0,0);
  b |= _mm_shuffle_epi8(table1, data);
  w |= _mm_shuffle_epi8(table2, data);
  b = _mm_maddubs_epi16(b, _mm_set1_epi16(0x1001));
  w = _mm_maddubs_epi16(w, _mm_set1_epi16(0x1001));
  __m128i res = _mm_packus_epi16(b, w);
  ull player = _mm_cvtsi128_si64x(res);
  ull opponent = _mm_extract_epi64(res, 1);
  return std::make_pair(player, opponent);
}
