#include <iomanip>
#include <iostream>
#include <random>

#include "to_board.hpp"

void fromBoard_n(const Board& board, char* const str) {
  for (size_t i = 0; i < 16; ++i) {
    uint8_t p = (board.first >> (i * 4)) & 0b1111;
    uint8_t o = (board.second >> (i * 4)) & 0b1111;
    uint8_t v = 33;
    uint8_t t[4] = {1, 3, 9, 32};
    for (size_t j = 0; j < 4; ++j) {
      if ((p >> j) & 1) v += t[j];
      if ((o >> j) & 1) v += 2 * t[j];
    }
    str[i] = v;
  }
}

int main() {
  constexpr size_t N = 1000;
  std::random_device rd;
  std::mt19937_64 mt(rd());
  std::uniform_int_distribution<uint64_t> dis_board;
  for (size_t i = 0; i < N; ++i) {
    auto x = dis_board(mt);
    auto y = dis_board(mt);
    auto p = x & y;
    auto o = x & ~y;
    Board b(p, o);
    char buf[17] = {};
    char buf_n[17] = {};
    fromBoard(b, buf);
    fromBoard_n(b, buf_n);
    auto b2 = toBoard(buf);
    if (b != b2) {
      std::cerr << std::hex << p << " " << o << " " << buf_n << " " << buf
                << std::endl;
    }
  }
  return 0;
}
