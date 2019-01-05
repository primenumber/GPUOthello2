#pragma once
#include "types.hpp"

constexpr int nodesPerBlock = 64;
constexpr int chunk_size = 2048;

enum class hand : char {
  PASS = 64,
  NOMOVE = 65
};

struct AlphaBetaProblem {
  __host__ __device__ AlphaBetaProblem(ull player, ull opponent, int alpha, int beta)
    : player(player), opponent(opponent), alpha(alpha), beta(beta) {}
  __host__ __device__ AlphaBetaProblem(ull player, ull opponent)
    : player(player), opponent(opponent), alpha(-64), beta(64) {}
  ull player;
  ull opponent;
  int alpha;
  int beta;
};
