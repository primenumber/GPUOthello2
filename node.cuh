#pragma once
#include "types.hpp"
#include "board.cuh"

class MobilityGenerator {
 public:
  __device__ MobilityGenerator() {}
  __host__ __device__ MobilityGenerator(ull player, ull opponent)
    : x(~opponent), y(~player) {}
  MobilityGenerator(const MobilityGenerator &) = default;
  MobilityGenerator& operator=(const MobilityGenerator &) = default;
  __host__ __device__ ull player_pos() const {
    return x & ~y;
  }
  __host__ __device__ ull opponent_pos() const {
    return ~x & y;
  }
  __host__ __device__ ull empty_pos() const {
    return ~(x ^ y);
  }
  __host__ __device__ ull next_bit() {
    ull p = not_checked_yet();
    ull bit = p & -p;
    reset(bit);
    return bit;
  }
  __host__ __device__ bool completed() const {
    return not_checked_yet() == 0;
  }
  __host__ __device__ MobilityGenerator move(ull flip, ull bit) const {
    ull p = player_pos();
    ull o = opponent_pos();
    return MobilityGenerator(o ^ flip, (p ^ flip) | bit);
  }
  __host__ __device__ MobilityGenerator pass() const {
    ull p = player_pos();
    ull o = opponent_pos();
    return MobilityGenerator(o , p);
  }
  __host__ __device__ int score() const {
    return final_score(player_pos(), opponent_pos());
  }
 private:
  __host__ __device__ ull not_checked_yet() const {
    return x & y;
  }
  __host__ __device__ void reset(ull bit) {
    x ^= bit;
    y ^= bit;
  }
  ull x, y;
};

struct Node {
  MobilityGenerator mg;
  short result;
  short alpha;
  short beta;
  bool not_pass;
  bool passed_prev;
  __device__ Node() {}
  __host__ __device__ Node(const MobilityGenerator &mg, int alpha, int beta, bool passed_prev = false)
    : mg(mg), result(-SHRT_MAX), alpha(alpha), beta(beta), not_pass(false), passed_prev(passed_prev) {}
  __host__ __device__ Node(const MobilityGenerator &mg)
    : Node(mg, -64, 64) {}
  Node(const Node &) = default;
  Node& operator=(const Node &) = default;
  __device__ void commit(short score) {
    result = max(result, score);
    alpha = max(alpha, result);
  }
};
