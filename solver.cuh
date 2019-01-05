#pragma once
#include <vector>
#include "types.hpp"
#include "table.cuh"
#include "eval.cuh"

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

class UpperNode;
class ThinkerNode;

struct BatchedTask {
  BatchedTask(const size_t batch_size, const size_t max_depth, const Table& table);
  ~BatchedTask();
  BatchedTask(const BatchedTask&) = default;
  BatchedTask(BatchedTask&&);
  void launch() const;
  bool is_ready() const;
  cudaStream_t *str;
  AlphaBetaProblem *abp;
  UpperNode *upper_stacks;
  Table table;
  int *result;
  size_t max_depth;
  size_t size;
  size_t grid_size;
  ull *total;
};

struct BatchedThinkTask {
  BatchedThinkTask(const size_t batch_size, const size_t depth,
      const Table& table, const Evaluator& evaluator);
  ~BatchedThinkTask();
  BatchedThinkTask(const BatchedThinkTask&) = default;
  BatchedThinkTask(BatchedThinkTask&&);
  void launch() const;
  bool is_ready() const;
  cudaStream_t *str;
  AlphaBetaProblem *abp;
  ThinkerNode *thinker_stacks;
  Table table;
  const Evaluator evaluator;
  int *result;
  hand *bestmove;
  size_t depth;
  size_t size;
  size_t grid_size;
  ull *total;
};
