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
  cudaStream_t *str;
  AlphaBetaProblem *abp;
  ThinkerNode *thinker_stacks;
  Table table;
  Evaluator evaluator;
  int *result;
  hand *bestmove;
  size_t depth;
  size_t size;
  size_t grid_size;
  ull *total;
};

void init_batch(BatchedTask &bt, size_t batch_size, size_t max_depth, const Table &table);
void init_batch(BatchedThinkTask &bt, size_t batch_size, size_t depth, const Table &table, const Evaluator &evaluator);
void launch_batch(const BatchedTask &bt);
void launch_batch(const BatchedThinkTask &bt);
bool is_ready_batch(const BatchedTask &bt);
bool is_ready_batch(const BatchedThinkTask &bt);
void destroy_batch(const BatchedTask &bt);
void destroy_batch(const BatchedThinkTask &bt);
