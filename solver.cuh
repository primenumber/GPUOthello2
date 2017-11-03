#pragma once
#include <vector>
#include "types.hpp"

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

struct BatchedTask {
  cudaStream_t *str;
  AlphaBetaProblem *abp;
  UpperNode *upper_stacks;
  int *result;
  size_t max_depth;
  size_t lower_stack_depth;
  size_t size;
  size_t grid_size;
};

void init_batch(BatchedTask &bt, size_t batch_size, size_t max_depth, size_t lower_stack_depth);
void launch_batch(const BatchedTask &bt);
bool is_ready_batch(const BatchedTask &bt);
void destroy_batch(const BatchedTask &bt);
