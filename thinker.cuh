#pragma once
#include "types.hpp"
#include "alphabeta.cuh"
#include "table.cuh"
#include "eval.cuh"

class ThinkerNode;

struct BatchedThinkTask {
  BatchedThinkTask(const size_t batch_size, const size_t depth,
      const Table& table, const Evaluator& evaluator);
  ~BatchedThinkTask();
  BatchedThinkTask(BatchedThinkTask&&);
  void launch() const;
  bool is_ready() const;
  cudaStream_t *str;
  AlphaBetaProblem *abp;
  ThinkerNode *thinker_stacks;
  const Table& table;
  const Evaluator evaluator;
  int *result;
  hand *bestmove;
  size_t depth;
  size_t size;
  size_t grid_size;
  ull *total;
 private:
  BatchedThinkTask(const BatchedThinkTask&) = default;
};
