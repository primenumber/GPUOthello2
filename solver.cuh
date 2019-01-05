#pragma once
#include "types.hpp"
#include "alphabeta.cuh"
#include "table.cuh"

class UpperNode;

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
