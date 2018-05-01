#include "solver.cuh"
#include <iostream>
#include <limits>
#include <vector>
#include <boost/timer/timer.hpp>
#include "to_board.hpp"
#include "expand.cuh"

Table init_table() {
  Table table;
  constexpr size_t table_size = 50000001;
  cudaMallocManaged((void**)&table.entries, sizeof(Entry) * table_size);
  cudaMallocManaged((void**)&table.mutex, sizeof(int) * table_size);
  cudaMallocManaged((void**)&table.update_count, sizeof(ull));
  cudaMallocManaged((void**)&table.hit_count, sizeof(ull));
  cudaMallocManaged((void**)&table.lookup_count, sizeof(ull));
  table.size = table_size;
  *table.update_count = 0;
  *table.hit_count = 0;
  *table.lookup_count = 0;
  memset(table.entries, 0, sizeof(Entry) * table_size);
  memset(table.mutex, 0, sizeof(int) * table_size);
  return table;
}

void destroy_table(Table &table) {
  cudaFree(table.entries);
  cudaFree(table.mutex);
  cudaFree(table.update_count);
  cudaFree(table.hit_count);
  cudaFree(table.lookup_count);
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " YBWC_DEPTH" << std::endl;
  }
  size_t max_depth = std::strtoul(argv[1], nullptr, 10);
  ull black = UINT64_C(0x0000000810000000);
  ull white = UINT64_C(0x0000001008000000);
  Table2 table;
  Evaluator evaluator("subboard6x6.txt", "value6x6/value");
  Table table_cache = init_table();
  while (true) {
    std::unordered_map<int, std::vector<AlphaBetaProblem>> tasks;
    constexpr float INF = std::numeric_limits<float>::infinity();
    float score = expand_ybwc(black, white, -INF, INF, table, evaluator, max_depth, tasks);
    std::cout << tasks.size() << std::endl;
    if (tasks.empty()) {
      std::cout << "Score: " << score << std::endl;
      break;
    }
    for (int level = 0;; ++level) {
      if (tasks[level].empty()) continue;
      std::cout << "Level: " << level << ", size: " << tasks[level].size() << std::endl;
      BatchedTask batched_task;
      init_batch(batched_task, tasks[level].size(), 32 - max_depth, table_cache);
      memcpy(batched_task.abp, tasks[level].data(), sizeof(AlphaBetaProblem) * tasks[level].size());
      launch_batch(batched_task);
      while (true) {
        if (is_ready_batch(batched_task)) break;
      }
      for (std::size_t i = 0; i < tasks[level].size(); ++i) {
        table[std::make_pair(tasks[level][i].player, tasks[level][i].opponent)] = batched_task.result[i];
      }
      destroy_batch(batched_task);
      break;
    }
  }
  destroy_table(table_cache);
  return 0;
}
