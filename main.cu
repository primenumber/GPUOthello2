#include "solver.cuh"
#include <algorithm>
#include <iostream>
#include <list>
#include <queue>
#include <string>
#include <vector>
#include <tuple>
#include <boost/timer/timer.hpp>
#include "to_board.hpp"
#include "task.cuh"

constexpr size_t batch_size = 8192;

void output_board(const ull p, const ull o) {
  for (int i = 0; i < 8; ++i) {
    for (int j = 0; j < 8; ++j) {
      int index = i*8+j;
      if ((p >> index) & 1) {
        printf("x");
      } else if ((o >> index) & 1) {
        printf("o");
      } else {
        printf(".");
      }
    }
    printf("\n");
  }
  printf("\n");
}

struct Batch {
  BatchedTask bt;
  std::vector<size_t> id;
};

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s INPUT OUTPUT DEPTH\n", argv[0]);
    return 1;
  }
  FILE *fp_in = fopen(argv[1], "r");
  FILE *fp_out = fopen(argv[2], "w");
  int max_depth = std::stoi(argv[3]);
  int n;
  fscanf(fp_in, "%d", &n);
  std::vector<std::string> vboard(n);
  for (int i = 0; i < n; ++i) {
    char buf[17];
    fscanf(fp_in, "%s", buf);
    vboard[i] = buf;
  }
  std::sort(std::begin(vboard), std::end(vboard));
  vboard.erase(std::unique(std::begin(vboard), std::end(vboard)), std::end(vboard));
  n = vboard.size();
  fprintf(stderr, "n = %d\n", n);
  std::list<Batch> batches;
  std::queue<size_t> cpu_queue, gpu_queue;
  std::vector<CPUStack> stacks;
  Table table;
  for (size_t i = 0; i < n; ++i) {
    cpu_queue.push(i);
    ull player, opponent;
    std::tie(player, opponent) = toBoard(vboard[i].c_str());
    stacks.emplace_back(CPUNode(player, opponent), table);
  }
  int cnt = 0;
  while (true) {
    bool finished = true;
    for (auto itr = std::begin(batches); itr != std::end(batches); ) {
      finished = false;
      if (is_ready_batch(itr->bt)) {
        fprintf(stderr, "ok %d %d\n", cnt++, itr->bt.size);
        for (size_t i = 0; i < itr->bt.size; ++i) {
          stacks[itr->id[i]].update(itr->bt.result[i]);
          cpu_queue.push(itr->id[i]);
        }
        destroy_batch(itr->bt);
        itr = batches.erase(itr);
      } else {
        ++itr;
      }
    }
    while (!cpu_queue.empty()) {
      finished = false;
      size_t id = cpu_queue.front();
      cpu_queue.pop();
      if (stacks[id].run()) continue;
      gpu_queue.push(id);
    }
    while (!gpu_queue.empty()) {
      finished = false;
      size_t size = std::min(batch_size, gpu_queue.size());
      batches.emplace_back();
      init_batch(batches.back().bt, size, gpu_depth);
      for (size_t i = 0; i < size; ++i) {
        size_t id = gpu_queue.front();
        gpu_queue.pop();
        CPUNode &top = stacks[id].get_top();
        if (i == 0) {
          fprintf(stderr, "%d %d\n", top.get_alpha(), top.get_beta());
        }
        batches.back().bt.abp[i] = AlphaBetaProblem(top.player_pos(), top.opponent_pos(), top.get_alpha(), top.get_beta());
        batches.back().id.push_back(id);
      }
      launch_batch(batches.back().bt);
    }
    if (finished) break;
  }
  for (size_t i = 0; i < n; ++i) {
    fprintf(fp_out, "%s %d\n", vboard[i].c_str(), stacks[i].get_top().get_result());
  }
  return 0;
}
