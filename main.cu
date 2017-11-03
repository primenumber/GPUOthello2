#include "solver.cuh"
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include "to_board.hpp"

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

int main(int argc, char **argv) {
  if (argc < 5) {
    fprintf(stderr, "usage: %s INPUT OUTPUT DEPTH NAIVE_DEPTH\n", argv[0]);
    return 1;
  }
  FILE *fp_in = fopen(argv[1], "r");
  FILE *fp_out = fopen(argv[2], "w");
  int max_depth = std::stoi(argv[3]);
  int lower_stack_depth = std::stoi(argv[4]);
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
  BatchedTask bt;
  init_batch(bt, n, max_depth, lower_stack_depth);
  for (int i = 0; i < n; ++i) {
    ull player, opponent;
    std::tie(player, opponent) = toBoard(vboard[i].c_str());
    bt.abp[i] = AlphaBetaProblem(player, opponent);
  }
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start, *bt.str);
  launch_batch(bt);
  cudaEventRecord(stop, *bt.str);
  cudaEventSynchronize(stop);
  float elapsed = 0;
  cudaEventElapsedTime(&elapsed, start, stop);
  fprintf(stderr, "%s, elapsed: %.6fs\n", cudaGetErrorString(cudaGetLastError()), elapsed/1000.0);
  for (int i = 0; i < n; ++i) {
    fprintf(fp_out, "%s %d\n", vboard[i].c_str(), bt.result[i]);
  }
  destroy_batch(bt);
}
