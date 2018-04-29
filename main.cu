#include "solver.cuh"
#include <iostream>
#include <limits>
#include <vector>
#include <boost/timer/timer.hpp>
#include "to_board.hpp"
#include "expand.cuh"

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " YBWC_DEPTH" << std::endl;
  }
  size_t max_depth = std::strtoul(argv[0], nullptr, 10);
  ull black = UINT64_C(0x0000000810000000);
  ull white = UINT64_C(0x0000001008000000);
  Table2 table;
  Evaluator evaluator("subboard.txt", "value6x6/value");
  std::vector<AlphaBetaProblem> tasks;
  float INF = std::numeric_limits<float>::infinity();
  expand_ybwc(black, white, -INF, INF, table, evaluator, max_depth, tasks);
  std::cout << tasks.size() << std::endl;
}
