#include <cinttypes>
#include <chrono>
#include <cstdio>
#include <algorithm>
#include <iostream>
#include <memory>
#include <string>
#include <vector>
#include <tuple>
#include "to_board.hpp"
#include "solver.cuh"
#include "thinker.cuh"

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

std::string hand_to_s(const hand move) {
  if (move == hand::PASS) {
    return "ps";
  } else {
    std::string res(1, static_cast<char>(static_cast<int>(move) % 8 + 'A'));
    res += static_cast<int>(move) / 8 + '1';
    return res;
  }
}

struct FPCloser {
  void operator()(FILE* const fp) const {
    fclose(fp);
  }
};

using file_ptr = std::unique_ptr<FILE, FPCloser>;

std::vector<Board> load_boards(const char* filename) {
  file_ptr fp_in(fopen(filename, "r"));
  int n;
  fscanf(fp_in.get(), "%d", &n);
  std::vector<Board> vboard(n);
  for (int i = 0; i < n; ++i) {
    char buf[33];
    const auto ptr = fgets(buf, 32, fp_in.get());
    if (ptr == nullptr) {
      std::cerr << "Failed to load boards" << std::endl;
      exit(EXIT_FAILURE);
    }
    //fscanf(fp_in.get(), "%s", buf);
    vboard[i] = toBoard(buf);
  }
  std::sort(std::begin(vboard), std::end(vboard));
  vboard.erase(std::unique(std::begin(vboard), std::end(vboard)), std::end(vboard));
  return vboard;
}

void think(int argc, char **argv) {
  using namespace std::literals;
  const auto vboard = load_boards(argv[1]);
  file_ptr fp_out(fopen(argv[2], "w"));
  int max_depth = std::stoi(argv[3]);
  int depth = std::stoi(argv[4]);
  Evaluator evaluator("subboard.txt", "value/value52");
  size_t n = vboard.size();
  fprintf(stderr, "n = %zu\n", n);
  constexpr size_t batch_size = 2000000;
  size_t batch_count = (n + batch_size - 1) / batch_size;
  std::vector<BatchedThinkTask> vb;
  constexpr size_t table_size = 50000001;
  Table table(table_size);
  for (size_t i = 0; i < batch_count; ++i) {
    int size = min(batch_size, n - i*batch_size);
    BatchedThinkTask bt(size, depth, table, evaluator);
    for (int j = 0; j < size; ++j) {
      const auto [p, o] = vboard[i*batch_size+j];
      bt.abp[j] = AlphaBetaProblem(p, o);
    }
    vb.emplace_back(std::move(bt));
  }
  auto start = std::chrono::high_resolution_clock::now();
  for (const auto &b : vb) {
    b.launch();
  }
  while (true) {
    bool finished = true;
    for (const auto &b : vb) {
      if (!b.is_ready()) finished = false;
    }
    if (finished) break;
  }
  const auto status = cudaGetLastError();
  auto ended = std::chrono::high_resolution_clock::now();
  fprintf(stderr, "%s (%s), elapsed: %ldms, table update count: %" PRId64 ", table hit: %" PRId64 ", table find: %" PRId64 "\n",
      cudaGetErrorName(status),
      cudaGetErrorString(status),
      (ended - start) / 1ms,
      table.get_update_count(), table.get_hit_count(), table.get_lookup_count());
  ull total = 0;
  char buf[17];
  for (const auto &b : vb) {
    total += *b.total;
    for (int j = 0; j < b.size; ++j) {
      fromBoard(Board(b.abp[j].player, b.abp[j].opponent), buf);
      fprintf(fp_out.get(), "%s %d %s\n", buf, b.result[j],
          hand_to_s(b.bestmove[j]).c_str());
    }
  }
  fprintf(stderr, "total nodes: %" PRId64 "\n", total);
}

void solve(int argc, char **argv) {
  using namespace std::literals;
  const auto vboard = load_boards(argv[1]);
  file_ptr fp_out(fopen(argv[2], "w"));
  int max_depth = std::stoi(argv[3]);
  const size_t n = vboard.size();
  fprintf(stderr, "n = %zu\n", n);
  constexpr size_t batch_size = 2000000;
  size_t batch_count = (n + batch_size - 1) / batch_size;
  std::vector<BatchedTask> vb;
  constexpr size_t table_size = 50000001;
  Table table(table_size);
  for (size_t i = 0; i < batch_count; ++i) {
    int size = min(batch_size, n - i*batch_size);
    BatchedTask bt(size, max_depth, table);
    for (int j = 0; j < size; ++j) {
      const auto [p, o] = vboard[i*batch_size+j];
      bt.abp[j] = AlphaBetaProblem(p, o);
    }
    vb.emplace_back(std::move(bt));
  }
  fprintf(stderr, "start!\n");
  auto start = std::chrono::high_resolution_clock::now();
  for (const auto &b : vb) {
    b.launch();
  }
  while (true) {
    bool finished = true;
    for (const auto &b : vb) {
      if (!b.is_ready()) finished = false;
    }
    if (finished) break;
  }
  const auto status = cudaGetLastError();
  auto ended = std::chrono::high_resolution_clock::now();
  fprintf(stderr, "%s (%s), elapsed: %ldms, table update count: %" PRId64 ", table hit: %" PRId64 ", table find: %" PRId64 "\n",
      cudaGetErrorName(status),
      cudaGetErrorString(status),
      (ended - start) / 1ms,
      table.get_update_count(), table.get_hit_count(), table.get_lookup_count());
  ull total = 0;
  char buf[17];
  for (const auto &b : vb) {
    total += *b.total;
    for (int j = 0; j < b.size; ++j) {
      fromBoard(Board(b.abp[j].player, b.abp[j].opponent), buf);
      fprintf(fp_out.get(), "%s %d\n", buf, b.result[j]);
    }
  }
  fprintf(stderr, "total nodes: %" PRId64 "\n", total);
}

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s INPUT OUTPUT DEPTH [THINK_DEPTH]\n", argv[0]);
    return 1;
  }
  if (argc == 5) {
    think(argc, argv);
    return 0;
  }
  solve(argc, argv);
}
