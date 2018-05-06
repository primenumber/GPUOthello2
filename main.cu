#include "solver.cuh"
#include <algorithm>
#include <iostream>
#include <string>
#include <vector>
#include <tuple>
#include <boost/timer/timer.hpp>
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

struct Batch {
  BatchedTask bt;
  std::vector<std::string> vstr;
};

struct ThinkBatch {
  BatchedThinkTask bt;
  std::vector<std::string> vstr;
};

void think(char **argv) {
  FILE *fp_in = fopen(argv[1], "r");
  FILE *fp_out = fopen(argv[2], "w");
  int max_depth = std::stoi(argv[3]);
  int depth = std::stoi(argv[4]);
  Evaluator evaluator("subboard.txt", "value/value52");
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
  constexpr size_t batch_size = 8192;
  size_t batch_count = (n + batch_size - 1) / batch_size;
  std::vector<ThinkBatch> vb(batch_count);
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
  for (size_t i = 0; i < batch_count; ++i) {
    int size = min(batch_size, n - i*batch_size);
    init_batch(vb[i].bt, size, depth, table, evaluator);
    for (int j = 0; j < vb[i].bt.size; ++j) {
      ull player, opponent;
      std::tie(player, opponent) = toBoard(vboard[i*batch_size+j].c_str());
      vb[i].bt.abp[j] = AlphaBetaProblem(player, opponent);
      vb[i].vstr.push_back(vboard[i*batch_size+j]);
    }
  }
  boost::timer::cpu_timer timer;
  for (const auto &b : vb) {
    launch_batch(b.bt);
  }
  while (true) {
    bool finished = true;
    for (const auto &b : vb) {
      if (!is_ready_batch(b.bt)) finished = false;
    }
    if (finished) break;
  }
  fprintf(stderr, "%s, elapsed: %.6fs, table update count: %llu, table hit: %llu, table find: %llu\n",
      cudaGetErrorString(cudaGetLastError()), timer.elapsed().wall/1000000000.0,
      *table.update_count, *table.hit_count, *table.lookup_count);
  ull total = 0;
  for (const auto &b : vb) {
    total += *b.bt.total;
    for (int j = 0; j < b.bt.size; ++j) {
      fprintf(fp_out, "%s %d\n", b.vstr[j].c_str(), b.bt.result[j]);
    }
    destroy_batch(b.bt);
  }
  fprintf(stderr, "total nodes: %llu\n", total);
  cudaFree(table.entries);
  cudaFree(table.mutex);
  cudaFree(table.update_count);
  cudaFree(table.hit_count);
  cudaFree(table.lookup_count);
}

int main(int argc, char **argv) {
  if (argc < 4) {
    fprintf(stderr, "usage: %s INPUT OUTPUT DEPTH [THINK_DEPTH]\n", argv[0]);
    return 1;
  }
  if (argc == 5) {
    think(argv);
    return 0;
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
  constexpr size_t batch_size = 8192;
  size_t batch_count = (n + batch_size - 1) / batch_size;
  std::vector<Batch> vb(batch_count);
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
  for (size_t i = 0; i < batch_count; ++i) {
    int size = min(batch_size, n - i*batch_size);
    init_batch(vb[i].bt, size, max_depth, table);
    for (int j = 0; j < vb[i].bt.size; ++j) {
      ull player, opponent;
      std::tie(player, opponent) = toBoard(vboard[i*batch_size+j].c_str());
      vb[i].bt.abp[j] = AlphaBetaProblem(player, opponent);
      vb[i].vstr.push_back(vboard[i*batch_size+j]);
    }
  }
  boost::timer::cpu_timer timer;
  for (const auto &b : vb) {
    launch_batch(b.bt);
  }
  while (true) {
    bool finished = true;
    for (const auto &b : vb) {
      if (!is_ready_batch(b.bt)) finished = false;
    }
    if (finished) break;
  }
  fprintf(stderr, "%s, elapsed: %.6fs, table update count: %llu, table hit: %llu, table find: %llu\n",
      cudaGetErrorString(cudaGetLastError()), timer.elapsed().wall/1000000000.0,
      *table.update_count, *table.hit_count, *table.lookup_count);
  ull total = 0;
  for (const auto &b : vb) {
    total += *b.bt.total;
    for (int j = 0; j < b.bt.size; ++j) {
      fprintf(fp_out, "%s %d\n", b.vstr[j].c_str(), b.bt.result[j]);
    }
    destroy_batch(b.bt);
  }
  fprintf(stderr, "total nodes: %llu\n", total);
  cudaFree(table.entries);
  cudaFree(table.mutex);
  cudaFree(table.update_count);
  cudaFree(table.hit_count);
  cudaFree(table.lookup_count);
}
