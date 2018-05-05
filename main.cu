#include "solver.cuh"
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
#include <boost/timer/timer.hpp>
#include "to_board.hpp"
#include "board.cuh"
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

void solve_parallel(const std::string &in, const std::string &out) {
  std::ifstream ifs(in);
  std::size_t n, depth;
  ifs >> n >> depth;
  std::vector<std::string> prob_str(n);
  Table table = init_table();
  BatchedTask bt;
  init_batch(bt, n, depth, table);
  for (std::size_t i = 0; i < n; ++i) {
    ifs >> prob_str[i];
    ull player, opponent;
    std::tie(player, opponent) = toBoard(prob_str[i].c_str());
    bt.abp[i] = AlphaBetaProblem(player, opponent, -64, 64);
  }
  launch_batch(bt);
  while (true) {
    if (is_ready_batch(bt)) break;
  }
  std::ofstream ofs(out);
  ofs << n << std::endl;
  for (std::size_t i = 0; i < n; ++i) {
    ofs << prob_str[i] << ' ' << bt.result[i] << std::endl;
  }
  destroy_batch(bt);
}

std::string to_base81(ull player, ull opponent) {
  std::string res;
  for (int i = 0; i < 16; ++i) {
    ull p = (player >> (i * 4)) & 0x0F;
    ull o = (opponent >> (i * 4)) & 0x0F;
    int coeff[4] = {1, 3, 9, 32};
    char c = 33;
    for (int j = 0; j < 4; ++j) {
      bool isp = (p >> j) & 1;
      bool iso = (o >> j) & 1;
      if (isp) {
        c += coeff[j];
      } else if (iso) {
        c += coeff[j] * 2;
      }
    }
    res += c;
  }
  return res;
}

std::pair<ull, bool> select(ull player, ull opponent, std::mt19937 &mt) {
  ull mobility_bits = mobility(player, opponent);
  bool pass = false;
  if (mobility_bits == 0) {
    pass = true;
    mobility_bits = mobility(opponent, player);
  }
  const int count = __builtin_popcountll(mobility_bits);
  if (count == 0) return std::make_pair(0ull, true);
  std::uniform_int_distribution<int> dis(0, count-1);
  const int index = dis(mt);
  for (int i = 0; mobility_bits; mobility_bits &= mobility_bits - 1, ++i) {
    if (i == index) {
      return std::make_pair(mobility_bits & -mobility_bits, pass);
    }
  }
  return std::make_pair(0ull, true);
}

bool make_prob_impl(std::size_t depth, std::mt19937 &mt) {
  ull player = UINT64_C(0x0000000810000000);
  ull opponent = UINT64_C(0x0000001008000000);
  for (std::size_t i = 0; i < depth; ++i) {
    ull pos_bit;
    bool pass;
    std::tie(pos_bit, pass) = select(player, opponent, mt);
    if (pos_bit == 0) return false;
    int pos = __builtin_popcountll(pos_bit - 1);
    if (!pass) {
      ull flip_bits = flip(player, opponent, pos);
      ull next_player = opponent ^ flip_bits;
      ull next_opponent = (player ^ flip_bits) | pos_bit;
      player = next_player;
      opponent = next_opponent;
    } else {
      ull flip_bits = flip(opponent, player, pos);
      ull next_player = player ^ flip_bits;
      ull next_opponent = (opponent ^ flip_bits) | pos_bit;
      player = next_player;
      opponent = next_opponent;
    }
  }
  std::cout << to_base81(player, opponent) << std::endl;
  return true;
}

void make_prob(std::size_t depth, std::size_t count) {
  std::random_device rd;
  std::mt19937 mt(rd());
  for (std::size_t i = 0; i < count; ) {
    if (make_prob_impl(depth, mt)) ++i;
  }
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: " << argv[0] << " YBWC_DEPTH" << std::endl;
  }
  char *ptr = nullptr;
  size_t max_depth = std::strtoul(argv[1], &ptr, 10);
  if (ptr == argv[1]) {
    solve_parallel(argv[1], argv[2]);
    return 0;
  }
  if (argc >= 3) {
    std::size_t count = std::strtoul(argv[2], nullptr, 10);
    make_prob(max_depth, count);
    return 0;
  }
  ull black = UINT64_C(0x0000000810000000);
  ull white = UINT64_C(0x0000001008000000);
  Table2 table;
  Evaluator evaluator("subboard6x6.txt", "value6x6/value16_b");
  Table table_cache = init_table();
  while (true) {
    std::cout << "collect tasks..." << std::endl;
    std::vector<AlphaBetaProblem> tasks;
    constexpr float INF = std::numeric_limits<float>::infinity();
    float score = expand_ybwc(black, white, -INF, INF, table, evaluator, max_depth, tasks);
    std::cout << "num = " << tasks.size() << std::endl;
    if (tasks.empty()) {
      std::cout << "Score: " << score << std::endl;
      break;
    }
    BatchedTask batched_task;
    init_batch(batched_task, tasks.size(), 32 - max_depth, table_cache);
    memcpy(batched_task.abp, tasks.data(), sizeof(AlphaBetaProblem) * tasks.size());
    launch_batch(batched_task);
    while (true) {
      if (is_ready_batch(batched_task)) break;
    }
    for (std::size_t i = 0; i < tasks.size(); ++i) {
      table[std::make_pair(tasks[i].player, tasks[i].opponent)] = batched_task.result[i];
    }
    destroy_batch(batched_task);
  }
  destroy_table(table_cache);
  return 0;
}
