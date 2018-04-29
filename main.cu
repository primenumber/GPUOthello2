#include "solver.cuh"
#include <algorithm>
#include <iostream>
#include <limits>
#include <string>
#include <vector>
#include <tuple>
#include <unordered_map>
#include <boost/timer/timer.hpp>
#include "to_board.hpp"
#include "board.cuh"
#include "eval.cuh"

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

namespace std {
template<>
struct hash<pair<ull, ull>> {
  size_t operator()(const pair<ull, ull> &p) const {
    return p.first * 17 + p.second;
  }
};
}

using Table2 = std::unordered_map<std::pair<ull, ull>, int>;

enum class NodeType {
  PV,
  Cut,
  All,
  Unknown
};

constexpr NodeType first_child(const NodeType type) {
  switch (type) {
    case NodeType::PV: return NodeType::PV;
    case NodeType::Cut: return NodeType::All;
    case NodeType::All: return NodeType::Cut;
    default: return NodeType::Unknown;
  }
}

constexpr NodeType other_child(const NodeType type) {
  switch (type) {
    case NodeType::PV: return NodeType::Cut;
    case NodeType::Cut: return NodeType::All;
    case NodeType::All: return NodeType::Cut;
    default: return NodeType::Unknown;
  }
}

template <NodeType type>
float expand_ybwc(const ull player, const ull opponent,
    float alpha, const float beta,
    Table2 &table, const Evaluator &evaluator, const int max_depth,
    std::vector<AlphaBetaProblem> &tasks, bool passed_prev = false) {
  if (stones_count(player, opponent)-4 == max_depth) {
    auto itr = table.find(std::make_pair(player, opponent));
    if (itr == std::end(table)) {
      tasks.emplace_back(player, opponent, alpha, beta);
      return evaluator.eval(player, opponent);
    } else {
      return itr->second;
    }
  }
  using T = std::tuple<float, ull, ull>;
  std::vector<T> children;
  for (ull bits = mobility(player, opponent); bits; bits &= bits-1) {
    const ull pos_bit = bits & -bits;
    const int pos = __builtin_popcountll(pos_bit - 1);
    const ull flip_bits = flip(player, opponent, pos);
    const ull next_player = opponent ^ flip_bits;
    const ull next_opponent = (player ^ flip_bits) | pos_bit;
    const float value = evaluator.eval(next_player, next_opponent);
    children.emplace_back(value, next_player, next_opponent);
  }
  std::sort(std::begin(children), std::end(children),
      [] (const T& lhs, const T& rhs) {
        return std::get<0>(lhs) < std::get<0>(rhs);
      });
  bool first = true;
  float result = -std::numeric_limits<float>::infinity();
  for (const auto &child : children) {
    float point;
    ull next_player, next_opponent;
    std::tie(point, next_player, next_opponent) = child;
    float child_val;
    if (first) {
      child_val = expand_ybwc<first_child(type)>(next_player, next_opponent, -beta, -alpha, table, evaluator, max_depth, tasks);
      first = false;
    } else {
      child_val = expand_ybwc<other_child(type)>(next_player, next_opponent, -beta, -alpha, table, evaluator, max_depth, tasks);
    }
    result = std::max(result, child_val);
    alpha = std::max(alpha, result);
    if (alpha >= beta) return alpha;
  }
  if (first) {
    if (passed_prev) {
      return final_score(player, opponent);
    } else {
      return expand_ybwc<first_child(type)>(opponent, player, -beta, -alpha, table, evaluator, max_depth, tasks, true);
    }
  }
  return result;
}

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
  expand_ybwc<NodeType::PV>(black, white, -INF, INF, table, evaluator, max_depth, tasks);
  std::cout << tasks.size() << std::endl;
  //constexpr size_t batch_size = 8192;
  //size_t batch_count = (n + batch_size - 1) / batch_size;
  //std::vector<Batch> vb(batch_count);
  //Table table;
  //constexpr size_t table_size = 50000001;
  //cudaMallocManaged((void**)&table.entries, sizeof(Entry) * table_size);
  //cudaMallocManaged((void**)&table.mutex, sizeof(int) * table_size);
  //cudaMallocManaged((void**)&table.update_count, sizeof(ull));
  //cudaMallocManaged((void**)&table.hit_count, sizeof(ull));
  //cudaMallocManaged((void**)&table.lookup_count, sizeof(ull));
  //table.size = table_size;
  //*table.update_count = 0;
  //*table.hit_count = 0;
  //*table.lookup_count = 0;
  //memset(table.entries, 0, sizeof(Entry) * table_size);
  //memset(table.mutex, 0, sizeof(int) * table_size);
  //for (size_t i = 0; i < batch_count; ++i) {
  //  int size = min(batch_size, n - i*batch_size);
  //  init_batch(vb[i].bt, size, max_depth, table);
  //  for (int j = 0; j < vb[i].bt.size; ++j) {
  //    ull player, opponent;
  //    std::tie(player, opponent) = toBoard(vboard[i*batch_size+j].c_str());
  //    vb[i].bt.abp[j] = AlphaBetaProblem(player, opponent);
  //    vb[i].vstr.push_back(vboard[i*batch_size+j]);
  //  }
  //}
  //boost::timer::cpu_timer timer;
  //for (const auto &b : vb) {
  //  launch_batch(b.bt);
  //}
  //while (true) {
  //  bool finished = true;
  //  for (const auto &b : vb) {
  //    if (!is_ready_batch(b.bt)) finished = false;
  //  }
  //  if (finished) break;
  //}
  //fprintf(stderr, "%s, elapsed: %.6fs, table update count: %llu, table hit: %llu, table find: %llu\n",
  //    cudaGetErrorString(cudaGetLastError()), timer.elapsed().wall/1000000000.0,
  //    *table.update_count, *table.hit_count, *table.lookup_count);
  //ull total = 0;
  //for (const auto &b : vb) {
  //  total += *b.bt.total;
  //  for (int j = 0; j < b.bt.size; ++j) {
  //    fprintf(fp_out, "%s %d\n", b.vstr[j].c_str(), b.bt.result[j]);
  //  }
  //  destroy_batch(b.bt);
  //}
  //fprintf(stderr, "total nodes: %llu\n", total);
  //cudaFree(table.entries);
  //cudaFree(table.mutex);
  //cudaFree(table.update_count);
  //cudaFree(table.hit_count);
  //cudaFree(table.lookup_count);
}
