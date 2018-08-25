#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>
#include "types.hpp"
#include "eval.cuh"
#include "solver.cuh"

namespace std {
template<>
struct hash<pair<ull, ull>> {
  size_t operator()(const pair<ull, ull> &p) const {
    return p.first * 17 + p.second;
  }
};
}

// (me, op), (lower, upper)
using Table2 = std::unordered_map<std::pair<ull, ull>, std::pair<int, int>>;

float expand_ybwc(const ull player, const ull opponent,
    float alpha, const float beta,
    Table2 &table, const Evaluator &evaluator, const int max_depth,
    std::vector<AlphaBetaProblem> &tasks, bool passed_prev = false);
