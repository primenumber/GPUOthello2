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

using Table2 = std::unordered_map<std::pair<ull, ull>, int>;

float expand_ybwc(const ull player, const ull opponent,
    float alpha, const float beta,
    Table2 &table, const Evaluator &evaluator, const int max_depth,
    std::unordered_map<int, std::vector<AlphaBetaProblem>> &tasks,
    bool passed_prev = false);
