#include "expand.cuh"
#include <algorithm>
#include <limits>
#include "board.cuh"

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

struct BoardWithValue {
  BoardWithValue(ull player, ull opponent, int score)
    : player(player), opponent(opponent), score(score) {}
  ull player, opponent;
  int score;
};

bool operator<(const BoardWithValue& lhs, const BoardWithValue &rhs) {
  return lhs.score < rhs.score;
}

struct Result {
  Result() = default;
  Result(float score, bool reliable)
    : score(score), reliable(reliable) {}
  float score;
  bool reliable;
};

template <NodeType type>
Result expand_ybwc_impl(const ull player, const ull opponent,
    float alpha, const float beta,
    Table2 &table, const Evaluator &evaluator, const int max_depth,
    std::unordered_map<int, std::vector<AlphaBetaProblem>> &tasks,
    int level, bool passed_prev = false) {
  if (stones_count(player, opponent)-4 == max_depth) {
    auto itr = table.find(std::make_pair(player, opponent));
    if (itr == std::end(table)) {
      tasks[level].emplace_back(player, opponent, -64, 64);
      return Result(evaluator.eval(player, opponent), false);
    } else {
      return Result(itr->second, true);
    }
  }
  std::vector<BoardWithValue> children;
  for (ull bits = mobility(player, opponent); bits; bits &= bits-1) {
    const ull pos_bit = bits & -bits;
    const int pos = __builtin_popcountll(pos_bit - 1);
    const ull flip_bits = flip(player, opponent, pos);
    const ull next_player = opponent ^ flip_bits;
    const ull next_opponent = (player ^ flip_bits) | pos_bit;
    const float value = evaluator.eval(next_player, next_opponent);
    children.emplace_back(next_player, next_opponent, value);
  }
  std::sort(std::begin(children), std::end(children));
  bool first = true;
  float result = -std::numeric_limits<float>::infinity();
  bool reliable = true;
  for (const auto &child : children) {
    Result child_res;
    if (first) {
      child_res = expand_ybwc_impl<first_child(type)>(child.player, child.opponent, -beta, -alpha, table, evaluator, max_depth, tasks, level);
      first = false;
      if (type == NodeType::PV || type == NodeType::Cut) {
        ++level;
      }
    } else {
      child_res = expand_ybwc_impl<other_child(type)>(child.player, child.opponent, -beta, -alpha, table, evaluator, max_depth, tasks, level);
    }
    result = std::max(result, -child_res.score);
    reliable = reliable && child_res.reliable;
    alpha = std::max(alpha, result);
    if (alpha >= beta) return Result(alpha, reliable);
  }
  if (first) {
    if (passed_prev) {
      return Result(final_score(player, opponent), true);
    } else {
      Result res = expand_ybwc_impl<first_child(type)>(opponent, player, -beta, -alpha, table, evaluator, max_depth, tasks, level, true);
      return Result(-res.score, res.reliable);
    }
  }
  return Result(result, reliable);
}

float expand_ybwc(const ull player, const ull opponent,
    float alpha, const float beta,
    Table2 &table, const Evaluator &evaluator, const int max_depth,
    std::unordered_map<int, std::vector<AlphaBetaProblem>> &tasks,
    bool passed_prev) {
  return expand_ybwc_impl<NodeType::PV>(player, opponent, alpha, beta,
      table, evaluator, max_depth, tasks, 0, passed_prev).score;
}
