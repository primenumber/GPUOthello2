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

template <NodeType type>
float expand_ybwc_impl(const ull player, const ull opponent,
    float alpha, const float beta,
    Table2 &table, const Evaluator &evaluator, const int max_depth,
    std::vector<AlphaBetaProblem> &tasks, bool passed_prev = false) {
  if (stones_count(player, opponent)-4 == max_depth) {
    auto itr = table.find(std::make_pair(player, opponent));
    if (itr == std::end(table)) {
      tasks.emplace_back(player, opponent, -64, 64);
      return evaluator.eval(player, opponent);
    } else {
      return itr->second;
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
  for (const auto &child : children) {
    float child_val;
    if (first) {
      child_val = expand_ybwc_impl<first_child(type)>(child.player, child.opponent, -beta, -alpha, table, evaluator, max_depth, tasks);
      first = false;
    } else {
      child_val = expand_ybwc_impl<other_child(type)>(child.player, child.opponent, -beta, -alpha, table, evaluator, max_depth, tasks);
    }
    result = std::max(result, child_val);
    alpha = std::max(alpha, result);
    if (alpha >= beta) return alpha;
  }
  if (first) {
    if (passed_prev) {
      return final_score(player, opponent);
    } else {
      return expand_ybwc_impl<first_child(type)>(opponent, player, -beta, -alpha, table, evaluator, max_depth, tasks, true);
    }
  }
  return result;
}

float expand_ybwc(const ull player, const ull opponent,
    float alpha, const float beta,
    Table2 &table, const Evaluator &evaluator, const int max_depth,
    std::vector<AlphaBetaProblem> &tasks, bool passed_prev) {
  return expand_ybwc_impl<NodeType::PV>(player, opponent, alpha, beta,
      table, evaluator, max_depth, tasks, passed_prev);
}
