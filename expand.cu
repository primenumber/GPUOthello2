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

std::unordered_map<std::pair<ull, ull>, std::pair<float, float>> cache;

template <NodeType type>
float think(const ull player, const ull opponent,
    float alpha, float beta,
    const Evaluator &evaluator, const int max_depth,
    bool passed_prev = false);

template <NodeType type>
float think_impl(const ull player, const ull opponent,
    float alpha, const float beta,
    const Evaluator &evaluator, const int max_depth,
    bool passed_prev = false) {
  if (stones_count(player, opponent)-4 == max_depth) {
    return evaluator.eval(player, opponent);
  }
  std::vector<BoardWithValue> children;
  for (ull bits = mobility(player, opponent); bits; bits &= bits-1) {
    const ull pos_bit = bits & -bits;
    const int pos = __builtin_popcountll(pos_bit - 1);
    const ull flip_bits = flip(player, opponent, pos);
    const ull next_player = opponent ^ flip_bits;
    const ull next_opponent = (player ^ flip_bits) | pos_bit;
    const float value = mobility_count(next_player, next_opponent);
    children.emplace_back(next_player, next_opponent, value);
  }
  std::sort(std::begin(children), std::end(children));
  bool first = true;
  float result = -std::numeric_limits<float>::infinity();
  for (const auto &child : children) {
    float child_val;
    if (first) {
      child_val = -think<first_child(type)>(child.player, child.opponent, -beta, -alpha, evaluator, max_depth);
      first = false;
    } else {
      child_val = -think<other_child(type)>(child.player, child.opponent, -beta, -alpha, evaluator, max_depth);
    }
    result = std::max(result, child_val);
    alpha = std::max(alpha, result);
    if (alpha >= beta) return alpha;
  }
  if (first) {
    if (passed_prev) {
      return final_score(player, opponent);
    } else {
      return -think<first_child(type)>(opponent, player, -beta, -alpha, evaluator, max_depth, true);
    }
  }
  return result;
}

template <NodeType type>
float think(const ull player, const ull opponent,
    float alpha, float beta,
    const Evaluator &evaluator, const int max_depth,
    bool passed_prev) {
  std::pair<ull, ull> bd(player, opponent);
  auto itr = cache.find(bd);
  if (itr != std::end(cache)) {
    float lower, upper;
    std::tie(lower, upper) = itr->second;
    if (lower == upper) {
      return lower;
    }
    alpha = std::max(alpha, lower);
    beta = std::min(beta, upper);
    if (alpha >= beta) {
      return std::min(alpha, upper);
    }
  }
  float val = think_impl<type>(player, opponent, alpha, beta,
      evaluator, max_depth, passed_prev);
  itr = cache.find(bd);
  if (itr != std::end(cache)) {
    float lower, upper;
    std::tie(lower, upper) = itr->second;
    if (val <= alpha) {
      cache[bd] = std::make_pair(lower, std::min(upper, val));
    } else if (val >= beta) {
      cache[bd] = std::make_pair(std::max(lower, val), upper);
    } else {
      cache[bd] = std::make_pair(val, val);
    }
  } else {
    if (val <= alpha) {
      cache[bd] = std::make_pair(-64.0, val);
    } else if (val >= beta) {
      cache[bd] = std::make_pair(val, 64.0);
    } else {
      cache[bd] = std::make_pair(val, val);
    }
  }
  return val;
}

template <NodeType type>
std::tuple<float, bool> expand_ybwc_top(const ull player, const ull opponent,
    float alpha, float beta,
    Table2 &table, const Evaluator &evaluator, const int max_depth,
    std::vector<AlphaBetaProblem> &tasks, bool passed_prev = false);

template <NodeType type>
std::tuple<float, bool> expand_ybwc_impl(const ull player, const ull opponent,
    float alpha, const float beta,
    Table2 &table, const Evaluator &evaluator, const int max_depth,
    std::vector<AlphaBetaProblem> &tasks, bool passed_prev = false) {
  if (stones_count(player, opponent)-4 == max_depth) {
    auto itr = table.find(std::make_pair(player, opponent));
    if (itr == std::end(table)) {
      tasks.emplace_back(player, opponent,
          std::max(-64.0f, floor(alpha)),
          std::min(64.0f, ceil(beta)));
      return std::make_tuple(evaluator.eval(player, opponent), false);
    } else if (itr->second.first == itr->second.second) {
      return std::make_tuple(itr->second.first, true);
    } else {
      int lower, upper;
      std::tie(lower, upper) = itr->second;
      float new_alpha = std::max(alpha, (float)lower);
      float new_beta = std::min(beta, (float)upper);
      if (new_alpha >= new_beta) {
        return std::make_tuple(std::min(new_alpha, (float)upper), true);
      }
      tasks.emplace_back(player, opponent,
          std::max(-64.0f, floor(new_alpha)),
          std::min(64.0f, ceil(new_beta)));
      return std::make_tuple(evaluator.eval(player, opponent), false);
    }
  }
  std::vector<BoardWithValue> children;
  for (ull bits = mobility(player, opponent); bits; bits &= bits-1) {
    const ull pos_bit = bits & -bits;
    const int pos = __builtin_popcountll(pos_bit - 1);
    const ull flip_bits = flip(player, opponent, pos);
    const ull next_player = opponent ^ flip_bits;
    const ull next_opponent = (player ^ flip_bits) | pos_bit;
    constexpr float INF = std::numeric_limits<float>::infinity();
    const float value = think<NodeType::PV>(next_player, next_opponent, -INF, INF, evaluator, max_depth);
    children.emplace_back(next_player, next_opponent, value);
  }
  std::sort(std::begin(children), std::end(children));
  bool first = true;
  float result = -std::numeric_limits<float>::infinity();
  bool exact = true;
  for (const auto &child : children) {
    std::tuple<float, bool> child_val;
    if (first) {
      child_val = expand_ybwc_top<first_child(type)>(child.player, child.opponent, -beta, -alpha, table, evaluator, max_depth, tasks);
      first = false;
    } else {
      child_val = expand_ybwc_top<other_child(type)>(child.player, child.opponent, -beta, -alpha, table, evaluator, max_depth, tasks);
    }
    result = std::max(result, -std::get<0>(child_val));
    if (!std::get<1>(child_val)) exact = false;
    alpha = std::max(alpha, result);
    if (alpha >= beta) return std::make_tuple(alpha, exact);
  }
  if (first) {
    if (passed_prev) {
      return std::make_tuple(final_score(player, opponent), true);
    } else {
      float val;
      bool exact;
      std::tie(val, exact) = expand_ybwc_top<first_child(type)>(opponent, player, -beta, -alpha, table, evaluator, max_depth, tasks, true);
      return std::make_tuple(-val, exact);
    }
  }
  return std::make_tuple(result, exact);
}

template <NodeType type>
std::tuple<float, bool> expand_ybwc_top(const ull player, const ull opponent,
    float alpha, float beta,
    Table2 &table, const Evaluator &evaluator, const int max_depth,
    std::vector<AlphaBetaProblem> &tasks, bool passed_prev) {
  std::pair<ull, ull> bd(player, opponent);
  auto itr = table.find(bd);
  float val;
  bool exact;
  if (itr == std::end(table)) {
    std::tie(val, exact) = expand_ybwc_impl<type>(player, opponent, alpha, beta,
        table, evaluator, max_depth, tasks, passed_prev);
  } else if (itr->second.first == itr->second.second) {
    return std::make_tuple(itr->second.first, true);
  } else {
    int lower, upper;
    std::tie(lower, upper) = itr->second;
    alpha = std::max(alpha, (float)lower);
    beta = std::min(beta, (float)upper);
    if (alpha >= beta) {
      return std::make_tuple(std::min(alpha, (float)upper), true);
    }
    std::tie(val, exact) = expand_ybwc_impl<type>(player, opponent, alpha, beta,
        table, evaluator, max_depth, tasks, passed_prev);
  }
  if (exact) {
    auto itr = table.find(bd);
    if (itr == std::end(table)) {
      if (val <= alpha) {
        table[bd] = std::make_pair(-64, val);
      } else if (val >= beta) {
        table[bd] = std::make_pair(val, 64);
      } else {
        table[bd] = std::make_pair(val, val);
      }
    } else {
      int lower, upper;
      std::tie(lower, upper) = itr->second;
      if (val <= alpha) {
        table[bd] = std::make_pair(std::max(-64, lower), std::min((int)val, upper));
      } else if (val >= beta) {
        table[bd] = std::make_pair(std::max((int)val, lower), std::min(64, upper));
      } else {
        table[bd] = std::make_pair(std::max((int)val, lower), std::min((int)val, upper));
      }
    }
  }
  return std::make_tuple(val, exact);
}

float expand_ybwc(const ull player, const ull opponent,
    float alpha, const float beta,
    Table2 &table, const Evaluator &evaluator, const int max_depth,
    std::vector<AlphaBetaProblem> &tasks, bool passed_prev) {
  return std::get<0>(expand_ybwc_top<NodeType::PV>(player, opponent, alpha, beta,
      table, evaluator, max_depth, tasks, passed_prev));
}
