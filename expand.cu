#include "expand.cuh"
#include <algorithm>
#include <iostream>
#include <limits>
#include <mutex>
#include <thread>
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

std::unordered_map<std::pair<ull, ull>, std::pair<float, float>> cache1;
std::unordered_map<std::pair<ull, ull>, std::pair<float, float>> cache2;
std::unordered_map<std::pair<ull, ull>, float> cache;

constexpr float INF = std::numeric_limits<float>::infinity();

template <NodeType type>
float think(const ull player, const ull opponent,
    float alpha, const float beta,
    const Evaluator &evaluator, const int max_depth,
    bool passed_prev = false);

template <NodeType type>
float think_impl(const ull player, const ull opponent,
    float alpha, const float beta,
    const Evaluator &evaluator, const int max_depth,
    bool passed_prev = false) {
  auto itr = cache.find(std::make_pair(player, opponent));
  if (itr != std::end(cache)) {
    return itr->second;
  }
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
    const auto itr = cache2.find(std::make_pair(next_player, next_opponent));
    if (itr == std::end(cache2)) {
      const float value = evaluator.eval(next_player, next_opponent) + 64.0;
      children.emplace_back(next_player, next_opponent, value);
    } else {
      const float value = itr->second.second;
      children.emplace_back(next_player, next_opponent, value);
    }
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

std::mutex m;
template <NodeType type>
float think(const ull player, const ull opponent,
    float alpha, const float beta,
    const Evaluator &evaluator, const int max_depth,
    bool passed_prev) {
  if (stones_count(player, opponent)-4 >= max_depth - 8) {
    return think_impl<type>(player, opponent, alpha, beta,
        evaluator, max_depth, passed_prev);
  } else {
    std::unique_lock<std::mutex> lock(m);
    auto itr = cache1.find(std::make_pair(player, opponent));
    if (itr == std::end(cache1)) {
      lock.unlock();
      return think_impl<type>(player, opponent, alpha, beta,
          evaluator, max_depth, passed_prev);
    }
    float lower, upper;
    std::tie(lower, upper) = itr->second;
    lock.unlock();
    float new_alpha = std::max(alpha, lower);
    float new_beta  = std::min(beta, upper);
    float val = think_impl<type>(player, opponent, new_alpha,
        new_beta, evaluator, max_depth, passed_prev);
    std::pair<float, float> range;
    if (val <= new_alpha) {
      range = std::make_pair(-INF, val);
    } else if (val >= new_beta) {
      range = std::make_pair(val, INF);
    } else {
      range = std::make_pair(val, val);
    }
    lock.lock();
    itr = cache1.find(std::make_pair(player, opponent));
    if (itr == std::end(cache1)) {
      cache1[std::make_pair(player, opponent)] = range;
    } else {
      cache1[std::make_pair(player, opponent)] =
        std::make_pair(std::max(itr->second.first, range.first),
            std::min(itr->second.second, range.second));
    }
    return val;
  }
}

template <NodeType type>
float expand_ybwc_impl(const ull player, const ull opponent,
    float alpha, const float beta,
    Table2 &table, const Evaluator &evaluator, const int max_depth,
    const int max_think_depth,
    std::vector<AlphaBetaProblem> &tasks, bool passed_prev = false) {
  if (stones_count(player, opponent)-4 == max_depth) {
    auto itr = table.find(std::make_pair(player, opponent));
    if (itr == std::end(table)) {
      tasks.emplace_back(player, opponent, -64, 64);
      return think<NodeType::PV>(player, opponent, -INF, INF, evaluator, max_depth, max_think_depth);
    } else {
      return itr->second;
    }
  }
  std::vector<BoardWithValue> children;
  std::vector<std::thread> vt;
  for (ull bits = mobility(player, opponent); bits; bits &= bits-1) {
    const ull pos_bit = bits & -bits;
    const int pos = __builtin_popcountll(pos_bit - 1);
    const ull flip_bits = flip(player, opponent, pos);
    const ull next_player = opponent ^ flip_bits;
    const ull next_opponent = (player ^ flip_bits) | pos_bit;
    if (cache.count(std::make_pair(next_player, next_opponent))) {
      std::lock_guard<std::mutex> lock(m);
      children.emplace_back(next_player, next_opponent, cache[std::make_pair(next_player, next_opponent)]);
    } else if (stones_count(player, opponent)-4 < max_depth - 4) {
      vt.emplace_back([=, &evaluator, &children]{
        const float value = think<NodeType::PV>(next_player, next_opponent, -INF, INF, evaluator, max_think_depth);
        children.emplace_back(next_player, next_opponent, value);
      });
    } else {
      const float value = think<NodeType::PV>(next_player, next_opponent, -INF, INF, evaluator, max_think_depth);
      children.emplace_back(next_player, next_opponent, value);
    }
  }
  for (auto &t : vt) {
    t.join();
  }
  std::sort(std::begin(children), std::end(children));
  bool first = true;
  float result = -std::numeric_limits<float>::infinity();
  for (const auto &child : children) {
    float child_val;
    if (first) {
      child_val = -expand_ybwc_impl<first_child(type)>(child.player, child.opponent, -beta, -alpha, table, evaluator, max_depth, max_think_depth, tasks);
      first = false;
    } else {
      child_val = -expand_ybwc_impl<other_child(type)>(child.player, child.opponent, -beta, -alpha, table, evaluator, max_depth, max_think_depth, tasks);
    }
    result = std::max(result, child_val);
    alpha = std::max(alpha, result);
    if (alpha >= beta) return alpha;
  }
  if (first) {
    if (passed_prev) {
      return final_score(player, opponent);
    } else {
      return -expand_ybwc_impl<first_child(type)>(opponent, player, -beta, -alpha, table, evaluator, max_depth, max_think_depth, tasks, true);
    }
  }
  return result;
}

float expand_ybwc(const ull player, const ull opponent,
    float alpha, const float beta, Table2 &table,
    const std::vector<Evaluator> &evaluators, const int max_depth,
    std::vector<AlphaBetaProblem> &tasks, bool passed_prev) {
  for (int i = 0; i < evaluators.size(); ++i) {
    cache1.clear();
    std::cerr << i << std::endl;
    think<NodeType::PV>(player, opponent, alpha, beta,
      evaluators[i], max_depth+i, passed_prev);
    swap(cache1, cache2);
  }
  std::cerr << "expand" << std::endl;
  return expand_ybwc_impl<NodeType::PV>(player, opponent, alpha, beta,
      table, evaluators.back(), max_depth, max_depth + evaluators.size() - 1, tasks, passed_prev);
}
