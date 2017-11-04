#pragma once
#include <algorithm>
#include <tuple>
#include <unordered_map>
#include <vector>
#include <boost/optional.hpp>
#include "board.cuh"
#include "solver.cuh"

constexpr int gpu_depth = 13;

class CPUNode {
 public:
  CPUNode(ull player, ull opponent, int alpha = -64, int beta = 64, bool passed_prev = false) 
    : player(player), opponent(opponent), result(-64), alpha(alpha), beta(beta), passed_prev(passed_prev), pass(false), next_flips(), index(0) {
    ull mobility_bits = mobility(player, opponent);
    if (mobility_bits == 0) {
      pass = true;
      if (!passed_prev) {
        next_flips.emplace_back(0, 0);
      }
      return;
    }
    for (; mobility_bits; mobility_bits &= mobility_bits - 1) {
      ull pos_bit = mobility_bits & -mobility_bits;
      int pos = __builtin_popcountll(pos_bit - 1);
      ull flip_bits = flip(player, opponent, pos);
      next_flips.emplace_back(pos_bit, flip_bits);
    }
    std::sort(std::begin(next_flips), std::end(next_flips),
        [](const auto &lhs, const auto &rhs) {
          return __builtin_popcountll(std::get<1>(lhs)) < __builtin_popcountll(std::get<1>(rhs));
        });
  }
  ull player_pos() const { return player; }
  ull opponent_pos() const { return opponent; }
  int get_result() const { return result; }
  int get_alpha() const { return alpha; }
  int get_beta() const { return beta; }
  bool is_pass() const { return pass; }
  bool is_gameover() const { return pass && passed_prev; }
  void update(int result_child) {
    result = std::max(result, result_child);
    alpha = std::max(alpha, result);
  }
  std::tuple<ull, ull> next_flip() {
    return next_flips[index++];
  }
  ull is_finished() const { return index == next_flips.size(); }
 private:
  ull player;
  ull opponent;
  int result;
  int alpha;
  int beta;
  bool passed_prev;
  bool pass;
  std::vector<std::tuple<ull, ull>> next_flips;
  size_t index;
};

struct Entry {
  Entry() = default;
  Entry(int upper, int lower)
    : upper(upper), lower(lower) {}
  Entry(const Entry &) = default;
  Entry &operator=(const Entry &) = default;
  int upper;
  int lower;
};

namespace std {

template<>
class hash<tuple<ull, ull>> {
 public:
  size_t operator()(const tuple<ull, ull> &t) const {
    hash<ull> h;
    return h(h(std::get<0>(t)) + 17 * h(std::get<1>(t)));
  }
};

}

class Table {
  static constexpr int value_max = 64;
 public:
  void update(ull player, ull opponent, int value, int alpha, int beta) {
    auto board = std::make_tuple(player, opponent);
    if (value > alpha && value < beta) {
      map[board] = Entry(value, value);
    } else {
      auto itr = map.find(std::make_tuple(player, opponent));
      if (value <= alpha) {
        if (itr == std::end(map)) {
          map[board] = Entry(value, -value_max);
        } else {
          map[board].upper = std::min(map[board].upper, value);
        }
      } else {
        if (itr == std::end(map)) {
          map[board] = Entry(value_max, value);
        } else {
          map[board].lower = std::max(map[board].lower, value);
        }
      }
    }
  }
  boost::optional<Entry> find(ull player, ull opponent) {
    auto itr = map.find(std::make_tuple(player, opponent));
    if (itr == std::end(map)) {
      return boost::none;
    } else {
      return itr->second;
    }
  }
 private:
  std::unordered_map<std::tuple<ull, ull>, Entry> map;
};

class CPUStack {
 public:
  CPUStack(const CPUNode &node, Table &table) 
    : stack(1, node), table(table) {}
  bool step() {
    CPUNode &top = stack.back();
    if (top.is_finished()) {
      if (stack.size() == 1) {
        return true;
      } else {
        CPUNode &top2 = stack[stack.size() - 2];
        if (top.is_gameover()) {
          top.update(final_score(top.player_pos(), top.opponent_pos()));
        } else {
          table.update(top.player_pos(), top.opponent_pos(), top.get_alpha(), -top2.get_beta(), -top2.get_alpha());
        }
        top2.update(-top.get_result());
        stack.pop_back();
      }
    } else if (top.get_alpha() >= top.get_beta()) {
      if (stack.size() == 1) {
        return true;
      } else {
        CPUNode &top2 = stack[stack.size() - 2];
        table.update(top.player_pos(), top.opponent_pos(), top.get_alpha(), -top2.get_beta(), -top2.get_alpha());
        top2.update(-top.get_result());
        stack.pop_back();
      }
    } else {
      ull pos_bit;
      ull flip_bits;
      std::tie(pos_bit, flip_bits) = top.next_flip();
      ull next_player = top.opponent_pos() ^ flip_bits;
      ull next_opponent = (top.player_pos() ^ flip_bits) | pos_bit;
      if (auto entry_opt = table.find(next_player, next_opponent)) {
        stack.emplace_back(next_player, next_opponent,
            std::max(-top.get_beta(), entry_opt->lower),
            std::min(-top.get_alpha(), entry_opt->upper), top.is_pass());
      } else {
        stack.emplace_back(next_player, next_opponent, -top.get_beta(), -top.get_alpha(), top.is_pass());
      }
    }
    return false;
  }
  bool run() {
    while (true) {
      if (step()) return true;
      if (64 - stones_count(stack.back().player_pos(), stack.back().opponent_pos()) == gpu_depth) return false;
    }
  }
  void update(int score) {
    assert(stack.size() >= 2);
    CPUNode &top = stack.back();
    CPUNode &top2 = stack[stack.size() - 2];
    top.update(score);
    top2.update(-top.get_result());
    stack.pop_back();
  }
  CPUNode &get_top() { return stack.back(); }
 private:
  std::vector<CPUNode> stack;
  Table &table;
};
