#include "solver.cuh"
#include <cstdlib>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include "board.cuh"

// parameters
constexpr int nodesPerBlock = 64;
constexpr int lower_stack_depth = 9;
constexpr int chunk_size = 2048;

class MobilityGenerator {
 public:
  __device__ MobilityGenerator() {}
  __host__ __device__ MobilityGenerator(ull player, ull opponent)
    : x(~opponent), y(~player) {}
  MobilityGenerator(const MobilityGenerator &) = default;
  MobilityGenerator& operator=(const MobilityGenerator &) = default;
  __host__ __device__ ull player_pos() const {
    return x & ~y;
  }
  __host__ __device__ ull opponent_pos() const {
    return ~x & y;
  }
  __host__ __device__ ull empty_pos() const {
    return ~(x ^ y);
  }
  __host__ __device__ ull next_bit() {
    ull p = not_checked_yet();
    ull bit = p & -p;
    reset(bit);
    return bit;
  }
  __host__ __device__ bool completed() const {
    return not_checked_yet() == 0;
  }
  __host__ __device__ MobilityGenerator move(ull flip, ull bit) const {
    ull p = player_pos();
    ull o = opponent_pos();
    return MobilityGenerator(o ^ flip, (p ^ flip) | bit);
  }
  __host__ __device__ MobilityGenerator pass() const {
    ull p = player_pos();
    ull o = opponent_pos();
    return MobilityGenerator(o , p);
  }
  __host__ __device__ int score() const {
    return final_score(player_pos(), opponent_pos());
  }
 private:
  __host__ __device__ ull not_checked_yet() const {
    return x & y;
  }
  __host__ __device__ void reset(ull bit) {
    x ^= bit;
    y ^= bit;
  }
  ull x, y;
};

struct Node {
  MobilityGenerator mg;
  char alpha;
  char beta;
  bool not_pass;
  bool passed_prev;
  __device__ Node() {}
  __host__ __device__ Node(const MobilityGenerator &mg, int alpha, int beta, bool passed_prev = false)
    : mg(mg), alpha(alpha), beta(beta), not_pass(false), passed_prev(passed_prev) {}
  __host__ __device__ Node(const MobilityGenerator &mg)
    : Node(mg, -64, 64) {}
  Node(const Node &) = default;
  Node& operator=(const Node &) = default;
};

__shared__ Node nodes_stack[nodesPerBlock * (lower_stack_depth + 1)];

__device__ Node& get_node(int stack_index, size_t upper_stack_size) {
  return nodes_stack[threadIdx.x + (stack_index - upper_stack_size) * blockDim.x];
}

__device__ Node& get_next_node(int stack_index, size_t upper_stack_size) {
  return nodes_stack[threadIdx.x + (stack_index - upper_stack_size + 1) * blockDim.x];
}

__device__ Node& get_parent_node(int stack_index, size_t upper_stack_size) {
  return nodes_stack[threadIdx.x + (stack_index - upper_stack_size - 1) * blockDim.x];
}

__device__ void commit_lower_impl(int& stack_index, const size_t upper_stack_size) {
  Node& node = get_node(stack_index, upper_stack_size);
  Node& parent = get_parent_node(stack_index, upper_stack_size);
  if (node.passed_prev) {
    parent.alpha = max(node.alpha, parent.alpha);
  } else {
    parent.alpha = max(-node.alpha, parent.alpha);
  }
  --stack_index;
}

__device__ void pass(int stack_index, const size_t upper_stack_size) {
  Node& node = get_node(stack_index, upper_stack_size);
  node.mg = node.mg.pass();
  int tmp = node.alpha;
  node.alpha = -node.beta;
  node.beta = -tmp;
  node.passed_prev = true;
}

class UpperNode {
 public:
  static constexpr int max_mobility_count = 46;
  __device__ UpperNode(ull player, ull opponent, char alpha, char beta, bool pass = false)
      : player(player), opponent(opponent), possize(0), index(0),
      alpha(alpha), beta(beta), prev_passed(pass) {
    MobilityGenerator mg(player, opponent);
    char cntary[max_mobility_count];
    while(!mg.completed()) {
      ull next_bit = mg.next_bit();
      int pos = __popcll(next_bit - 1);
      ull flip_bits = flip(player, opponent, pos);
      if (flip_bits) {
        cntary[possize] = mobility_count(opponent ^ flip_bits, (player ^ flip_bits) | next_bit);
        posary[possize++] = pos;
      }
    }
    thrust::sort_by_key(thrust::seq, cntary, cntary + possize, posary);
  }
  UpperNode& operator=(const UpperNode &) = default;
  __device__ bool completed() const {
    return index == possize;
  }
  __device__ int pop() {
    return posary[index++];
  }
  __device__ int size() const {
    return possize;
  }
  __device__ ull player_pos() const {
    return player;
  }
  __device__ ull opponent_pos() const {
    return opponent;
  }
  __device__ bool passed() const {
    return prev_passed;
  }
  __device__ int score() const {
    return final_score(player, opponent);
  }
  __device__ UpperNode move(ull bits, ull pos_bit, Table table) const {
    ull next_player = opponent ^ bits;
    ull next_opponent = (player ^ bits) | pos_bit;
    Entry entry = table.find(next_player, next_opponent);
    if (entry.enable) {
      char next_alpha = max(-beta, entry.lower);
      char next_beta = min(-alpha, entry.upper);
      return UpperNode(next_player, next_opponent, next_alpha, next_beta);
    } else {
      return UpperNode(next_player, next_opponent, -beta, -alpha);
    }
  }
  __device__ UpperNode pass(Table table) const {
    Entry entry = table.find(opponent, player);
    if (entry.enable) {
      char next_alpha = max(-beta, entry.lower);
      char next_beta = min(-alpha, entry.upper);
      return UpperNode(opponent, player, next_alpha, next_beta, true);
    } else {
      return UpperNode(opponent, player, -beta, -alpha, true);
    }
  }
  char alpha;
  char beta;
 private:
  ull player, opponent;
  char posary[max_mobility_count];
  char possize;
  char index;
  bool prev_passed;
};

__device__ void pass_upper(UpperNode * const upper_stack, const int stack_index, Table table) {
  UpperNode& node = upper_stack[stack_index];
  node = node.pass(table);
}

__shared__ unsigned int index_shared;

__device__ bool next_game(
    const AlphaBetaProblem * const abp, int * const result, UpperNode * const upper_stack,
    const size_t count, size_t &index) {
  UpperNode &node = upper_stack[0];
  result[index] = node.alpha;
  index = atomicAdd(&index_shared, gridDim.x);
  if (index < count) {
    upper_stack[0] = UpperNode(abp[index].player, abp[index].opponent, abp[index].alpha, abp[index].beta);
  }
  return index < count;
}

__device__ void commit_upper(UpperNode * const upper_stack, int &stack_index, Table table) {
  UpperNode &parent = upper_stack[stack_index-1];
  UpperNode &node = upper_stack[stack_index];
  table.update(node.player_pos(), node.opponent_pos(), -parent.beta, -parent.alpha, node.alpha);
  parent.alpha = max(parent.alpha, node.passed() ? node.alpha : -node.alpha);
  stack_index--;
}

__device__ bool commit_or_next(
    const AlphaBetaProblem * const abp, int * const result, UpperNode * const upper_stack,
    const size_t count, size_t &index, int &stack_index, Table table) {
  if (stack_index == 0) {
    if (!next_game(abp, result, upper_stack, count, index))
      return true;
  } else {
    commit_upper(upper_stack, stack_index, table);
  }
  return false;
}

__device__ void commit_to_upper(UpperNode * const upper_stack, int &stack_index, const size_t upper_stack_size) {
  UpperNode &parent = upper_stack[stack_index-1];
  Node &node = get_node(stack_index, upper_stack_size);
  parent.alpha = max(parent.alpha, node.passed_prev ? node.alpha : -node.alpha);
  --stack_index;
}

__device__ void commit_lower(UpperNode * const upper_stack, int& stack_index, const size_t upper_stack_size) {
  if (stack_index == upper_stack_size) {
    commit_to_upper(upper_stack, stack_index, upper_stack_size);
  } else {
    commit_lower_impl(stack_index, upper_stack_size);
  }
}

__device__ bool solve_all_upper(
    const AlphaBetaProblem * const abp, int * const result, UpperNode * const upper_stack, const size_t count, const size_t upper_stack_size,
    size_t &index, int &stack_index, Table table) {
  UpperNode& node = upper_stack[stack_index];
  if (node.completed()) {
    if (node.size() == 0) { // pass
      if (node.passed()) {
        node.alpha = node.score();
        if (commit_or_next(abp, result, upper_stack, count, index, stack_index, table)) return true;
      } else {
        pass_upper(upper_stack, stack_index, table);
      }
    } else { // completed
      if (commit_or_next(abp, result, upper_stack, count, index, stack_index, table)) return true;
    }
  } else if (node.alpha >= node.beta) {
    if (commit_or_next(abp, result, upper_stack, count, index, stack_index, table)) return true;
  } else {
    int pos = node.pop();
    ull flip_bits = flip(node.player_pos(), node.opponent_pos(), pos);
    assert(flip_bits);
    if (stack_index < upper_stack_size - 1) {
      UpperNode& next_node = upper_stack[stack_index+1];
      next_node = node.move(flip_bits, UINT64_C(1) << pos, table);
    } else {
      Node& next_node = get_next_node(stack_index, upper_stack_size);
      next_node = Node(MobilityGenerator(node.opponent_pos() ^ flip_bits, (node.player_pos() ^ flip_bits) | (UINT64_C(1) << pos)), -node.beta, -node.alpha);
    }
    ++stack_index;
  }
  return false;
}

__device__ void solve_all_lower(UpperNode * const upper_stack, const size_t upper_stack_size, int &stack_index) {
  Node& node = get_node(stack_index, upper_stack_size);
  if (node.mg.completed()) {
    if (node.not_pass) {
      commit_lower(upper_stack, stack_index, upper_stack_size);
    } else { // pass
      if (node.passed_prev) { // end game
        node.alpha = node.mg.score();
        commit_lower(upper_stack, stack_index, upper_stack_size);
      } else { // pass
        pass(stack_index, upper_stack_size);
      }
    }
  } else if (node.alpha >= node.beta) { // beta cut
    commit_lower(upper_stack, stack_index, upper_stack_size);
  } else {
    ull next_bit = node.mg.next_bit();
    int pos = __popcll(next_bit - 1);
    ull flip_bits = flip(node.mg.player_pos(), node.mg.opponent_pos(), pos);
    if (flip_bits) { // movable
      node.not_pass = true;
      Node& next_node = get_next_node(stack_index, upper_stack_size);
      next_node = Node(node.mg.move(flip_bits, next_bit), -node.beta, -node.alpha);
      ++stack_index;
    }
  }
}

__device__ void solve_all(const AlphaBetaProblem * const abp, int * const result, UpperNode * const upper_stack,
    const size_t count, const size_t upper_stack_size, size_t &index, Table table) {
  int stack_index = 0;
  while (true) {
    assert(index < count);
    if (stack_index < upper_stack_size) {
      if (solve_all_upper(abp, result, upper_stack, count, upper_stack_size, index, stack_index, table)) return;
    } else {
      solve_all_lower(upper_stack, upper_stack_size, stack_index);
    }
  }
}

__global__ void alpha_beta_kernel(
    const AlphaBetaProblem * const abp, int * const result, UpperNode * const upper_stack,
    size_t count, size_t upper_stack_size, Table table) {
  index_shared = blockIdx.x;
  __syncthreads();
  size_t index = atomicAdd(&index_shared, gridDim.x);
  if (index < count) {
    UpperNode *ustack = upper_stack + index * upper_stack_size;
    const AlphaBetaProblem &problem = abp[index];
    ustack[0] = UpperNode(problem.player, problem.opponent, problem.alpha, problem.beta);
    solve_all(abp, result, ustack, count, upper_stack_size, index, table);
  }
}

void init_batch(BatchedTask &bt, size_t batch_size, size_t max_depth, const Table &table) {
  bt.str = (cudaStream_t*)malloc(sizeof(cudaStream_t));
  cudaStreamCreate(bt.str);
  cudaMallocManaged((void**)&bt.abp, sizeof(AlphaBetaProblem) * batch_size);
  cudaMallocManaged((void**)&bt.result, sizeof(int) * batch_size);
  bt.table = table;
  bt.size = batch_size;
  bt.grid_size = (batch_size + chunk_size - 1) / chunk_size;
  bt.max_depth = max_depth;
  cudaMalloc((void**)&bt.upper_stacks, sizeof(UpperNode) * bt.grid_size * nodesPerBlock * (bt.max_depth - lower_stack_depth));
}

void launch_batch(const BatchedTask &bt) {
  alpha_beta_kernel<<<bt.grid_size, nodesPerBlock, 0, *bt.str>>>(bt.abp, bt.result, bt.upper_stacks, bt.size, bt.max_depth - lower_stack_depth, bt.table);
}

bool is_ready_batch(const BatchedTask &bt) {
  return cudaStreamQuery(*bt.str) == cudaSuccess;
}

void destroy_batch(const BatchedTask &bt) {
  cudaStreamDestroy(*bt.str);
  free(bt.str);
  cudaFree(bt.abp);
  cudaFree(bt.result);
  cudaFree(bt.upper_stacks);
}
