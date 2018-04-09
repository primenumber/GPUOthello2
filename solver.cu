#include "solver.cuh"
#include <cstdlib>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include "board.cuh"

// parameters
constexpr int lower_stack_depth = 9;

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

extern __shared__ Node nodes_stack[];

struct UpperNode;

struct Solver {
  int stack_index;
  UpperNode * const upper_stack;
  const size_t upper_stack_size;
  const AlphaBetaProblem * const abp;
  int *result;
  size_t count;
  size_t index;
  Table table;

  __device__ Node& get_node();
  __device__ Node& get_next_node();
  __device__ Node& get_parent_node();
  __device__ void commit_lower_impl();
  __device__ void pass();
  __device__ void pass_upper();
  __device__ bool next_game();
  __device__ void commit_upper();
  __device__ bool commit_or_next();
  __device__ void commit_to_upper();
  __device__ void commit_lower();
  __device__ bool solve_all_upper();
  __device__ void solve_all_lower();
  __device__ int solve_all();
};

__device__ Node& Solver::get_node() {
  return nodes_stack[threadIdx.x + (stack_index - upper_stack_size) * blockDim.x];
}

__device__ Node& Solver::get_next_node() {
  return nodes_stack[threadIdx.x + (stack_index - upper_stack_size + 1) * blockDim.x];
}

__device__ Node& Solver::get_parent_node() {
  return nodes_stack[threadIdx.x + (stack_index - upper_stack_size - 1) * blockDim.x];
}

__device__ void Solver::commit_lower_impl() {
  Node& node = get_node();
  Node& parent = get_parent_node();
  if (node.passed_prev) {
    parent.alpha = max(node.alpha, parent.alpha);
  } else {
    parent.alpha = max(-node.alpha, parent.alpha);
  }
  --stack_index;
}

__device__ void Solver::pass() {
  Node& node = get_node();
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

__device__ void Solver::pass_upper() {
  UpperNode& node = upper_stack[stack_index];
  node = node.pass(table);
}

__shared__ unsigned int index_shared;

__device__ bool Solver::next_game() {
  UpperNode &node = upper_stack[0];
  result[index] = node.alpha;
  index = atomicAdd(&index_shared, gridDim.x);
  if (index < count) {
    upper_stack[0] = UpperNode(abp[index].player, abp[index].opponent, abp[index].alpha, abp[index].beta);
  }
  return index < count;
}

__device__ void Solver::commit_upper() {
  UpperNode &parent = upper_stack[stack_index-1];
  UpperNode &node = upper_stack[stack_index];
  table.update(node.player_pos(), node.opponent_pos(), -parent.beta, -parent.alpha, node.alpha);
  parent.alpha = max(parent.alpha, node.passed() ? node.alpha : -node.alpha);
  stack_index--;
}

__device__ bool Solver::commit_or_next() {
  if (stack_index == 0) {
    if (!next_game())
      return true;
  } else {
    commit_upper();
  }
  return false;
}

__device__ void Solver::commit_to_upper() {
  UpperNode &parent = upper_stack[stack_index-1];
  Node &node = get_node();
  parent.alpha = max(parent.alpha, node.passed_prev ? node.alpha : -node.alpha);
  --stack_index;
}

__device__ void Solver::commit_lower() {
  if (stack_index == upper_stack_size) {
    commit_to_upper();
  } else {
    commit_lower_impl();
  }
}

__device__ bool Solver::solve_all_upper() {
  UpperNode& node = upper_stack[stack_index];
  if (node.completed()) {
    if (node.size() == 0) { // pass
      if (node.passed()) {
        node.alpha = node.score();
        if (commit_or_next()) return true;
      } else {
        pass_upper();
      }
    } else { // completed
      if (commit_or_next()) return true;
    }
  } else if (node.alpha >= node.beta) {
    if (commit_or_next()) return true;
  } else {
    int pos = node.pop();
    ull flip_bits = flip(node.player_pos(), node.opponent_pos(), pos);
    assert(flip_bits);
    if (stack_index < upper_stack_size - 1) {
      UpperNode& next_node = upper_stack[stack_index+1];
      next_node = node.move(flip_bits, UINT64_C(1) << pos, table);
    } else {
      Node& next_node = get_next_node();
      next_node = Node(MobilityGenerator(node.opponent_pos() ^ flip_bits, (node.player_pos() ^ flip_bits) | (UINT64_C(1) << pos)), -node.beta, -node.alpha);
    }
    ++stack_index;
  }
  return false;
}

__device__ void Solver::solve_all_lower() {
  Node& node = get_node();
  if (node.mg.completed()) {
    if (node.not_pass) {
      commit_lower();
    } else { // pass
      if (node.passed_prev) { // end game
        node.alpha = node.mg.score();
        commit_lower();
      } else { // pass
        pass();
      }
    }
  } else if (node.alpha >= node.beta) { // beta cut
    commit_lower();
  } else {
    ull next_bit = node.mg.next_bit();
    int pos = __popcll(next_bit - 1);
    ull flip_bits = flip(node.mg.player_pos(), node.mg.opponent_pos(), pos);
    if (flip_bits) { // movable
      node.not_pass = true;
      Node& next_node = get_next_node();
      next_node = Node(node.mg.move(flip_bits, next_bit), -node.beta, -node.alpha);
      ++stack_index;
    }
  }
}

__device__ int Solver::solve_all() {
  ull nodes_count = 0;
  while (true) {
    ++nodes_count;
    assert(index < count);
    if (stack_index < upper_stack_size) {
      if (solve_all_upper()) return nodes_count;
    } else {
      solve_all_lower();
    }
  }
}

__global__ void alpha_beta_kernel(
    const AlphaBetaProblem * const abp, int * const result, UpperNode * const upper_stack,
    size_t count, size_t upper_stack_size, Table table, ull * const nodes_total) {
  index_shared = blockIdx.x;
  __syncthreads();
  size_t index = atomicAdd(&index_shared, gridDim.x);
  if (index < count) {
    UpperNode *ustack = upper_stack + index * upper_stack_size;
    const AlphaBetaProblem &problem = abp[index];
    Solver solver = {
      0, // stack_index
      ustack, // upper_stack
      upper_stack_size, // upper_stack_size
      abp, // abp
      result, // result
      count, // count
      index, // index
      table // table
    };
    solver.upper_stack[0] = UpperNode(problem.player, problem.opponent, problem.alpha, problem.beta);
    ull nodes_count = solver.solve_all();
    atomicAdd(nodes_total, nodes_count);
  }
}

struct Thinker {
  int stack_index;
  ull leaf_me, leaf_op;
  const size_t stack_size;
  const AlphaBetaProblem * const abp;
  int *result;
  size_t count;
  size_t index;
  Table table;
  Evaluator evaluator;

  __device__ Node& get_node();
  __device__ Node& get_next_node();
  __device__ Node& get_parent_node();
  __device__ void pass();
  __device__ bool next_game();
  __device__ void commit();
  __device__ bool commit_or_next();
  __device__ void commit_from_leaf(int);
  __device__ int think();
};

__device__ Node& Thinker::get_node() {
  return nodes_stack[threadIdx.x + stack_index * blockDim.x];
}

__device__ Node& Thinker::get_next_node() {
  return nodes_stack[threadIdx.x + (stack_index + 1) * blockDim.x];
}

__device__ Node& Thinker::get_parent_node() {
  return nodes_stack[threadIdx.x + (stack_index - 1) * blockDim.x];
}

__device__ void Thinker::pass() {
  Node& node = get_node();
  node.mg = node.mg.pass();
  int tmp = node.alpha;
  node.alpha = -node.beta;
  node.beta = -tmp;
  node.passed_prev = true;
}

__device__ bool Thinker::next_game() {
  Node &node = get_node();
  result[index] = node.alpha;
  index = atomicAdd(&index_shared, gridDim.x);
  if (index < count) {
    node = Node(MobilityGenerator(abp[index].player, abp[index].opponent), abp[index].alpha, abp[index].beta);
  }
  return index < count;
}

__device__ void Thinker::commit() {
  Node &parent = get_parent_node();
  Node &node = get_node();
  table.update(node.mg.player_pos(), node.mg.opponent_pos(), -parent.beta, -parent.alpha, node.alpha);
  parent.alpha = max(parent.alpha, node.passed_prev ? node.alpha : -node.alpha);
  stack_index--;
}

__device__ bool Thinker::commit_or_next() {
  if (stack_index == 0) {
    if (!next_game())
      return true;
  } else {
    commit();
  }
  return false;
}

__device__ void Thinker::commit_from_leaf(int score) {
  Node &parent = get_parent_node();
  parent.alpha = max(parent.alpha, -score);
  stack_index--;
}

__device__ int Thinker::think() {
  ull nodes_count = 0;
  while (true) {
    nodes_count++;
    if (stack_index == stack_size) {
      int score = round(evaluator.eval(leaf_me, leaf_op));
      commit_from_leaf(score);
    } else {
      Node &node = get_node();
      if (node.mg.completed()) {
        if (!node.not_pass) { // pass
          if (node.passed_prev) { // end game
            node.alpha = node.mg.score();
            if (commit_or_next()) return nodes_count;
          } else {
            pass();
          }
        } else { // completed
          if (commit_or_next()) return nodes_count;
        }
      } else if (node.alpha >= node.beta) {
        if (commit_or_next()) return nodes_count;
      } else {
        ull next_bit = node.mg.next_bit();
        int pos = __popcll(next_bit - 1);
        ull flip_bits = flip(node.mg.player_pos(), node.mg.opponent_pos(), pos);
        if (flip_bits) {
          node.not_pass = true;
          if (stack_index == stack_size - 1) {
            leaf_me = node.mg.opponent_pos() ^ flip_bits;
            leaf_op = (node.mg.player_pos() ^ flip_bits) | UINT64_C(1) << pos;
          } else {
            Node& next_node = get_next_node();
            MobilityGenerator next_mg = node.mg.move(flip_bits, next_bit);
            Entry entry = table.find(next_mg.player_pos(), next_mg.opponent_pos());
            if (entry.enable) {
              char next_alpha = max(-node.beta, entry.lower);
              char next_beta = min(-node.alpha, entry.upper);
              next_node = Node(next_mg, next_alpha, next_beta);
            } else {
              next_node = Node(next_mg, -node.beta, -node.alpha);
            }
          }
          ++stack_index;
        }
      }
    }
  }
}

__global__ void think_kernel(
    const AlphaBetaProblem * const abp, int * const result,
    size_t count, size_t depth, Table table, Evaluator evaluator, ull * const nodes_total) {
  index_shared = blockIdx.x;
  __syncthreads();
  size_t index = atomicAdd(&index_shared, gridDim.x);
  if (index < count) {
    const AlphaBetaProblem &problem = abp[index];
    Thinker thinker = {
      0, // stack_index
      0, 0, // leaf
      depth,
      abp, // abp
      result, // result
      count, // count
      index, // index
      table, // table
      evaluator // evaluator
    };
    nodes_stack[threadIdx.x] = Node(MobilityGenerator(problem.player, problem.opponent), problem.alpha, problem.beta);
    ull nodes_count = thinker.think();
    atomicAdd(nodes_total, nodes_count);
  }
}

void init_batch(BatchedTask &bt, size_t batch_size, size_t max_depth, const Table &table) {
  bt.str = (cudaStream_t*)malloc(sizeof(cudaStream_t));
  cudaStreamCreate(bt.str);
  cudaMallocManaged((void**)&bt.abp, sizeof(AlphaBetaProblem) * batch_size);
  cudaMallocManaged((void**)&bt.result, sizeof(int) * batch_size);
  cudaMallocManaged((void**)&bt.total, sizeof(ull));
  *bt.total = 0;
  bt.table = table;
  bt.size = batch_size;
  bt.grid_size = (batch_size + chunk_size - 1) / chunk_size;
  bt.max_depth = max_depth;
  cudaMalloc((void**)&bt.upper_stacks, sizeof(UpperNode) * bt.grid_size * nodesPerBlock * (bt.max_depth - lower_stack_depth));
}

void launch_batch(const BatchedTask &bt) {
  alpha_beta_kernel<<<bt.grid_size, nodesPerBlock, sizeof(Node) * nodesPerBlock * (lower_stack_depth + 1), *bt.str>>>(
      bt.abp, bt.result, bt.upper_stacks, bt.size, bt.max_depth - lower_stack_depth, bt.table, bt.total);
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
  cudaFree(bt.total);
}

void init_batch(BatchedThinkTask &bt, size_t batch_size, size_t depth, const Table &table, const Evaluator &evaluator) {
  bt.str = (cudaStream_t*)malloc(sizeof(cudaStream_t));
  cudaStreamCreate(bt.str);
  cudaMallocManaged((void**)&bt.abp, sizeof(AlphaBetaProblem) * batch_size);
  cudaMallocManaged((void**)&bt.result, sizeof(int) * batch_size);
  cudaMallocManaged((void**)&bt.total, sizeof(ull));
  *bt.total = 0;
  bt.table = table;
  bt.evaluator = evaluator;
  bt.size = batch_size;
  bt.grid_size = (batch_size + chunk_size - 1) / chunk_size;
  bt.depth = depth;
}

void launch_batch(const BatchedThinkTask &bt) {
  think_kernel<<<bt.grid_size, nodesPerBlock, sizeof(Node) * nodesPerBlock * bt.depth, *bt.str>>>(
      bt.abp, bt.result, bt.size, bt.depth, bt.table, bt.evaluator, bt.total);
}

bool is_ready_batch(const BatchedThinkTask &bt) {
  return cudaStreamQuery(*bt.str) == cudaSuccess;
}

void destroy_batch(const BatchedThinkTask &bt) {
  cudaStreamDestroy(*bt.str);
  free(bt.str);
  cudaFree(bt.abp);
  cudaFree(bt.result);
  cudaFree(bt.total);
}
