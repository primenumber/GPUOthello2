#include "solver.cuh"
#include <cstdlib>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#include "board.cuh"
#include "node.cuh"

// parameters
constexpr int lower_stack_depth = 9;

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
  unsigned int *index_shared;

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
    parent.commit(node.result);
  } else {
    parent.commit(-node.result);
  }
  --stack_index;
}

__device__ void Solver::pass() {
  Node& node = get_node();
  node.mg = node.mg.pass();
  int tmp = node.alpha;
  node.result = -SHRT_MAX;
  node.alpha = -node.beta;
  node.beta = -tmp;
  node.passed_prev = true;
}

class UpperNode {
 public:
  static constexpr int max_mobility_count = 46;
  __device__ UpperNode(ull player, ull opponent, char result, char alpha, char beta, bool pass = false)
      : player(player), opponent(opponent), possize(0), index(0),
      result(result), start_alpha(alpha), alpha(alpha), beta(beta), prev_passed(pass) {
    MobilityGenerator mg(player, opponent);
    char cntary[max_mobility_count];
    while(!mg.completed()) {
      ull next_bit = mg.next_bit();
      int pos = __popcll(next_bit - 1);
      ull flip_bits = flip(player, opponent, pos);
      if (flip_bits) {
        cntary[possize] = mobility_count(opponent ^ flip_bits, (player ^ flip_bits) | next_bit);
        posary[possize++] = static_cast<hand>(pos);
      }
    }
    thrust::sort_by_key(thrust::seq, cntary, cntary + possize, posary);
  }
  UpperNode& operator=(const UpperNode &) = default;
  __device__ bool completed() const {
    return index == possize;
  }
  __device__ hand pop() {
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
    //Entry entry = table.find(next_player, next_opponent);
    //if (entry.enable) {
    //  char next_alpha = max(-beta, entry.lower);
    //  char next_beta = min(-alpha, entry.upper);
    //  if (next_alpha >= next_beta) {
    //    return UpperNode(next_player, next_opponent, next_beta, next_alpha, next_beta);
    //  } else {
    //    return UpperNode(next_player, next_opponent, -64, next_alpha, next_beta);
    //  }
    //} else {
      return UpperNode(next_player, next_opponent, -64, -beta, -alpha);
    //}
  }
  __device__ UpperNode pass(Table table) const {
    //Entry entry = table.find(opponent, player);
    //if (entry.enable) {
    //  char next_alpha = max(-beta, entry.lower);
    //  char next_beta = min(-alpha, entry.upper);
    //  if (next_alpha >= next_beta) {
    //    return UpperNode(opponent, player, next_beta, next_alpha, next_beta, true);
    //  } else {
    //    return UpperNode(opponent, player, -64, next_alpha, next_beta, true);
    //  }
    //} else {
      return UpperNode(opponent, player, -64, -beta, -alpha, true);
    //}
  }
  __device__ void commit(char score) {
    result = max(result, score);
    alpha = max(alpha, result);
  }
  char result;
  char start_alpha;
  char alpha;
  char beta;
 private:
  ull player, opponent;
  hand posary[max_mobility_count];
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
  result[index] = node.passed() ? -node.result : node.result;
  index = atomicAdd(index_shared, gridDim.x);
  if (index < count) {
    upper_stack[0] = UpperNode(abp[index].player, abp[index].opponent, -64, abp[index].alpha, abp[index].beta);
  }
  return index < count;
}

__device__ void Solver::commit_upper() {
  UpperNode &parent = upper_stack[stack_index-1];
  UpperNode &node = upper_stack[stack_index];
  //table.update(node.player_pos(), node.opponent_pos(), node.beta, node.start_alpha, node.result);
  parent.commit(node.passed() ? node.result: -node.result);
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
  parent.commit(node.passed_prev ? node.result : -node.result);
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
        node.result = node.score();
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
    hand pos = node.pop();
    ull flip_bits = flip(node.player_pos(), node.opponent_pos(), static_cast<int>(pos));
    assert(flip_bits);
    ull bit = UINT64_C(1) << static_cast<int>(pos);
    if (stack_index < upper_stack_size - 1) {
      UpperNode& next_node = upper_stack[stack_index+1];
      next_node = node.move(flip_bits, bit, table);
    } else {
      Node& next_node = get_next_node();
      next_node = Node(MobilityGenerator(node.opponent_pos() ^ flip_bits, (node.player_pos() ^ flip_bits) | bit), -node.beta, -node.alpha);
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
        node.result = node.mg.score();
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
  int index_global = blockIdx.x * blockDim.x + threadIdx.x;
  __shared__ unsigned int index_shared;
  index_shared = blockIdx.x;
  __syncthreads();
  size_t index = atomicAdd(&index_shared, gridDim.x);
  if (index < count) {
    UpperNode *ustack = upper_stack + index_global * upper_stack_size;
    const AlphaBetaProblem &problem = abp[index];
    Solver solver = {
      0, // stack_index
      ustack, // upper_stack
      upper_stack_size, // upper_stack_size
      abp, // abp
      result, // result
      count, // count
      index, // index
      table, // table
      &index_shared
    };
    solver.upper_stack[0] = UpperNode(problem.player, problem.opponent, -64, problem.alpha, problem.beta);
    ull nodes_count = solver.solve_all();
    atomicAdd(nodes_total, nodes_count);
  }
}

BatchedTask::BatchedTask(const size_t batch_size, const size_t max_depth,
    const Table &table) : table(table), size(batch_size), max_depth(max_depth) {
  str = (cudaStream_t*)malloc(sizeof(cudaStream_t));
  cudaStreamCreate(str);
  cudaMallocManaged((void**)&abp, sizeof(AlphaBetaProblem) * size);
  cudaMallocManaged((void**)&result, sizeof(int) * size);
  cudaMallocManaged((void**)&total, sizeof(ull));
  *total = 0;
  grid_size = (batch_size + chunk_size - 1) / chunk_size;
  cudaMalloc((void**)&upper_stacks, sizeof(UpperNode) * grid_size * nodesPerBlock * (max_depth - lower_stack_depth));
}

BatchedTask::BatchedTask(BatchedTask&& that)
  : BatchedTask(that) {
  that.str = nullptr;
}

void BatchedTask::launch() const {
  alpha_beta_kernel<<<grid_size, nodesPerBlock, sizeof(Node) * nodesPerBlock * (lower_stack_depth + 1), *str>>>(
      abp, result, upper_stacks, size, max_depth - lower_stack_depth, table, total);
}

bool BatchedTask::is_ready() const {
  return cudaStreamQuery(*str) == cudaSuccess;
}

BatchedTask::~BatchedTask() {
  if (str != nullptr) {
    cudaStreamDestroy(*str);
    free(str);
    cudaFree(abp);
    cudaFree(result);
    cudaFree(upper_stacks);
    cudaFree(total);
  }
}
