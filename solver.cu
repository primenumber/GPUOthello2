#include "solver.cuh"
#include <cstdlib>
#include <thrust/sort.h>
#include <thrust/execution_policy.h>

constexpr int nodesPerBlock = 64;
constexpr int lower_stack_depth = 9;

__constant__ ull mask1[4] = {
  0x0080808080808080ULL,
  0x7f00000000000000ULL,
  0x0102040810204000ULL,
  0x0040201008040201ULL
};
constexpr ull mask1_host[4] = {
  0x0080808080808080ULL,
  0x7f00000000000000ULL,
  0x0102040810204000ULL,
  0x0040201008040201ULL
};
__constant__ ull mask2[4] = {
  0x0101010101010100ULL,
  0x00000000000000feULL,
  0x0002040810204080ULL,
  0x8040201008040200ULL
};
constexpr ull mask2_host[4] = {
  0x0101010101010100ULL,
  0x00000000000000feULL,
  0x0002040810204080ULL,
  0x8040201008040200ULL
};

__host__ __device__ ull flip_impl(ull player, ull opponent, int pos, int simd_index) {
  ull om = opponent;
  if (simd_index) om &= 0x7E7E7E7E7E7E7E7EULL;
#ifdef __CUDA_ARCH__
  ull mask = mask1[simd_index] >> (63 - pos);
  ull outflank = (0x8000000000000000ULL >> __clzll(~om & mask)) & player;
#else
  ull mask = mask1_host[simd_index] >> (63 - pos);
  ull outflank = (0x8000000000000000ULL >> __builtin_clzll(~om & mask)) & player;
#endif
  ull flipped = (-outflank << 1) & mask;
#ifdef __CUDA_ARCH__
  mask = mask2[simd_index] << pos;
#else
  mask = mask2_host[simd_index] << pos;
#endif
  outflank = mask & ((om | ~mask) + 1) & player;
  flipped |= (outflank - (outflank != 0)) & mask;
  return flipped;
}

__host__ __device__ ull flip_seq(ull player, ull opponent, int pos) {
  return flip_impl(player, opponent, pos, 0)
    | flip_impl(player, opponent, pos, 1)
    | flip_impl(player, opponent, pos, 2)
    | flip_impl(player, opponent, pos, 3);
}

//__device__ ull flip_simd(ull player, ull opponent, int pos) {
//  ull flip = flip_impl(player, opponent, pos, threadIdx.x);
//  flip |= __shfl_xor(flip, 0x1);
//  flip |= __shfl_xor(flip, 0x2);
//  return flip;
//}

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
#ifdef __CUDA_ARCH__
    int pcnt = __popcll(player_pos());
    int ocnt = __popcll(opponent_pos());
#else
    int pcnt = __builtin_popcountll(player_pos());
    int ocnt = __builtin_popcountll(opponent_pos());
#endif
    if (pcnt == ocnt) return 0;
    if (pcnt > ocnt) return 64 - 2*ocnt;
    return 2*pcnt - 64;
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

__host__ __device__ ull mobility_impl(ull player, ull opponent, int simd_index) {
  ull PP, mOO, MM, flip_l, flip_r, pre_l, pre_r, shift2;
  ull shift1[4] = { 1, 7, 9, 8 };
  ull mflipH[4] = { 0x7e7e7e7e7e7e7e7eULL, 0x7e7e7e7e7e7e7e7eULL, 0x7e7e7e7e7e7e7e7eULL, 0xffffffffffffffffULL };

  PP = player;
  mOO = opponent & mflipH[simd_index];
  flip_l  = mOO & (PP << shift1[simd_index]);       flip_r  = mOO & (PP >> shift1[simd_index]);
  flip_l |= mOO & (flip_l << shift1[simd_index]);   flip_r |= mOO & (flip_r >> shift1[simd_index]);
  pre_l   = mOO & (mOO << shift1[simd_index]);      pre_r   = pre_l >> shift1[simd_index];
  shift2 = shift1[simd_index] + shift1[simd_index];
  flip_l |= pre_l & (flip_l << shift2);             flip_r |= pre_r & (flip_r >> shift2);
  flip_l |= pre_l & (flip_l << shift2);             flip_r |= pre_r & (flip_r >> shift2);
  MM = flip_l << shift1[simd_index];                MM |= flip_r >> shift1[simd_index];
  return MM & ~(player|opponent);
}

__host__ __device__ ull mobility(ull player, ull opponent) {
  return mobility_impl(player, opponent, 0)
    | mobility_impl(player, opponent, 1)
    | mobility_impl(player, opponent, 2)
    | mobility_impl(player, opponent, 3);
}

__host__ __device__ int mobility_count(ull player, ull opponent) {
#ifdef __CUDA_ARCH__
  return __popcll(mobility(player, opponent));
#else
  return __builtin_popcountll(mobility(player, opponent));
#endif
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
      ull flip = flip_seq(player, opponent, pos);
      if (flip) {
        cntary[possize] = mobility_count(opponent ^ flip, (player ^ flip) | next_bit);
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
    int pcnt = __popcll(player);
    int ocnt = __popcll(opponent);
    if (pcnt == ocnt) return 0;
    if (pcnt > ocnt) return 64 - 2*ocnt;
    return 2*pcnt - 64;
  }
  __device__ UpperNode move(ull bits, ull pos_bit) const {
    return UpperNode(opponent ^ bits, (player ^ bits) | pos_bit, -beta, -alpha);
  }
  __device__ UpperNode pass() const {
    return UpperNode(opponent, player, -beta, -alpha, true);
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

__device__ void pass_upper(UpperNode * const upper_stack, const int stack_index) {
  UpperNode& node = upper_stack[stack_index];
  node = node.pass();
}

__device__ bool next_game(
    const AlphaBetaProblem * const abp, int * const result, UpperNode * const upper_stack,
    const size_t count, size_t &index) {
  UpperNode &node = upper_stack[0];
  result[index] = node.alpha;
  index += gridDim.x * blockDim.x;
  if (index < count) {
    upper_stack[0] = UpperNode(abp[index].player, abp[index].opponent, abp[index].alpha, abp[index].beta);
  }
  return index < count;
}

__device__ void commit_upper(UpperNode * const upper_stack, int &stack_index) {
  UpperNode &parent = upper_stack[stack_index-1];
  UpperNode &node = upper_stack[stack_index];
  parent.alpha = max(parent.alpha, node.passed() ? node.alpha : -node.alpha);
  stack_index--;
}

__device__ bool commit_or_next(
    const AlphaBetaProblem * const abp, int * const result, UpperNode * const upper_stack,
    const size_t count, size_t &index, int &stack_index, int &nodes_count2) {
  if (stack_index == 0) {
    if (!next_game(abp, result, upper_stack, count, index))
      return true;
    nodes_count2 = 0;
  } else {
    commit_upper(upper_stack, stack_index);
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
    size_t &index, int &stack_index, int &nodes_count2) {
  ++nodes_count2;
  UpperNode& node = upper_stack[stack_index];
  if (node.completed()) {
    if (node.size() == 0) { // pass
      if (node.passed()) {
        node.alpha = node.score();
        if (commit_or_next(abp, result, upper_stack, count, index, stack_index, nodes_count2)) return true;
      } else {
        pass_upper(upper_stack, stack_index);
      }
    } else { // completed
      if (commit_or_next(abp, result, upper_stack, count, index, stack_index, nodes_count2)) return true;
    }
  } else if (node.alpha >= node.beta) {
    if (commit_or_next(abp, result, upper_stack, count, index, stack_index, nodes_count2)) return true;
  } else {
    int pos = node.pop();
    ull flip = flip_seq(node.player_pos(), node.opponent_pos(), pos);
    assert(flip);
    if (stack_index < upper_stack_size - 1) {
      UpperNode& next_node = upper_stack[stack_index+1];
      next_node = node.move(flip, UINT64_C(1) << pos);
    } else {
      Node& next_node = get_next_node(stack_index, upper_stack_size);
      next_node = Node(MobilityGenerator(node.opponent_pos() ^ flip, (node.player_pos() ^ flip) | (UINT64_C(1) << pos)), -node.beta, -node.alpha);
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
    ull flip = flip_seq(node.mg.player_pos(), node.mg.opponent_pos(), pos);
    if (flip) { // movable
      node.not_pass = true;
      Node& next_node = get_next_node(stack_index, upper_stack_size);
      next_node = Node(node.mg.move(flip, next_bit), -node.beta, -node.alpha);
      ++stack_index;
    }
  }
}

__device__ void solve_all(const AlphaBetaProblem * const abp, int * const result, UpperNode * const upper_stack,
    const size_t count, const size_t upper_stack_size, size_t &index) {
  int stack_index = 0;
  int nodes_count = 0;
  int nodes_count2 = 0;
  while (true) {
    ++nodes_count;
    //printf("%d %d: %d %d %d\n", threadIdx.x, blockIdx.x, stack_index, index, nodes_count);
    assert(index < count);
    if (stack_index < upper_stack_size) {
      if (solve_all_upper(abp, result, upper_stack, count, upper_stack_size, index, stack_index, nodes_count2)) return;
    } else {
      solve_all_lower(upper_stack, upper_stack_size, stack_index);
    }
  }
}

__global__ void alpha_beta_kernel(
    const AlphaBetaProblem * const abp, int * const result, UpperNode * const upper_stack,
    size_t count, size_t upper_stack_size) {
  size_t index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < count) {
    UpperNode *ustack = upper_stack + index * upper_stack_size;
    const AlphaBetaProblem &problem = abp[index];
    ustack[0] = UpperNode(problem.player, problem.opponent, problem.alpha, problem.beta);
    solve_all(abp, result, ustack, count, upper_stack_size, index);
  }
}

void init_batch(BatchedTask &bt, size_t batch_size, size_t max_depth) {
  constexpr int chunk_size = 2048;
  bt.str = (cudaStream_t*)malloc(sizeof(cudaStream_t));
  cudaStreamCreate(bt.str);
  cudaMallocManaged((void**)&bt.abp, sizeof(AlphaBetaProblem) * batch_size);
  cudaMallocManaged((void**)&bt.result, sizeof(int) * batch_size);
  bt.size = batch_size;
  bt.grid_size = (batch_size + chunk_size - 1) / chunk_size;
  bt.max_depth = max_depth;
  cudaMalloc((void**)&bt.upper_stacks, sizeof(UpperNode) * bt.grid_size * nodesPerBlock * (bt.max_depth - lower_stack_depth));
}

void launch_batch(const BatchedTask &bt) {
  alpha_beta_kernel<<<bt.grid_size, nodesPerBlock, 0, *bt.str>>>(bt.abp, bt.result, bt.upper_stacks, bt.size, bt.max_depth - lower_stack_depth);
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
