#pragma once
#include <vector>
#include "types.hpp"

struct AlphaBetaProblem {
  __host__ __device__ AlphaBetaProblem(ull player, ull opponent, int alpha, int beta)
    : player(player), opponent(opponent), alpha(alpha), beta(beta) {}
  __host__ __device__ AlphaBetaProblem(ull player, ull opponent)
    : player(player), opponent(opponent), alpha(-64), beta(64) {}
  ull player;
  ull opponent;
  int alpha;
  int beta;
};

struct Entry {
 public:
  __host__ __device__ Entry() : player(0), opponent(0), upper(64), lower(64), enable(false) {}
  __host__ __device__ Entry(ull player, ull opponent, char upper, char lower)
    : player(player), opponent(opponent), upper(upper), lower(lower), enable(true) {}
  ull player;
  ull opponent;
  char upper;
  char lower;
  bool enable;
};

class Table {
 public:
  __device__ Entry find(ull player, ull opponent) const {
    atomicAdd(lookup_count, 1);
    ull hash = (player + 17 * opponent) % size;
    Entry result;
    for (int i = 0; i < 32; ++i) {
      if (threadIdx.x % 32 == i) {
        lock(hash);
        result = entries[hash];
        if (result.player != player || result.opponent != opponent) {
          result.enable = false;
        } else {
          atomicAdd(hit_count, 1);
        }
        unlock(hash);
      }
    }
    return result;
  }
  __device__ void update(ull player, ull opponent, char upper, char lower, char value) const {
    atomicAdd(update_count, 1);
    Entry entry;
    if (value > lower && value < upper) {
      entry = Entry(player, opponent, value, value);
    } else {
      if (value <= lower) {
        entry = Entry(player, opponent, -64, value);
      } else {
        entry = Entry(player, opponent, value, 64);
      }
    }
    ull hash = (player + 17 * opponent) % size;
    for (int i = 0; i < 32; ++i) {
      if (threadIdx.x % 32 == i) {
        lock(hash);
        const Entry tmp = entries[hash];
        if (tmp.player != player || tmp.opponent != opponent || !tmp.enable) {
          entries[hash] = entry;
        } else {
          entries[hash].upper = min(tmp.upper, entry.upper);
          entries[hash].lower = max(tmp.lower, entry.lower);
        }
        unlock(hash);
      }
    }
  }
  Entry * entries;
  mutable int *mutex;
  size_t size;
  ull *update_count;
  ull *hit_count;
  ull *lookup_count;
 private:
  __device__ bool try_lock(ull index) const {
    bool result = atomicCAS(mutex + index, 0, 1) == 0;
    __threadfence();
    return result;
  }
  __device__ void lock(ull index) const {
    while (atomicCAS(mutex + index, 0, 1) != 0);
  }
  __device__ void unlock(ull index) const {
    __threadfence();
    atomicExch(mutex + index, 0);
  }
};

class UpperNode;

struct BatchedTask {
  cudaStream_t *str;
  AlphaBetaProblem *abp;
  UpperNode *upper_stacks;
  Table table;
  int *result;
  size_t max_depth;
  size_t size;
  size_t grid_size;
  ull *total;
};

void init_batch(BatchedTask &bt, size_t batch_size, size_t max_depth, const Table &table);
void launch_batch(const BatchedTask &bt);
bool is_ready_batch(const BatchedTask &bt);
void destroy_batch(const BatchedTask &bt);
