#pragma once
#include "types.hpp"

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
  __device__ Entry find(ull player, ull opponent) const;
  __device__ void update(ull player, ull opponent, char upper, char lower, char value) const;
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
    __threadfence();
  }
  __device__ void unlock(ull index) const {
    __threadfence();
    atomicExch(mutex + index, 0);
  }
};
