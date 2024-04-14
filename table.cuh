#pragma once
#include "types.hpp"

struct Entry {
 public:
  __host__ __device__ Entry() : player(0), opponent(0), upper(64), lower(-64), enable(false) {}
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
  __host__ Table(const size_t table_size);
  __host__ __device__ ~Table();
  Table(Table&&);
  __device__ Entry find(ull player, ull opponent);
  __device__ void update(ull player, ull opponent, char upper, char lower, char value);
  __host__ Table weak_clone() const {
    return Table(entries, mutex, size, update_count, hit_count, lookup_count, false);
  }
  __host__ ull get_update_count() const {
    ull result = 0;
    cudaMemcpy(&result, update_count, sizeof(ull), cudaMemcpyDeviceToHost);
    return result;
  }
  __host__ ull get_hit_count() const {
    ull result = 0;
    cudaMemcpy(&result, hit_count, sizeof(ull), cudaMemcpyDeviceToHost);
    return result;
  }
  __host__ ull get_lookup_count() const {
    ull result = 0;
    cudaMemcpy(&result, lookup_count, sizeof(ull), cudaMemcpyDeviceToHost);
    return result;
  }
  Entry * entries;
  int *mutex;
  size_t size;
  ull *update_count;
  ull *hit_count;
  ull *lookup_count;
  bool enable = true;
 private:
  __host__ Table(Entry * entries, int *mutex, size_t size,
      ull *update_count, ull *hit_count, ull *lookup_count, bool enable)
    : entries(entries), mutex(mutex), size(size), update_count(update_count),
      hit_count(hit_count), lookup_count(lookup_count), enable(enable) {}
  __device__ bool try_lock(ull index) {
    bool result = atomicCAS(mutex + index, 0, 1) == 0;
    __threadfence();
    return result;
  }
  __device__ void lock(ull index) {
    while (atomicCAS(mutex + index, 0, 1) != 0);
    __threadfence();
  }
  __device__ void unlock(ull index) {
    __threadfence();
    atomicExch(mutex + index, 0);
  }
};
