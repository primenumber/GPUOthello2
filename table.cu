#include "table.cuh"
#include "board.cuh"

__host__ Table::Table(const size_t table_size) : size(table_size) {
  cudaMallocManaged((void**)&entries, sizeof(Entry) * size);
  cudaMallocManaged((void**)&mutex, sizeof(int) * size);
  cudaMallocManaged((void**)&update_count, sizeof(ull));
  cudaMallocManaged((void**)&hit_count, sizeof(ull));
  cudaMallocManaged((void**)&lookup_count, sizeof(ull));
  *update_count = 0;
  *hit_count = 0;
  *lookup_count = 0;
  memset(entries, 0, sizeof(Entry) * size);
  memset(mutex, 0, sizeof(int) * size);
}

__host__ Table::~Table() {
#ifndef __CUDA_ARCH__
  if (enable) {
    cudaFree(entries);
    cudaFree(mutex);
    cudaFree(update_count);
    cudaFree(hit_count);
    cudaFree(lookup_count);
  }
#endif
}

Table::Table(Table&& that)
  : entries(that.entries), mutex(that.mutex),
    update_count(that.update_count), hit_count(that.hit_count),
    lookup_count(that.lookup_count) {
  that.enable = false;
}

__device__ Entry Table::find(ull player, ull opponent) const {
  atomicAdd(lookup_count, 1);
  ull hash = (player + 17 * opponent) % size;
  Entry result;
  for (int i = 0; i < 32; ++i) {
    if (threadIdx.x % 32 == i) {
      lock(hash);
      result = entries[hash];
      if (result.player != player || result.opponent != opponent) {
        result.enable = false;
      } else if (result.enable) {
        atomicAdd(hit_count, 1);
      }
      unlock(hash);
    }
  }
  return result;
}

__device__ void Table::update(ull player, ull opponent, char upper, char lower, char value) const {
  if (upper <= lower) {
    return;
  }
  atomicAdd(update_count, 1);
  Entry entry;
  if (value > lower && value < upper) {
    entry = Entry(player, opponent, value, value);
  } else {
    if (value <= lower) {
      entry = Entry(player, opponent, value, -64);
    } else if (value >= upper) {
      entry = Entry(player, opponent, 64, value);
    } else {
      return;
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
