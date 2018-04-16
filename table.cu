#include "table.cuh"
#include "board.cuh"

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
