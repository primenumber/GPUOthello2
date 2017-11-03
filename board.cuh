#pragma once
#include "types.hpp"

__host__ __device__ ull flip(ull player, ull opponent, int pos);
__host__ __device__ ull mobility(ull player, ull opponent);
__host__ __device__ int mobility_count(ull player, ull opponent);
