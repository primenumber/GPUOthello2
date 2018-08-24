#pragma once
#include <string>
#include "types.hpp"

class Evaluator {
 public:
  __host__ Evaluator() {};
  __host__ Evaluator(const std::string &features_file_name, const std::string &values_file_name);
  __host__ ~Evaluator();
  __host__ __device__ float eval(const ull me, const ull op) const;
 private:
  __host__ __device__ int get_index(const ull me, const ull op, const ull feature) const;
  size_t features_count;
  float **values;
  float offset;
  ull *features;
  int *base3_table;
};
