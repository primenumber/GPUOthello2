#include "eval_host.hpp"
#include <x86intrin.h>

ull pext(ull x, ull mask) {
  return _pext_u64(x, mask);
}
