#pragma once
#include <utility>

#include "types.hpp"

using Board = std::pair<ull, ull>;

Board toBoard(const char* const str);
void fromBoard(const Board&, char* const);
