#pragma once

#include "mirage/vector_types.h"
#include <algorithm>
#include <unordered_set>
#include <vector>

inline bool operator==(dim3 const &lhs, dim3 const &rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

inline bool operator==(int3 const &lhs, int3 const &rhs) {
  return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

inline std::vector<unsigned int> to_vector(dim3 const &d) {
  return {d.x, d.y, d.z};
}

inline std::vector<int> to_vector(int3 const &d) {
  return {d.x, d.y, d.z};
}

template <typename T>
std::vector<T> to_vector(int n, T *arr) {
  std::vector<T> v;
  for (int i = 0; i < n; ++i) {
    v.push_back(arr[i]);
  }
  return v;
}