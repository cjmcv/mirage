/* Copyright 2023-2025 CMU
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "mirage/config.h"
#include "mirage/type.h"
#include <algorithm>

namespace mirage {
namespace utils {

using namespace mirage::type;
using namespace mirage::config;
using namespace std;

#ifdef MIRAGE_FINGERPRINT_USE_CUDA
#define __execution_space__ __device__
#else
#define __execution_space__
#endif

inline __execution_space__ FPType compute_add_fingerprint(FPType a, FPType b) {
  uint32_t x = a;
  uint32_t y = b;
  return (x + y) % FP_PQ;
}

#undef __execution_space__

} // namespace utils
} // namespace mirage
