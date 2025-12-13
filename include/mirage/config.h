/* Copyright 2023-2024 CMU
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
#include <cstddef>
#include <cstdint>

namespace mirage {
namespace config {

size_t const MAX_NUM_THREADBLOCKS_PER_KERNEL = 4096;
int const MAX_NUM_DEVICES = 16;
constexpr int MAX_TENSOR_DIMS = 4;
int const DEFAULT_TB_REDUCTION_DIMX = 64;
int const MAX_NUM_WARP_GROUPS = 4;
int const NUM_THREADS_PER_WARP = 32;
int const NUM_WARPS_PER_GROUP = 4;
int const NUM_THREADS_PER_GROUP = NUM_WARPS_PER_GROUP * NUM_THREADS_PER_WARP;
constexpr int MAX_TMA_DESC_PER_TENSOR = 3;

#if defined(MIRAGE_BACKEND_USE_CUDA)
size_t const MAX_DMEM_SIZE = (size_t)2 * 1024 * 1024 * 1024;    // 2 GB
size_t const MAX_SMEM_SIZE = 96 * 1024;                         // 96 KB
#else
#error "Please define MIRAGE_BACKEND_USE_CUDA."
#endif

} // namespace config
} // namespace mirage
