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

#include "mirage/config.h"
#include "mirage/kernel/device_tensor.h"
#include "mirage/threadblock/operator.h"
#include "mirage/threadblock/smem_tensor.h"
// #include "mirage/vector_types.h"
#include <vector>

namespace mirage {
namespace threadblock {

class Graph {

public:
  Graph();
  Graph(dim3 grid_dim, dim3 block_dim, int forloop_range, int reduction_dimx);
  ~Graph();
  Graph(Graph const &) = delete;
  Graph &operator=(Graph const &) = delete;
  // input operator

  STensor new_input(mirage::kernel::DTensor const &dtensor,
                    int3 input_map,
                    int forloop_dim,
                    mirage::layout::SmemLayout layout,
                    bool store_in_dmem = false);
  STensor *new_input(mirage::kernel::DTensor const *dtensor,
                     int3 input_map,
                     int forloop_dim,
                     mirage::layout::SmemLayout layout,
                     bool store_in_dmem = false);
  TBOperator *create_input_op(mirage::kernel::DTensor const &dtensor,
                              int3 input_map,
                              int forloop_dim,
                              mirage::layout::SmemLayout layout,
                              bool store_in_dmem = false);

  size_t calculate_shared_memory_usage(TBOperator *new_op);
  int get_smem_size_with_pipeline() const;

public:
  dim3 grid_dim, block_dim, cluster_dim{4, 4, 1};
  int forloop_range;
  int reduction_dimx;
  std::vector<mirage::threadblock::TBOperator *> operators;
  // memory allocator
  off_t smem_offset;
  std::vector<std::pair<off_t, size_t>> allocated_tensors;

  using OpType = TBOperator;
  using TensorType = STensor;
};

} // namespace threadblock
} // namespace mirage
