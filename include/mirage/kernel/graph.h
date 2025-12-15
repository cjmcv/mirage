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

#include "mirage/kernel/customized.h"
#include "mirage/kernel/device_tensor.h"
#include "mirage/kernel/operator.h"
#include "mirage/kernel/runtime.h"
#include "mirage/threadblock/graph.h"
#include <vector>

namespace mirage {
namespace kernel {

class Graph {

public:
  Graph(dim3 gpu_dim = {1, 1, 1});
  ~Graph();
  Graph(Graph const &) = delete;
  Graph &operator=(Graph const &) = delete;
  // input operator
  DTensor new_input(std::vector<int> const &dims,
                    std::vector<size_t> const &strides,
                    mirage::type::DataType data_type,
                    mirage::layout::DmemLayout layout);
  DTensor *new_input_ptr(std::vector<int> const &dims,
                         std::vector<size_t> const &strides,
                         mirage::type::DataType data_type,
                         mirage::layout::DmemLayout layout);
  KNOperator *create_input_op(std::vector<int> const &dims,
                              std::vector<size_t> const &strides,
                              mirage::type::DataType data_type,
                              mirage::layout::DmemLayout layout);
  // customized operator
  std::vector<DTensor> customized(std::vector<DTensor> const &inputs,
                                  mirage::threadblock::Graph const &_graph);
  int customized(std::vector<DTensor const *> inputs,
                 DTensor **outputs,
                 mirage::threadblock::Graph const *bgraph);
  KNOperator *create_customized_op(std::vector<DTensor> const &inputs,
                                   mirage::threadblock::Graph const &_graph);
  // persistent kernel functions
  void attach_torch_tensor(DTensor const *input,
                           void *torch_ptr,
                           char const *name);
  void attach_cuda_tensor(DTensor const *input, char const *name);
  void attach_nvshmem_tensor(DTensor const *input, char const *name);
  DTensor *fuse_tensors(std::vector<DTensor const *> inputs,
                        int fused_dim,
                        int num_groups,
                        char const *name);
  DTensor *shuffle_tensors(std::vector<DTensor const *> inputs,
                           int shuffled_dim,
                           int num_groups,
                           char const *name);
  void register_task(char const *task_type, std::vector<int> params);
  runtime::TaskGraphResult generate_task_graph(int num_gpus, int my_gpu_id);

  // helper functions
  int get_num_input_dtensors() const;
  // int get_num_output_dtensors() const;
  int get_input_dtensors(DTensor **inputs) const;
  int get_input_dtensor_shape_and_stride(DTensor const *input,
                                         int *strides,
                                         int *dims) const;

  bool allocate(DTensor &tensor);
  void free(DTensor &tensor);

public:
  std::vector<mirage::kernel::KNOperator *> operators;
  dim3 gpu_dim;
  // memory allocator
  // device memory offset manager
  off_t dmem_data_offset;
  std::vector<std::pair<off_t, size_t>> allocated_data_tensors;

  // Fields for persistent kernels
  std::map<mirage::type::GuidType, mirage::runtime::IODesc> io_config;
  std::unordered_map<mirage::kernel::KNOperator const *,
                     std::tuple<int, int, runtime::TaskType, int>> task_config;

  using OpType = KNOperator;
  using TensorType = DTensor;
};

} // namespace kernel
} // namespace mirage
