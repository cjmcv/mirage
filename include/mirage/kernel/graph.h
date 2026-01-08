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
                    mirage::layout::DmemLayout layout) {
    KNInputOp *op = new KNInputOp(dims, strides, data_type, layout);
    assert(op != nullptr);
    std::vector<DTensor>& output_tensors = op->get_output_dtensors();
    for (int i=0; i<output_tensors.size(); i++) {
      this->allocate(output_tensors[i]);
    }
    operators.push_back(op);
    return op->output_tensors[0];
  }
  DTensor *new_input_ptr(std::vector<int> const &dims,
                         std::vector<size_t> const &strides,
                         mirage::type::DataType data_type,
                         mirage::layout::DmemLayout layout) {
    KNInputOp *op = new KNInputOp(dims, strides, data_type, layout);
    assert(op != nullptr);
    std::vector<DTensor>& output_tensors = op->get_output_dtensors();
    for (int i=0; i<output_tensors.size(); i++) {
      this->allocate(output_tensors[i]);
    }
    operators.push_back(op);
    return &op->output_tensors[0];
  }
  // KNOperator *create_input_op(std::vector<int> const &dims,
  //                             std::vector<size_t> const &strides,
  //                             mirage::type::DataType data_type,
  //                             mirage::layout::DmemLayout layout) {
  //   KNInputOp *op = new KNInputOp(this, dims, strides, data_type, layout);
  //   return op;
  // }
  // customized operator
  std::vector<DTensor> customized(std::vector<DTensor> const &inputs,
                                  mirage::threadblock::Graph const &_graph) {
    KNOperator *op = create_customized_op(inputs, _graph);
    assert(op != nullptr);
    operators.push_back(op);
    return op->output_tensors;
  }

  int customized(std::vector<DTensor const *> _inputs,
                 DTensor **outputs,
                 mirage::threadblock::Graph const *bgraph) {
    std::vector<DTensor> inputs;
    for (auto const &t : _inputs) {
      inputs.push_back(t == nullptr ? DTensor::EMPTY_TENSOR : *t);
    }
    KNOperator *op = create_customized_op(inputs, *bgraph);
    assert(op != nullptr);
    operators.push_back(op);
    for (size_t i = 0; i < op->output_tensors.size(); i++) {
      outputs[i] = &op->output_tensors[i];
    }
    return op->output_tensors.size();
  }
  KNOperator *create_customized_op(std::vector<DTensor> const &inputs,
                                   mirage::threadblock::Graph const &_graph) {
    // Assert that _graph's dtensor inputs align with inputs
    {
      int num_inputs = 0;
      for (auto const &op : _graph.operators) {
        if (op->op_type == mirage::type::TB_INPUT_OP) {
          mirage::threadblock::TBInputOp const *input_op =
              static_cast<mirage::threadblock::TBInputOp const *>(op);
          assert(inputs[num_inputs] == input_op->dtensor);
          num_inputs++;
        }
      }
      assert(num_inputs == (int)inputs.size());
    }


    KNCustomizedOp *op = new KNCustomizedOp(this, inputs, _graph);
    return op;
  }
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
