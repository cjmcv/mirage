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
#include <vector>

namespace mirage {
namespace threadblock {

class Graph {

public:
  Graph()    
    : grid_dim(1, 1, 1), block_dim(1, 1, 1), thread_num(128),
      reduction_dimx(1), smem_offset(0) {}

  Graph(dim3 _grid_dim, dim3 _block_dim, int _thread_num, int _reduction_dimx)
    : grid_dim(_grid_dim), block_dim(_block_dim), thread_num(_thread_num),
    reduction_dimx(_reduction_dimx), smem_offset(0) {
  // A bgraph cannot have more than MAX_NUM_THREADBLOCKS_PER_KERNEL threadblocks
  // otherwise we don't have enough buffers in device memory for saving
  // fingerprints
  assert(grid_dim.x * grid_dim.y * grid_dim.z <=
      mirage::config::MAX_NUM_THREADBLOCKS_PER_KERNEL);
  assert(reduction_dimx > 0);
  }

  ~Graph() {
    while (!operators.empty()) {
      delete operators.back();
      operators.pop_back();
    }
  }

  Graph(Graph const &) = delete;
  Graph &operator=(Graph const &) = delete;
  // input operator

  STensor new_input(mirage::kernel::DTensor const &dtensor,
                    int3 input_map,
                    mirage::layout::SmemLayout layout,
                    bool store_in_dmem = false) {
    TBOperator *op =
        create_input_op(dtensor, input_map, layout, store_in_dmem);
    assert(op != nullptr);
    operators.push_back(op);
    return op->output_tensors[0];
  }
                    
  STensor *new_input(mirage::kernel::DTensor const *dtensor,
                     int3 input_map,
                     mirage::layout::SmemLayout layout,
                     bool store_in_dmem = false){
    TBOperator *op = create_input_op(
        dtensor == nullptr ? kernel::DTensor::EMPTY_TENSOR : *dtensor,
        input_map,
        layout,
        store_in_dmem);
    assert(op != nullptr);
    operators.push_back(op);
    return &op->output_tensors[0];
  }
  TBOperator *create_input_op(mirage::kernel::DTensor const &dtensor,
                              int3 input_map,
                              mirage::layout::SmemLayout layout,
                              bool store_in_dmem = false){
    TBInputOp *op = new TBInputOp(
      grid_dim, smem_offset, dtensor, input_map, layout, store_in_dmem);

    // Check shmem usage
    size_t smem_usage = calculate_shared_memory_usage(op);
    if (smem_usage > mirage::config::MAX_SMEM_SIZE) {
      delete op;
      return nullptr;
    } else {
      return op;
    }
  }

  size_t calculate_shared_memory_usage(TBOperator *new_op) {
    size_t usage = 0;
    if (new_op != nullptr) {
      operators.push_back(new_op);
    }
  
    // currently use a simple heuristic to calculate shmem usage
    // TODO: replace the following with a transpiler-based method
    for (auto const &op : operators) {
      // printf("op->op_type: %d.\n", op->op_type);
      switch (op->op_type) {
        case mirage::type::TB_INPUT_OP: {
          for (size_t i = 0; i < op->output_tensors.size(); i++) {
            // Do not store in smem when store_in_demm is set
            if (op->output_tensors[i].store_in_dmem) {
              continue;
            }
            usage += op->output_tensors[i].size();
          }
          break;
        }
        default: {
          assert(false && "Unsupported operator");
        }
      }
    }
  
    if (new_op != nullptr) {
      operators.pop_back();
    }
    return usage;
  }

  int get_smem_size_with_pipeline() const {
    int ret = smem_offset;
    // For pipelining, we use double buffers for all input loaders
    for (size_t i = 0; i < operators.size(); i++) {
      if (operators[i]->op_type == mirage::type::TB_INPUT_OP) {
        STensor stensor = operators[i]->output_tensors[0];
        ret += stensor.size();
      }
    }
    return ret;
  }

public:
  dim3 grid_dim, block_dim, cluster_dim{4, 4, 1};
  int thread_num;
  int reduction_dimx;
  std::vector<mirage::threadblock::TBOperator *> operators;
  // memory allocator
  off_t smem_offset;
  std::vector<std::pair<off_t, size_t>> allocated_tensors;

  using OpType = TBOperator;
  using TensorType = STensor;
};

////////////////////////////////////////////////////////////////////////

} // namespace threadblock
} // namespace mirage
