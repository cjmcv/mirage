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

#include "mirage/threadblock/graph.h"

namespace mirage {
namespace threadblock {

Graph::Graph()
    : grid_dim(1, 1, 1), block_dim(1, 1, 1), forloop_range(1),
      reduction_dimx(1), smem_offset(0) {}

Graph::Graph(dim3 _grid_dim,
             dim3 _block_dim,
             int _forloop_range,
             int _reduction_dimx)
    : grid_dim(_grid_dim), block_dim(_block_dim), forloop_range(_forloop_range),
      reduction_dimx(_reduction_dimx), smem_offset(0) {
  // A bgraph cannot have more than MAX_NUM_THREADBLOCKS_PER_KERNEL threadblocks
  // otherwise we don't have enough buffers in device memory for saving
  // fingerprints
  assert(grid_dim.x * grid_dim.y * grid_dim.z <=
         mirage::config::MAX_NUM_THREADBLOCKS_PER_KERNEL);
  assert(reduction_dimx > 0);
}

Graph::~Graph() {
  while (!operators.empty()) {
    delete operators.back();
    operators.pop_back();
  }
}

size_t Graph::calculate_shared_memory_usage(TBOperator *new_op) {
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

int Graph::get_smem_size_with_pipeline() const {
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

} // namespace threadblock
} // namespace mirage
