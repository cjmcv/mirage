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

#include "mirage/kernel/graph.h"
#include "mirage/kernel/device_tensor.h"
#include "mirage/kernel/operator.h"
#include "mirage/threadblock/graph.h"
#include "mirage/threadblock/operator.h"
#include "mirage/threadblock/smem_tensor.h"

// #include "mirage/vector_types.h"
#include <tuple>

namespace mirage {
namespace kernel {

using mirage::threadblock::STensor;

class KNCustomizedOp : public mirage::kernel::KNOperator {
public:
  KNCustomizedOp(Graph *_kgraph,
                 std::vector<DTensor> const &_inputs,
                 mirage::threadblock::Graph const &_graph)
                 : KNOperator(mirage::type::KN_CUSTOMIZED_OP, _inputs),
      bgraph(_graph.grid_dim,
             _graph.block_dim,
             _graph.thread_num,
             _graph.reduction_dimx) {
    size_t input_idx = 0;
    for (auto const &op : _graph.operators) {
      std::vector<STensor> my_inputs;
      std::vector<std::pair<int, int>> indices;
      for (size_t i = 0; i < op->input_tensors.size(); i++) {
        int op_idx = -1, ts_idx = op->input_tensors[i].owner_ts_idx;
        for (size_t l = 0; l < _graph.operators.size(); l++) {
          if (_graph.operators[l] == op->input_tensors[i].owner_op) {
            assert(op_idx == -1);
            op_idx = static_cast<int>(l);
          }
        }
        assert(op_idx != -1);
        my_inputs.push_back(bgraph.operators[op_idx]->output_tensors[ts_idx]);
        indices.push_back({op_idx, ts_idx});
      }
      switch (op->op_type) {
        case mirage::type::TB_INPUT_OP: {
          assert(my_inputs.size() == 0);
          mirage::threadblock::TBInputOp *input_op =
              static_cast<mirage::threadblock::TBInputOp *>(op);
          DTensor const &dtensor = _inputs[input_idx++];
          bgraph.new_input(dtensor,
                          input_op->input_map,
                          input_op->output_tensors[0].layout,
                          input_op->output_tensors[0].store_in_dmem);
          break;
        }
        default: {
          assert(false && "Unsupported threadblock operator");
        }
      }
    }
  }
  virtual ~KNCustomizedOp() { // CJM
    // for (int i = output_tensors.size() - 1; i >= 0; i--) {
    //   kgraph->free(output_tensors[i]);
    // }
  }

public:
  mirage::threadblock::Graph bgraph;
  void get_bgraph(mirage::threadblock::Graph **bgraph) {
    *bgraph = &(this->bgraph);
  }
};

} // namespace kernel
} // namespace mirage
