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

#include "mirage/kernel/customized.h"
#include "mirage/kernel/graph.h"
#include "mirage/threadblock/graph.h"
#include "mirage/threadblock/operator.h"
#include "mirage/threadblock/smem_tensor.h"
#include <cassert>

namespace mirage {
namespace kernel {

using mirage::threadblock::STensor;

std::vector<DTensor> Graph::customized(std::vector<DTensor> const &inputs,
                                       threadblock::Graph const &bgraph) {
  KNOperator *op = create_customized_op(inputs, bgraph);
  assert(op != nullptr);
  operators.push_back(op);
  return op->output_tensors;
}

int Graph::customized(std::vector<DTensor const *> _inputs,
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

KNOperator *Graph::create_customized_op(std::vector<DTensor> const &inputs,
                                        threadblock::Graph const &_graph) {
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

KNCustomizedOp::KNCustomizedOp(mirage::kernel::Graph *_kgraph,
                               std::vector<DTensor> const &_inputs,
                               mirage::threadblock::Graph const &_graph)
    : KNOperator(_kgraph, mirage::type::KN_CUSTOMIZED_OP, _inputs),
      bgraph(_graph.grid_dim,
             _graph.block_dim,
             _graph.forloop_range,
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
                         input_op->forloop_dim,
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

void KNCustomizedOp::get_bgraph(mirage::threadblock::Graph **bgraph_) {
  *bgraph_ = &(this->bgraph);
}

KNCustomizedOp::~KNCustomizedOp() {
  for (int i = output_tensors.size() - 1; i >= 0; i--) {
    kgraph->free(output_tensors[i]);
  }
}

} // namespace kernel
} // namespace mirage
