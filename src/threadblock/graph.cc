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
// #include "mirage/threadblock/serializer/concat_serializer.h"
// #include "mirage/threadblock/serializer/element_binary_serializer.h"
// #include "mirage/threadblock/serializer/element_unary_serializer.h"
// #include "mirage/threadblock/serializer/forloop_accum_serializer.h"
// #include "mirage/threadblock/serializer/input_loader_serializer.h"
// #include "mirage/threadblock/serializer/matmul_serializer.h"
// #include "mirage/threadblock/serializer/output_saver_serializer.h"
// #include "mirage/threadblock/serializer/reduction_serializer.h"
// #include "mirage/threadblock/serializer/rms_norm_serializer.h"
// #include "mirage/utils/hash_utils.h"

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

// size_t Graph::pair_hash::operator()(std::pair<int, int> const &p) const {
//   size_t h1 = std::hash<int>{}(p.first);
//   size_t h2 = std::hash<int>{}(p.second);
//   hash_combine(h1, h2);
//   return h1;
// }

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
      case mirage::type::TB_INPUT_OP:
      case mirage::type::TB_OUTPUT_OP: {
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

Graph::operator json() const {
  json j = {{"graph_level", "thread_block_graph"},
            {"grid_dim", grid_dim},
            {"block_dim", block_dim},
            {"forloop_range", forloop_range},
            {"reduction_dimx", reduction_dimx},
            {"operators", {}},
            {"smem_offset", smem_offset}};
  for (TBOperator *const op : operators) {
    j["operators"].push_back(json(*op));
  }
  return j;
}

void from_json(json const &j, Graph &graph) {
  graph.grid_dim = j.at("grid_dim").get<dim3>();
  graph.block_dim = j.at("block_dim").get<dim3>();
  graph.forloop_range = j.at("forloop_range").get<int>();
  graph.reduction_dimx = j.at("reduction_dimx").get<int>();
  graph.operators.clear();
  graph.smem_offset = 0;

  std::unordered_map<int, int> guid_mapping;
  auto get_tensor_from_guid = [&](int guid) {
    for (auto const &op : graph.operators) {
      for (auto const &tensor : op->output_tensors) {
        if (guid_mapping.at(tensor.guid) == guid) {
          return tensor;
        }
      }
    }
    assert(false);
  };

  for (json const &op : j["operators"]) {
    type::TBOperatorType op_type = op.at("op_type").get<type::TBOperatorType>();
    switch (op_type) {
      case type::TBOperatorType::TB_INPUT_OP: {
        STensor const &output =
            graph.new_input(op.at("dtensor").get<kernel::DTensor>(),
                            op.at("input_map").get<int3>(),
                            op.at("forloop_dim").get<int>(),
                            layout::SmemRowMajor);
        guid_mapping[output.guid] =
            op.at("output_tensors")[0].at("guid").get<int>();
        break;
      }
      case type::TBOperatorType::TB_OUTPUT_OP: {
        graph.mark_output(get_tensor_from_guid(
                              op.at("input_tensors")[0].at("guid").get<int>()),
                          op.at("output_map").get<int3>(),
                          -1,
                          type::TBEpilogueType::TB_EPILOGUE_NONE);
        break;
      }
      default:
        assert(false && "Unsupported operator");
    }
  }
}

} // namespace threadblock
} // namespace mirage
