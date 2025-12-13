# Copyright 2024 CMU
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

ctypedef unsigned long int size_t

cdef extern from "mirage/vector_types.h":
    ctypedef struct dim3:
        unsigned int x
        unsigned int y
        unsigned int z
    ctypedef struct int3:
        int x
        int y
        int z

cdef extern from "mirage/type.h" namespace "mirage::type":
    # This must be consistent with mirage/type.h
    cdef enum DataType:
        DT_FLOAT4 = 920,
        DT_INT4 = 925,
        DT_UINT4 = 926,
        DT_FLOAT8 = 930,
        DT_INT8 = 935,
        DT_UINT8 = 936,
        DT_FLOAT16 = 940,
        DT_BFLOAT16 = 941,
        DT_INT16 = 945,
        DT_UINT16 = 946,
        DT_FLOAT32 = 950,
        DT_INT32 = 955,
        DT_UINT32 = 956,
        DT_DOUBLE = 960,
        DT_INT64 = 965,
        DT_UINT64 = 966,
        DT_UNKNOWN = 999,
    cdef enum TBEpilogueType:
        TB_EPILOGUE_NONE = 3100,
        TB_EPILOGUE_ALLREDUCE = 3101,
        TB_EPILOGUE_ALLTOALL = 3102,
        TB_EPILOGUE_INVALID = 3199,
    cdef enum KNOperatorType:
        KN_UNKOWN = 1000,
        KN_INPUT_OP = 1001,
        KN_OUTPUT_OP = 1002,
        KN_CUSTOMIZED_OP = 1999,
    cdef enum TBOperatorType:
        TB_UNKOWN = 2000,
        TB_INPUT_OP = 2001,
        TB_OUTPUT_OP = 2002,
        TB_CUSTOMIZED_OP = 2999

cdef extern from "mirage/layout.h" namespace "mirage::layout":
    # This must be consistent with mirage/layout.h
    cdef enum DmemLayout:
        DmemRowMajor = 100,
        DmemColumnMajor = 101,
        DmemUnknownLayout = 199,
    cdef enum SmemLayout:
        SmemRowMajor = 200,
        SmemColumnMajor = 201,
        SmemUnknownLayout = 299

cdef cppclass CppTBGraph "mirage::threadblock::Graph"

cdef extern from "mirage/kernel/device_tensor.h" namespace "mirage::kernel":
    cdef struct CppDTensor "mirage::kernel::DTensor":
        DataType data_type
        DmemLayout layout
        int num_dims
        int dim[4]
        size_t guid
        #KNOperator *owner_op
        #void *data_ptr
        int owner_ts_idx

cdef extern from "mirage/kernel/runtime.h" namespace "mirage::runtime":
    ctypedef struct TaskGraphResult:
        string cuda_code
        string json_file

cdef extern from "mirage/kernel/graph.h" namespace "mirage::kernel":

    cdef cppclass CppKNOperator "mirage::kernel::KNOperator":
        KNOperatorType op_type
        vector[CppDTensor] input_tensors
        vector[CppDTensor] output_tensors
        int get_input_dtensors(CppDTensor** cinputs)
        int get_output_dtensors(CppDTensor** cinputs)
 
    cdef cppclass CppKNCustomizedOp "mirage::kernel::KNCustomizedOp"(CppKNOperator):
        CppTBGraph bgraph
        void get_bgraph(CppTBGraph** bgraph)

    cdef cppclass CppKNGraph "mirage::kernel::Graph":
        CppKNGraph(dim3 gpu_dim)
        CppDTensor* new_input_ptr(vector[int] dims,
                                  vector[size_t] strides,
                                  DataType data_type,
                                  DmemLayout layout)
        # void mark_output(const CppDTensor* A, vector[size_t] strides)
        int customized(vector[const CppDTensor*] inputs,
                       CppDTensor** outputs,
                       CppTBGraph* bgraph)
        int get_num_input_dtensors()
        # int get_num_output_dtensors()
        int get_input_dtensors(CppDTensor** cinputs)
        int get_input_dtensor_shape_and_stride(const CppDTensor *input, int *strides, int *dims)
        # void generate_triton_program(const char *filepath)
        # void generate_cuda_program(const char *filepath)
        size_t get_owner_independent_hash() const
        # Persistent kernel functions
        void attach_torch_tensor(const CppDTensor *input,
                                 void *torch_data_ptr,
                                 const char *name)
        void attach_cuda_tensor(const CppDTensor *input,
                                const char *name)
        void attach_nvshmem_tensor(const CppDTensor *input,
                                   const char *name)
        CppDTensor* fuse_tensors(vector[const CppDTensor*] inputs,
                                 int fused_dim,
                                 int num_groups,
                                 const char *name)
        CppDTensor* shuffle_tensors(vector[const CppDTensor*] inputs,
                                 int shuffled_dim,
                                 int num_groups,
                                 const char *name)
        void register_task(const char *task_type,
                           vector[int] params)
        TaskGraphResult generate_task_graph(int num_gpus, int my_gpu_id)

        vector[CppKNOperator*] operators

cdef extern from "mirage/threadblock/graph.h" namespace "mirage::threadblock":
    ctypedef struct CppSTensor "mirage::threadblock::STensor":
        DataType data_type
        SmemLayout layout
        int num_dims
        int dim[4]
        int owner_ts_idx
        size_t guid
    
    cdef cppclass CppTBOperator "mirage::threadblock::TBOperator":
        TBOperatorType op_type
        vector[CppSTensor] input_tensors
        vector[CppSTensor] output_tensors
        int get_input_stensors(CppSTensor** cinputs)
        int get_output_stensors(CppSTensor** cinputs)

    cdef cppclass CppTBInputOp "mirage::threadblock::TBInputOp"(CppTBOperator):
        int forloop_dim
        int3 input_map
        size_t get_dtensor_guid()

    cdef cppclass CppTBOutputOp "mirage::threadblock::TBOutputOp"(CppTBOperator):
        int forloop_dim
        int3 output_map
        size_t get_dtensor_guid()

    cdef cppclass CppTBGraph "mirage::threadblock::Graph":
        CppTBGraph(dim3 grid_dim,
                   dim3 block_dim,
                   int forloop_range,
                   int reduction_dimx)

        CppSTensor* new_input(const CppDTensor* dtensor,
                             int3 input_map,
                             int forloop_dim,
                             SmemLayout layout,
                             bool store_in_dmem)
        CppDTensor* new_output(const CppSTensor* stensor,
                            int3 output_map,
                            int forloop_dim,
                            TBEpilogueType epilogue)
        # CppSTensor* matmul(const CppSTensor *A,
        #                 const CppSTensor *B)
        # CppSTensor* exp(const CppSTensor *A)
        # CppSTensor* silu(const CppSTensor *A)
        # CppSTensor* gelu(const CppSTensor *A)
        # CppSTensor* relu(const CppSTensor *A)
        # CppSTensor* clamp(const CppSTensor *A, float min_val, float max_val)
        # CppSTensor* square(const CppSTensor *A)
        # CppSTensor* sqrt(const CppSTensor *A)
        # CppSTensor* mul_scalar(const CppSTensor *A, float scalar)
        # CppSTensor* add(const CppSTensor *A,
        #              const CppSTensor *B)
        # CppSTensor* mul(const CppSTensor *A,
        #              const CppSTensor *B)
        # CppSTensor* div(const CppSTensor *A,
        #              const CppSTensor *B)
        # CppSTensor* sub(const CppSTensor *A,
        #              const CppSTensor *B)
        # CppSTensor* reduction(const CppSTensor *A, int dim)
        # vector[CppSTensor*] reduction_max(const CppSTensor *A, int dim)
        # CppSTensor* rms_norm(const CppSTensor *A)
        # CppSTensor* concat(const CppSTensor *A,
        #                 const CppSTensor *B,
        #                 int dim)
        # CppSTensor* forloop_accum(const CppSTensor *A,
        #                        TBOperatorType optype)
        # CppSTensor* forloop_accum_rescale(const CppSTensor *A,
        #                        const CppSTensor *B,
        #                        TBOperatorType optype)
        # CppSTensor* forloop_accum_max(const CppSTensor *A)
        dim3 grid_dim
        dim3 block_dim
        int forloop_range
        int reduction_dimx
        vector[CppTBOperator*] operators

# cdef extern from "mirage/search/search_c.h" namespace "mirage::search_c":
#     ctypedef struct MInt3:
#         int x
#         int y
#         int z
#     ctypedef struct MDim3:
#         unsigned int x
#         unsigned int y
#         unsigned int z

#     cdef int cython_search(const CppKNGraph *input_graph,
#                            const char *backend,
#                            int max_num_new_graphs,
#                            CppKNGraph** new_graphs,
#                            vector[MInt3] imaps,
#                            vector[MInt3] omaps,
#                            vector[MDim3] griddims,
#                            vector[MDim3] blockdims,
#                            vector[int] fmaps,
#                            vector[int] franges,
#                            const char * filename,
#                            bool verbose,
#                            const char * default_config,
#                            bool is_formal_verified)
    
#     cdef void cython_to_json(const CppKNGraph *input_graph,
#                              const char *filename)
#     cdef CppKNGraph *cython_from_json(const char *filename)

# cdef extern from "mirage/transpiler/transpile.h" namespace "mirage::transpiler":
#     ctypedef struct TranspilerConfig:
#         int target_cc
#         int num_consumer_wgs
#         int num_producer_wgs;
#         int pipeline_stages;
#         bool profiling;
#         bool enable_online_softmax;
#     ctypedef struct OutputTensorDirective:
#         size_t alloc_size
#         vector[int] shape
#         vector[size_t] strides
#     ctypedef struct TranspileResult:
#         string code
#         size_t buf_size
#         size_t max_smem_size
#         size_t profiler_buf_size
#         vector[OutputTensorDirective] output_directives
#     cdef TranspileResult transpile(const CppKNGraph *graph,
#                        const TranspilerConfig config,
#                        vector[vector[size_t]] input_strides)

# cdef extern from "mirage/nki_transpiler/transpile.h" namespace "mirage::nki_transpiler":
#     ctypedef struct NKITranspilerConfig:
#         int target_cc
#     ctypedef struct NKIErrorInfo:
#         vector[string] errors
#     ctypedef struct NKITranspileResult:
#         string code
#         NKIErrorInfo error_state
#     cdef NKITranspileResult transpile(const CppKNGraph *graph,
#                                       const NKITranspilerConfig config)

# cdef extern from "mirage/triton_transpiler/transpile.h" namespace "mirage::triton_transpiler":
#     ctypedef struct TritonTranspilerConfig:
#         int target_cc
#     ctypedef struct TritonTranspileResult:
#         string code
#         vector[vector[int]] output_shapes
#     cdef TritonTranspileResult transpile(const CppKNGraph *graph,
#                                          const TritonTranspilerConfig config)

# cdef extern from "mirage/kernel/device_memory_manager.h" namespace "mirage::kernel":
#     cdef int cython_set_gpu_device_id(int gpu_id)
