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

from cpython cimport array
import ctypes
import array
import numpy as np
import torch
from libcpp.memory cimport shared_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool

# Code snippet from OpenAI Triton

ctypedef unsigned long int size_t

cdef extern from "mirage/type.h":
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
    cdef enum KNOperatorType:
        KN_UNKOWN = 1000,
        KN_INPUT_OP = 1001,
        KN_CUSTOMIZED_OP = 1999,
    cdef enum TBOperatorType:
        TB_UNKOWN = 2000,
        TB_INPUT_OP = 2001,
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

# cdef cppclass CppTBGraph "mirage::threadblock::Graph"

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
        int customized(vector[const CppDTensor*] inputs,
                       CppDTensor** outputs,
                       CppTBGraph* bgraph)
        int get_num_input_dtensors()
        int get_input_dtensors(CppDTensor** cinputs)
        int get_input_dtensor_shape_and_stride(const CppDTensor *input, int *strides, int *dims)
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
        int3 input_map
        size_t get_dtensor_guid()

    cdef cppclass CppTBGraph "mirage::threadblock::Graph":
        CppTBGraph(dim3 grid_dim,
                   dim3 block_dim,
                   int thread_num,
                   int reduction_dimx)

        CppSTensor* new_input(const CppDTensor* dtensor,
                             int3 input_map,
                             SmemLayout layout,
                             bool store_in_dmem)

        dim3 grid_dim
        dim3 block_dim
        int thread_num
        int reduction_dimx
        vector[CppTBOperator*] operators

################################################################

class dtype:
    SINT_TYPES = ['int8', 'int16', 'int32', 'int64']
    UINT_TYPES = ['uint8', 'uint16', 'uint32', 'uint64']
    FP_TYPES = ['fp16', 'bf16', 'fp32', 'fp64']

    def __init__(self, name):
        self.name = name
        assert name in dtype.SINT_TYPES + dtype.UINT_TYPES + dtype.FP_TYPES, name

    def is_fp16(self):
        return self.name == 'fp16'

    def is_bf16(self):
        return self.name == 'bf16'

    def is_fp32(self):
        return self.name == 'fp32'

    def is_fp64(self):
        return self.name == 'fp64'

    def is_int1(self):
        return self.name == 'int1'

    def is_int8(self):
        return self.name == 'int8'

    def is_int16(self):
        return self.name == 'int16'

    def is_int32(self):
        return self.name == 'int32'

    def is_int64(self):
        return self.name == 'int64'

    def is_uint8(self):
        return self.name == 'uint8'

    def is_uint16(self):
        return self.name == 'uint16'

    def is_uint32(self):
        return self.name == 'uint32'

    def is_uint64(self):
        return self.name == 'uint64'

    def __eq__(self, other: dtype):
        if not isinstance(other, dtype):
            return False
        return self.name == other.name

    def __ne__(self, other: dtype):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.name, ))

    def __str__(self):
        return self.name

    def is_dtype(type_str):
        return type_str in dtype.SINT_TYPES + dtype.UINT_TYPES + dtype.FP_TYPES

# data types
int8 = dtype('int8')
int16 = dtype('int16')
int32 = dtype('int32')
int64 = dtype('int64')
uint8 = dtype('uint8')
uint16 = dtype('uint16')
uint32 = dtype('uint32')
uint64 = dtype('uint64')
float16 = dtype('fp16')
bfloat16 = dtype('bf16')
float32 = dtype('fp32')
float64 = dtype('fp64')

def get_kn_operator_type_string(int op_type):
    if op_type == KN_UNKOWN:
        return "kn_unknown"
    elif op_type == KN_INPUT_OP:
        return "kn_input_op"
    elif op_type == KN_CUSTOMIZED_OP:
        return "kn_customized_op"
    else:
        return "unknown_op_type" + str(op_type)


def get_tb_operator_type_string(int op_type):
    if op_type == TB_UNKOWN:
        return "tb_unknown"
    elif op_type == TB_INPUT_OP:
        return "tb_input_op"
    elif op_type == TB_CUSTOMIZED_OP:
        return "tb_customized_op"
    else:
        return "unknown_op_type" + str(op_type)


def convert_dtype_to_ctype(type : dtype):
    if type.is_int8():
        return DT_INT8
    elif type.is_uint16():
        return DT_UINT16
    elif type.is_fp16():
        return DT_FLOAT16
    elif type.is_bf16():
        return DT_BFLOAT16
    elif type.is_fp32():
        return DT_FLOAT32
    elif type.is_int32():
        return DT_INT32
    elif type.is_int64():
        return DT_INT64
    elif type.is_fp64():
        return DT_DOUBLE
    else:
        raise RuntimeError(f"Unsupported dtype: {dtype}")

def convert_dtype_to_torch_type(type : dtype):
    if type.is_int8():
        return torch.int8
    elif type.is_uint16():
        return torch.uint16
    elif type.is_fp16():
        return torch.float16
    elif type.is_bf16():
        return torch.bfloat16
    elif type.is_int32():
        return torch.int32
    elif type.is_fp32():
        return torch.float32
    elif type.is_int64():
        return torch.int64
    elif type.is_fp64():
        return torch.float64
    else:
        assert False, "Unsupported dtype: {}".format(type)

def convert_ctype_to_dtype(type):
    if type == DT_INT8:
        return int8
    elif type == DT_UINT16:
        return uint16
    elif type == DT_FLOAT16:
        return float16
    elif type == DT_BFLOAT16:
        return bfloat16
    elif type == DT_INT32:
        return int32
    elif type == DT_FLOAT32:
        return float32
    elif type == DT_DOUBLE:
        return float64
    else:
        return None

def convert_torch_type_to_dtype(type):
    if type is torch.int8:
        return int8
    elif type is torch.uint16:
        return uint16
    elif type is torch.int32:
        return int32
    elif type is torch.float16:
        return float16
    elif type is torch.bfloat16:
        return bfloat16
    elif type is torch.float32:
        return float32
    elif type is torch.int64:
        return int64
    elif type is torch.float64:
        return float64
    else:
        raise RuntimeError(f"Unsupported dtype: {type}")


cdef class DTensor:
    cdef CppDTensor* c_ptr # Hold a Tensor instance

    cdef inline _set_tensor(self, tensor):
        cdef unsigned long long ptr
        if tensor is None:
            self.c_ptr = <CppDTensor*>(NULL)
        else:
            ptr = ctypes.cast(tensor, ctypes.c_void_p).value
            self.c_ptr = <CppDTensor*>(ptr)

    property guid:
        def __get__(self):
            if self.c_ptr == NULL:
                return None
            else:
                return self.c_ptr.guid

    property tensor:
        def __get__(self):
            if self.c_ptr == NULL:
                return None
            else:
                return ctypes.cast(<unsigned long long>self.c_ptr, ctypes.c_void_p)
        
        def __set__(self, value):
            self._set_tensor(value)

    property num_dims:
        def __get__(self):
            if self.c_ptr == NULL:
                print("Error: tensor is None in num_dims property")
                return None
            else:
                return self.c_ptr.num_dims

    property dtype:
        def __get__(self):
            if self.c_ptr == NULL:
                return None
            else:
                return convert_ctype_to_dtype(self.c_ptr.data_type)

    def __cinit__(self, tensor):
        self._set_tensor(tensor)

    def dim(self, int idx):
        if (idx < self.c_ptr.num_dims):
            return self.c_ptr.dim[idx]
        else:
            assert False , "Error: index out of range"
            return None

cdef class STensor:
    cdef CppSTensor* c_ptr # Hold a CppSTensor instance

    cdef inline _set_tensor(self, tensor):
        cdef unsigned long long ptr
        if tensor is None:
            self.c_ptr = <CppSTensor*>(NULL)
        else:
            ptr = ctypes.cast(tensor, ctypes.c_void_p).value
            self.c_ptr = <CppSTensor*>(ptr)
    property guid:
        def __get__(self):
            if self.c_ptr == NULL:
                return None
            else:
                return self.c_ptr.guid
    property tensor:
        def __get__(self):
            if self.c_ptr == NULL:
                return None
            else:
                return ctypes.cast(<unsigned long long>self.c_ptr, ctypes.c_void_p)
        
        def __set__(self, value):
            self._set_tensor(value)

    property num_dims:
        def __get__(self):
            if self.c_ptr == NULL:
                return None
            else:
                return self.c_ptr.num_dims

    property dtype:
        def __get__(self):
            if self.c_ptr == NULL:
                return None
            else:
                return convert_ctype_to_dtype(self.c_ptr.data_type)

    def __cinit__(self, tensor):
        self._set_tensor(tensor)

    def dim(self, int idx):  
        if (idx < self.c_ptr.num_dims):
            return self.c_ptr.dim[idx]
        else:
            assert False , "Error: index out of range"
            return None

cdef class CyKNOperator:
    cdef CppKNOperator* c_ptr # Hold a CppKNOperator instance

    cdef inline _set_operator(self, op):
        cdef unsigned long long ptr
        if op is None:
            self.c_ptr = <CppKNOperator*>(NULL)
        else:
            ptr = ctypes.cast(op, ctypes.c_void_p).value
            self.c_ptr = <CppKNOperator*>(ptr)
    
    def get_input_dtensors(self):
        cdef CppDTensor* cinputs[1024]
        num = self.c_ptr.get_input_dtensors(cinputs)
        inputs = list()
        for i in range(num):
            ptr = ctypes.cast(<unsigned long long>cinputs[i], ctypes.c_void_p)
            inputs.append(DTensor(ptr))
        return inputs

    def get_output_dtensors(self):
        cdef CppDTensor* coutputs[1024]
        num = self.c_ptr.get_output_dtensors(coutputs)
        outputs = list()
        for i in range(num):
            ptr = ctypes.cast(<unsigned long long>coutputs[i], ctypes.c_void_p)
            outputs.append(DTensor(ptr))
        return outputs

    property op_type:
        def __get__(self):
            if self.c_ptr == NULL:
                return None
            else:
                return get_kn_operator_type_string(int(self.c_ptr.op_type))

    def __cinit__(self, op):
        self._set_operator(op)

cdef class CyKNCustomizedOp(CyKNOperator):
    cdef CppKNCustomizedOp* c_customized_ptr

    def __cinit__(self, op):
        cdef unsigned long long ptr
        if op is None:
            self.c_customized_ptr = <CppKNCustomizedOp*>(NULL)
        else:
            ptr = ctypes.cast(op, ctypes.c_void_p).value
            self.c_customized_ptr = <CppKNCustomizedOp*>(ptr)

    def get_bgraph(self):
        cdef CppTBGraph* bgraph
        self.c_customized_ptr.get_bgraph(&bgraph)

        ptr = ctypes.cast(<unsigned long long>bgraph, ctypes.c_void_p)
        cybgraph = CyTBGraph(bgraph = ptr)
        return cybgraph

cdef class CyTBOperator:
    cdef CppTBOperator* c_ptr # Hold a CppTBOperator instance

    cdef inline _set_operator(self, op):
        cdef unsigned long long ptr
        if op is None:
            self.c_ptr = <CppTBOperator*>(NULL)
        else:
            ptr = ctypes.cast(op, ctypes.c_void_p).value
            self.c_ptr = <CppTBOperator*>(ptr)

    def get_input_stensors(self):
        cdef CppSTensor* cinputs[1024]
        num = self.c_ptr.get_input_stensors(cinputs)
        inputs = list()
        for i in range(num):
            ptr = ctypes.cast(<unsigned long long>cinputs[i], ctypes.c_void_p)
            inputs.append(STensor(ptr))
        return inputs

    def get_output_stensors(self):
        cdef CppSTensor* coutputs[1024]
        num = self.c_ptr.get_output_stensors(coutputs)
        outputs = list()
        for i in range(num):
            ptr = ctypes.cast(<unsigned long long>coutputs[i], ctypes.c_void_p)
            outputs.append(STensor(ptr))
        return outputs

    property op_type:
        def __get__(self):
            if self.c_ptr == NULL:
                return None
            else:
                return get_tb_operator_type_string(int(self.c_ptr.op_type))

    def __cinit__(self, op):
        self._set_operator(op)

cdef class CyTBInputOp(CyTBOperator):
    cdef CppTBInputOp* c_input_ptr

    def __cinit__(self, op):
        cdef unsigned long long ptr
        if op is None:
            self.c_input_ptr = <CppTBInputOp*>(NULL)
        else:
            ptr = ctypes.cast(op, ctypes.c_void_p).value
            self.c_input_ptr = <CppTBInputOp*>(ptr)

    property input_map:
        def __get__(self):
            if self.c_input_ptr == NULL:
                return None
            else:
                return {
                    "x": self.c_input_ptr.input_map.x,
                    "y": self.c_input_ptr.input_map.y,
                    "z": self.c_input_ptr.input_map.z
                }

    property dtensor_guid:
        def __get__(self):
            if self.c_input_ptr == NULL:
                return None
            else:
                return self.c_input_ptr.get_dtensor_guid()

cdef class CyKNGraph:
    cdef CppKNGraph *p_kgraph #Hold a CppKNGraph instance

    def __cinit__(self, graph = None):
        cdef unsigned long long ptr
        cdef dim3 c_gpu_dim
        if graph is None:
            c_gpu_dim.x = 1
            c_gpu_dim.y = 1
            c_gpu_dim.z = 1
            self.p_kgraph = new CppKNGraph(c_gpu_dim)
        else:
            ptr = ctypes.cast(graph, ctypes.c_void_p).value
            self.p_kgraph = <CppKNGraph*>(ptr)

    def new_input(self, tuple dims, tuple strides, dtype : dtype = float16):
        cdef vector[int] cdims
        cdef vector[size_t] cstrides
        cdims.resize(len(dims))
        for i in range(len(dims)):
            cdims[i] = dims[i]
        cstrides.resize(len(strides))
        for i in range(len(strides)):
            cstrides[i] = strides[i]

        c_type = convert_dtype_to_ctype(dtype)
        cdef CppDTensor* ptr = self.p_kgraph.new_input_ptr(cdims, cstrides, c_type, DmemRowMajor)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return DTensor(t)

    def customized(self, list inputs, CyTBGraph bgraph):
        cdef vector[const CppDTensor*] cinputs
        cinputs.resize(len(inputs))
        cdef DTensor t
        for i in range(len(inputs)):
            if inputs[i] is None:
                cinputs[i] = NULL
            else:
                assert (type(inputs[i]) == DTensor)
                t = inputs[i]
                cinputs[i] = t.c_ptr
        cdef CppDTensor* coutputs[1024]
        num_outputs = self.p_kgraph.customized(cinputs, coutputs, bgraph.p_bgraph)
        outputs = list()
        for i in range(num_outputs):
            ptr = ctypes.cast(<unsigned long long>coutputs[i], ctypes.c_void_p)
            outputs.append(DTensor(ptr))
        return outputs

    def get_input_dtensors(self):
        cdef CppDTensor* cinputs[1024]
        num = self.p_kgraph.get_input_dtensors(cinputs)
        inputs = list()
        for i in range(num):
            ptr = ctypes.cast(<unsigned long long>cinputs[i], ctypes.c_void_p)
            inputs.append(DTensor(ptr))
        return inputs
    
    # visualizer utils

    def _kn_tensor_to_dict(self, DTensor t):
        return {
            "num_dims": t.num_dims,
            "dim": [t.dim(i) for i in range(t.num_dims)],
            "guid": t.guid
        }

    def _tb_tensor_to_dict(self, STensor t):
        return {
            "num_dims": t.num_dims,
            "dim": [t.dim(i) for i in range(t.num_dims)],
            "guid": t.guid
        }

    def _get_tb_operator_info(self, CyTBOperator op):
        ans = {
            "op_type": op.op_type,
            "input_tensors": [self._tb_tensor_to_dict(t) for t in op.get_input_stensors()],
            "output_tensors": [self._tb_tensor_to_dict(t) for t in op.get_output_stensors()],
        }
        if "input" in op.op_type:
            input_op = CyTBInputOp(ctypes.cast(<unsigned long long>(op.c_ptr), ctypes.c_void_p))
            ans["input_map"] = input_op.input_map
            ans["dtensor"] = {
                "guid": input_op.dtensor_guid
            }
        return ans

    def _get_bgraph_info(self, CyKNOperator op):
        cop = CyKNCustomizedOp(ctypes.cast(<unsigned long long>(op.c_ptr), ctypes.c_void_p))
        bgraph = cop.get_bgraph()
        return {
            "grid_dim": bgraph.grid_dim,
            "thread_num": bgraph.thread_num,
            "operators": [self._get_tb_operator_info(i) for i in bgraph.operators]
        }

    def _get_kn_operator_info(self, CyKNOperator op):
        if op.op_type == "kn_customized_op":
            return {
                "op_type": op.op_type,
                "input_tensors": [self._kn_tensor_to_dict(t) for t in op.get_input_dtensors()],
                "output_tensors": [self._kn_tensor_to_dict(t) for t in op.get_output_dtensors()],
                "bgraph": self._get_bgraph_info(op)
            }
        else:
            return {
                "op_type": op.op_type,
                "input_tensors": [self._kn_tensor_to_dict(t) for t in op.get_input_dtensors()],
                "output_tensors": [self._kn_tensor_to_dict(t) for t in op.get_output_dtensors()],
            }

    def get_graph_structure(self):
        operators = []
        ops = self.p_kgraph.operators
        for i in range(ops.size()):
            op = CyKNOperator(None)
            op.c_ptr = ops[i]
            operators.append(self._get_kn_operator_info(op))
        return operators

    def get_num_inputs(self):
        return self.p_kgraph.get_num_input_dtensors()

    def get_input_dtensor_shape_and_stride(self, DTensor A):
        cdef int cstrides[128]
        cdef int cdims[128]
        num = self.p_kgraph.get_input_dtensor_shape_and_stride(A.c_ptr, cstrides, cdims)
        strides = list()
        dims = list()
        for i in range(num):
            strides.append(cstrides[i])
            dims.append(cdims[i])
        return tuple(dims), tuple(strides)

    # Functions for ersistent kernels
    def attach_torch_tensor(self, DTensor tensor, torch_tensor, str name):
        # cdef unsigned long long torch_data_ptr = ctypes.cast(torch_tensor.data_ptr(), ctypes.c_void_p).value
        cdef unsigned long long torch_data_ptr = <unsigned long long>(torch_tensor.data_ptr())
        cdef char* cname = NULL
        if name is not None:
            py_byte_string = name.encode('UTF-8')
            cname = py_byte_string
        self.p_kgraph.attach_torch_tensor(tensor.c_ptr, <void *>torch_data_ptr, cname)

    def attach_cuda_tensor(self, DTensor tensor, str name):
        cdef char* cname = NULL
        if name is not None:
            py_byte_string = name.encode('UTF-8')
            cname = py_byte_string
        self.p_kgraph.attach_cuda_tensor(tensor.c_ptr, cname)

    def attach_nvshmem_tensor(self, DTensor tensor, str name):
        cdef char* cname = NULL
        if name is not None:
            py_byte_string = name.encode('UTF-8')
            cname = py_byte_string
        self.p_kgraph.attach_nvshmem_tensor(tensor.c_ptr, cname)

    def fuse_tensors(self, list[DTensor] inputs, int fused_dim, int num_groups, str name):
        cdef vector[const CppDTensor*] cinputs
        cinputs.resize(len(inputs))
        cdef DTensor t
        for i in range(len(inputs)):
            assert(type(inputs[i]) == DTensor)
            t = inputs[i]
            cinputs[i] = t.c_ptr
        cdef char* cname = NULL
        if name is not None:
            py_byte_string = name.encode('UTF-8')
            cname = py_byte_string
        cdef CppDTensor* ptr = self.p_kgraph.fuse_tensors(cinputs, fused_dim, num_groups, cname)
        output = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return DTensor(output)

    def shuffle_tensors(self, list[DTensor] inputs, int shuffled_dim, int num_groups, str name):
        cdef vector[const CppDTensor*] cinputs
        cinputs.resize(len(inputs))
        cdef DTensor t
        for i in range(len(inputs)):
            assert(type(inputs[i]) == DTensor)
            t = inputs[i]
            cinputs[i] = t.c_ptr
        cdef char* cname = NULL
        if name is not None:
            py_byte_string = name.encode('UTF-8')
            cname = py_byte_string
        cdef CppDTensor* ptr = self.p_kgraph.shuffle_tensors(cinputs, shuffled_dim, num_groups, cname)
        output = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return DTensor(output)


    def register_task(self, CyTBGraph bgraph, str task_type, list[int] params):
        cdef char* cname = NULL
        if task_type is not None:
            py_byte_string = task_type.encode('UTF-8')
            cname = py_byte_string
        cdef vector[int] cparams
        cparams.resize(0)
        if params is not None:
            cparams.resize(len(params))
            for i in range(len(params)):
                cparams[i] = params[i]
        self.p_kgraph.register_task(cname, cparams)

    def generate_task_graph(self, int num_gpus, int my_gpu_id):
        cdef TaskGraphResult result = self.p_kgraph.generate_task_graph(num_gpus, my_gpu_id)
        return {
            "cuda_code": result.cuda_code.decode("UTF-8"),
            "json_file": result.json_file.decode("UTF-8"),
        }
     

cdef class CyTBGraph:
    cdef CppTBGraph *p_bgraph #Hold a CppTBGraph instance

    def __cinit__(self, tuple grid_dim = (), tuple block_dim = (), int thread_num = 128, int dimx = -1, bgraph = None):
        cdef unsigned long long ptr
        cdef dim3 c_grid_dim
        cdef dim3 c_block_dim
        if bgraph is None:
            if len(grid_dim) == 0 or len(block_dim) == 0 or dimx == -1:
                assert False, "grid_dim, block_dim, thread_num, dimx must be provided"
            assert len(grid_dim) == 3, "grid_dim must include 3 dimensions"
            assert len(block_dim) == 3, "block_dim must include 3 dimensions"
            c_grid_dim.x = grid_dim[0]
            c_grid_dim.y = grid_dim[1]
            c_grid_dim.z = grid_dim[2]
            c_block_dim.x = block_dim[0]
            c_block_dim.y = block_dim[1]
            c_block_dim.z = block_dim[2]
            self.p_bgraph = new CppTBGraph(c_grid_dim, c_block_dim, thread_num, dimx)
        else:
            ptr = ctypes.cast(bgraph, ctypes.c_void_p).value
            if isinstance(bgraph, int):
                self.p_bgraph = <CppTBGraph*>(ptr)
            elif isinstance(bgraph, ctypes.c_void_p):
                self.p_bgraph = <CppTBGraph*>(ptr)
            else:
                assert False, "bgraph must be an integer or ctypes.c_void_p, but got " + str(type(bgraph))
    
    def new_input(self, DTensor dtensor, tuple input_map, bool store_in_dmem = False):
        assert len(input_map) == 3, "input_map must be of length 3"
        cdef int3 c_input_map
        c_input_map.x = input_map[0]
        c_input_map.y = input_map[1]
        c_input_map.z = input_map[2]
        cdef CppDTensor* dtensor_cptr = NULL
        if dtensor is not None:
            dtensor_cptr = dtensor.c_ptr
        cdef CppSTensor* ptr = self.p_bgraph.new_input(dtensor_cptr, c_input_map, SmemRowMajor, store_in_dmem)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        return STensor(t)

    property grid_dim:
        def __get__(self):
            return {
                "x": self.p_bgraph.grid_dim.x,
                "y": self.p_bgraph.grid_dim.y,
                "z": self.p_bgraph.grid_dim.z
            }

    property thread_num:
        def __get__(self):
            return self.p_bgraph.thread_num

    property operators:
        def __get__(self):
            cdef vector[CppTBOperator*] coperators
            coperators = self.p_bgraph.operators
            operators = list()
            for i in range(coperators.size()):
                ptr = ctypes.cast(<unsigned long long>coperators[i], ctypes.c_void_p)
                operators.append(CyTBOperator(ptr))
            return operators
