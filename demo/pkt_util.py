from models.modeling_qwen3 import Qwen3ForCausalLM
from transformers import AutoTokenizer, AutoConfig
import torch
import torch.distributed as dist
import argparse
import os
import mirage as mi
import torch.nn.functional as F

class TestUtil:
    @staticmethod
    def create_matrix_arange_row(shape, dtype=torch.bfloat16, device='cuda'):
        M, N = shape
        row_indices = torch.arange(M, dtype=dtype, device=device)
        matrix = row_indices.unsqueeze(1).expand(M, N).contiguous()  # contiguous is very important!
        return matrix
    
    @staticmethod
    def create_matrix_arange_col(shape, dtype=torch.bfloat16, device='cuda'):
        M, N = shape
        col_indices = torch.arange(N, dtype=dtype, device=device)
        matrix = col_indices.unsqueeze(0).expand(M, N).contiguous()
        return matrix
class TorchRef:
    @staticmethod
    def compile_capture(fn, is_compile):
        if is_compile:
            compiled_ref_fn = torch.compile(fn, backend="inductor")
        else:
            compiled_ref_fn = fn
            
        for _ in range(20):
            output = compiled_ref_fn()
        torch.cuda.synchronize()
        
        graph = torch.cuda.CUDAGraph()
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            with torch.cuda.graph(graph):
                output = compiled_ref_fn()
        return graph, output
    
    @staticmethod
    def linear(x, w):
        return F.linear(x, w)
    
    @staticmethod
    def rms_norm(hidden_states, weight):
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance)
        return weight * hidden_states
    
    @staticmethod
    def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1] // 2
        return torch.nn.functional.silu(x[..., :d]) * x[..., d:]
        
    @staticmethod
    def mlp(x, w_gatedup, w_down_proj):
        O2 = TorchRef.linear(x, w_gatedup)
        O3 = TorchRef.silu_and_mul(O2)
        D  = TorchRef.linear(O3, w_down_proj)
        return D
    
    @staticmethod
    def norm_mlp(x, w_rms_norm, w_gatedup, w_down_proj):
        O1 = TorchRef.rms_norm(x, w_rms_norm)
        O2 = TorchRef.linear(O1, w_gatedup)
        O3 = TorchRef.silu_and_mul(O2)
        D  = TorchRef.linear(O3, w_down_proj)
        return D
    
    
class PersistentKernelTest:
    def __init__(self, world_size, rank, trace_name, profiling):
        if profiling:
            self.profiler_tensor = torch.zeros(
                3000 * 128, dtype=torch.uint64, device="cuda"
            ).contiguous()
        else:
            self.profiler_tensor = None
            
        num_workers, num_schedulers = mi.get_configurations_from_gpu(rank)
        print("num_workers: ", num_workers)
        print("num_schedulers: ", num_schedulers)
        
        self.mpk = mi.PersistentKernel(
            mode="offline",
            world_size=world_size,
            mpi_rank=rank,
            num_workers=num_workers,
            num_local_schedulers=num_schedulers,
            num_remote_schedulers=0,
            meta_tensors={}, #  meta_tensors={"qo_indptr_buffer": self.qo_indptr_buffer,},
            profiler_tensor=self.profiler_tensor,
            trace_name=trace_name,
            use_cutlass_kernel=False,
        )
    
    def get_mpk(self):
        return self.mpk

    def compile_load(self, is_no_compile, output_dir):
        if is_no_compile is True:
            module_path = output_dir + "/test.cpython-38-x86_64-linux-gnu.so"
            self.mpk.load_module(module_path)
        else:
            module_path = self.mpk.compile(output_dir=output_dir)
            print("module_path: ", module_path)
            self.mpk.load_module(module_path) 
            
    def memory_footprint_simulation(self, rank):
        torch.cuda.set_device(rank)
        with torch.device("cuda"):
            model_name = "/home/cjmcv/project/llm_models/Qwen/Qwen3-0.6B"
            self.model = Qwen3ForCausalLM.from_pretrained(model_name, world_size=1, max_num_pages=16, page_size=4096).to("cuda")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)    
            
    def torch_profile(self, func):
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            func()
        print(prof.key_averages().table(sort_by="cuda_time_total"))
        prof.export_chrome_trace("trace.json") # chrome://tracing/        
    
    def check_allclose(self, mpk_run, mpk_out, splitk, torch_out, iter, print_all):
        if (print_all):
            torch.set_printoptions(threshold=float('inf'))
        torch.cuda.synchronize()
        
        # print("inner: ", torch_out, torch_out.data_ptr())
        for _ in range(iter):
            mpk_out.zero_()
            mpk_run()
            # print("inner2: ", torch_out)
            torch.cuda.synchronize()
            
            mpk_result = mpk_out
            torch_result = torch_out
            # print("inner3: ", torch_out)
            if splitk != 1:
                for i in range(1, splitk):
                    mpk_out[0] += mpk_out[i]
                    
                mpk_result = mpk_out[0]
                torch_result = torch_out[0]
                total_num = torch_result.shape[0]
            else:
                mpk_result = mpk_out
                torch_result = torch_out
                total_num = torch_result.shape[0] * torch_result.shape[1]
                
            if (torch.allclose(mpk_result, torch_result, rtol=1e-2, atol=0)):
                print("allclose: True")
            else:
                print("mpk_out:", mpk_result.shape, "\n", mpk_result)
                print("torch_out:", torch_result.shape, "\n", torch_result)
                print("diff: ", mpk_result - torch_result)
                
                radio = abs((mpk_result - torch_result)/torch_result)
                
                threshold = [0.05, 0.10]
                count0 = (radio > threshold[0]).sum().item()
                count1 = (radio > threshold[1]).sum().item()
                print("radio > ", threshold[0], ": ", count0, "-", count0/total_num, " / ", threshold[1], ": ", count1, "-", count1/total_num)
                 
    def time_event_record(self, name, func, test_iter):
        starter = torch.cuda.Event(enable_timing=True)
        ender = torch.cuda.Event(enable_timing=True)
            
        starter.record()
        for _ in range(test_iter):
            func()
        ender.record()
        torch.cuda.synchronize()
        run_time = starter.elapsed_time(ender)
        print(name, "run time (ms): ", run_time / test_iter)
     
    def generate_report(self, mpk_run, mpk_out, splitk, torch_run, torch_out, warnup_iter, test_iter, allclose_iter, print_all):
        for _ in range(warnup_iter):
            torch_run()
        
        self.check_allclose(mpk_run, mpk_out, splitk, torch_out, allclose_iter, print_all)      
            
        self.time_event_record("mpk", mpk_run, test_iter)
        self.time_event_record("torch_ref", torch_run, test_iter)

        self.torch_profile(torch_run)
        self.torch_profile(mpk_run)


    # pushd build && make -j8 && popd
    
    # git clone --recursive https://www.github.com/mirage-project/mirage
    # pip install -e . -v
    # export MIRAGE_HOME=$(pwd)
    # python demo/fused_norm_mlp.py
    # --profiling https://ui.perfetto.dev/
    
    # nsys profile --trace=cuda,nvtx --output=my_nsys
    # ncu --set full --section "SpeedOfLight_RooflineChart" -k "persistent_kernel" -o my_profile python...
    # "kernel"
