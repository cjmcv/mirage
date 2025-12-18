from models.modeling_qwen3 import Qwen3ForCausalLM
from transformers import AutoTokenizer, AutoConfig
import torch
import torch.distributed as dist
import argparse
import os
import mirage as mi
import torch.nn.functional as F

class TorchRef:
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
    def __init__(self, world_size, rank, max_num_batched_requests, max_num_batched_tokens, trace_name, profiling):
        if profiling:
            self.profiler_tensor = torch.zeros(
                3000 * 128, dtype=torch.uint64, device="cuda"
            ).contiguous()
        else:
            self.profiler_tensor = None
            
        num_workers, num_schedulers = mi.get_configurations_from_gpu(rank)
        print("num_workers: ", num_workers)
        print("num_schedulers: ", num_schedulers)
        
        self.qo_indptr_buffer = torch.empty(
            max_num_batched_requests + 1, dtype=torch.int32, device="cuda")
        self.mpk = mi.PersistentKernel(
            mode="offline",
            world_size=world_size,
            mpi_rank=rank,
            num_workers=num_workers,
            num_local_schedulers=num_schedulers,
            num_remote_schedulers=0,
            max_num_batched_requests=max_num_batched_requests,
            max_num_batched_tokens=max_num_batched_tokens,
            meta_tensors={
                "qo_indptr_buffer": self.qo_indptr_buffer,
            },
            profiler_tensor=self.profiler_tensor,
            trace_name=trace_name,
            # spec_decode_config=spec_decode_config,
            use_cutlass_kernel=False,
        )
        self.batch_size = max_num_batched_tokens
    
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
    
    def check_allclose(self, mpk_run, mpk_out, splitk, torch_ref_run):
        # torch.set_printoptions(threshold=float('inf'))
        torch.cuda.synchronize()
        
        torch_out = torch_ref_run()
        for _ in range(5):
            mpk_out.zero_()
            mpk_run()
            torch.cuda.synchronize()
            
            mpk_result = mpk_out
            torch_result = torch_out
            
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
                print("mpk_out:", mpk_result)
                print("torch_out:", torch_result)
                print("diff: ", mpk_result - torch_result)
                
                threshold = 0.05
                radio_num = abs((mpk_result - torch_result)/torch_result) > threshold
                count = radio_num.sum().item()
                print("radio > ", threshold, ": ", count, "-", count/total_num)
                
                
                
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
     
     
    ##########################################################
    # pushd build && make -j8 && popd
    
    # git clone --recursive https://www.github.com/mirage-project/mirage
    # pip install -e . -v
    # export MIRAGE_HOME=$(pwd)
    # python demo/fused_norm_mlp.py
    # --profiling https://ui.perfetto.dev/
    
    # nsys profile --trace=cuda,nvtx --output=my_nsys
    # ncu --set full --section "SpeedOfLight_RooflineChart" -k "persistent_kernel" -o my_profile python...
    # "kernel"
