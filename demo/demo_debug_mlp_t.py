from models.modeling_qwen3 import Qwen3ForCausalLM
from transformers import AutoTokenizer, AutoConfig
from safetensors.torch import load_model
import torch
import torch.distributed as dist
import argparse
import os
import mirage as mi

# print limitation
torch.set_printoptions(profile="full")

def silu_and_mul(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.nn.functional.silu(x[..., :d]) * x[..., d:]
    
def test_torch_mlp2(x, w_gatedup, w_down_proj):
    import torch.nn.functional as F
    O1 = F.linear(x, w_gatedup)
    D = silu_and_mul(O1)
    return F.linear(D, w_down_proj)
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-mirage", action="store_true", help="Use Mirage kernels")
    parser.add_argument("--max-num-batched-tokens", default=1, type=int, help="Max number of tokens in a batch")
    parser.add_argument("--max-num-batched-requests", default=1, type=int, help="Max number of requests in a batch")
    parser.add_argument("--page-size", default=4096, type=int, help="Page size")
    parser.add_argument("--max-num-pages", default=16, type=int, help="Max num pages")
    parser.add_argument("--output-dir", default="./gen", help="Output files directory")
    parser.add_argument("--trace-name", default="qwen3", help="Perfetto trace output name")
    parser.add_argument("--profiling", action="store_true", help="Use Profiler to generate trace")
    
    # lookahead or promptlookup
    parser.add_argument(
        "--spec-decode",
        default=None,
        choices=["promptlookup", "lookahead"],
        help="Enable speculative decoding with 'lookahead' or 'promptlookup' mode.",
    )
    parser.add_argument(
        "--ngram-size",
        default=3,
        type=int,
        help="Ngram size for lookahead spec decode",
    )
    parser.add_argument(
        "--max-seq-length",
        default=1024,
        type=int,
        help="Max sequence length for lookahead spec decode",
    )
    parser.add_argument(
        "--spec-length",
        default=3,
        type=int,
        help="Spec length for lookahead spec decode",
    )

    parser.add_argument("--model-path", type=str, default=None, help="Path to a local model (necessary for multi-GPU demo)")
    parser.add_argument(
        "--model", type=str, default='Qwen/Qwen3-8B', help="Model path on hugging face"
    )
    args = parser.parse_args()
    torch.set_default_dtype(torch.bfloat16)
    model_name = args.model
    torch.cuda.set_device(0)
    # with torch.device("cuda"):
    #     model = Qwen3ForCausalLM.from_pretrained(model_name, world_size=1, max_num_pages=args.max_num_pages, page_size=args.page_size).to("cuda")
    #     tokenizer = AutoTokenizer.from_pretrained(model_name)
        
    batch_size = 1
    hidden_size = 2560
    intermediate_size = 9728
    x_torch = torch.randn((batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")
    w_gatedup_torch = torch.randn((intermediate_size*2, hidden_size), dtype=torch.bfloat16, device="cuda")
    w_down_proj_torch = torch.randn((hidden_size, intermediate_size), dtype=torch.bfloat16, device="cuda")
    mlp_out_torch = torch.zeros((batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")

    ###
    warnup_iter = 20
    test_iter = 100
    for _ in range(warnup_iter):
        test_torch_mlp2(x_torch, w_gatedup_torch, w_down_proj_torch)
    ###
    
    starter = torch.cuda.Event(enable_timing=True)
    ender = torch.cuda.Event(enable_timing=True)
        
    starter.record()
    for _ in range(test_iter):
        test_torch_mlp2(x_torch, w_gatedup_torch, w_down_proj_torch)
    ender.record()
    torch.cuda.synchronize()
    run_time = starter.elapsed_time(ender)
    print("torch run time (ms): ", run_time / test_iter)


    from torch.profiler import profile, ProfilerActivity
    if 1:
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            test_torch_mlp2(x_torch, w_gatedup_torch, w_down_proj_torch)
        print(prof.key_averages().table(sort_by="cuda_time_total"))
        prof.export_chrome_trace("trace.json") # chrome://tracing/
    ##########################################################
    
    # pushd build && make -j8 && popd
    
    # git clone --recursive https://www.github.com/mirage-project/mirage
    # pip install -e . -v
    # export MIRAGE_HOME=$(pwd)
    # python demo/qwen3/demo_debug_mlp.py --model=/home/cjmcv/project/llm_models/Qwen/Qwen3-0.6B --use-mirage
    # --profiling https://ui.perfetto.dev/
    
    # nsys profile --trace=cuda,nvtx --output=my_nsys
    # ncu --set full --section "SpeedOfLight_RooflineChart" -o my_profile
