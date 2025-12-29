
import torch
import argparse
import mirage as mi

from common.pkt_util import TorchRef, MpkReporter
from common.mpk_layers import MpkLayers

if __name__ == "__main__":
    max_batch_size = 1
    batch_size = 1
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="./gen", help="Output files directory")
    parser.add_argument("--trace-name", default="qwen3", help="Perfetto trace output name")
    parser.add_argument("--profiling", action="store_true", help="Use Profiler to generate trace")
    parser.add_argument("--nc", action="store_true", help="no-compile: Use the specified compiled library instead of recompiling it")

    args = parser.parse_args()
    world_size = 1
    rank = 0

    global print
    if rank != 0:
        print = lambda *_, **__: None

    print("Input arguments:", args)
    print(f"world_size({world_size}) rank({rank})")
    # model_name = args.model
    torch.set_default_dtype(torch.bfloat16)

    
    layers = MpkLayers(0, 1, world_size, rank, max_batch_size, args.trace_name, args.profiling)
    mpk = layers.get_mpk()
    
    reporter = MpkReporter() 
    model, tokenizer = reporter.memory_footprint_simulation(rank) 
    w_rms_torch, w_gatedup_torch, w_down_proj_torch = reporter.get_weight_qwen3_mlp(layer_id=0)
    
    # hidden_size = 2560
    # intermediate_size = 9728
    hidden_size = 1024
    intermediate_size = 3072

    # w_rms_torch = torch.randn((1, hidden_size), dtype=torch.bfloat16, device="cuda")
    # w_gatedup_torch = torch.randn((intermediate_size*2, hidden_size), dtype=torch.bfloat16, device="cuda")
    # w_down_proj_torch = torch.randn((hidden_size, intermediate_size), dtype=torch.bfloat16, device="cuda")
       
    # (batch_size, 38, 19, 20)
    # (batch_size, 76, 38, 40)
    # gridsize = [max_batch_size, 76, 38, 40]
    gridsize = [max_batch_size, 48, 24, 32] # 1024 3072
    x_torch, out_torch = layers.create_qwen3_norm_mlp(gridsize, hidden_size, intermediate_size, w_rms_torch, w_gatedup_torch, w_down_proj_torch)    
    layers.compile_load(args.nc, args.output_dir)
    
    # pkt.memory_footprint_simulation(rank)
        
    ###

    def ref_run():
        return TorchRef.norm_mlp(x_torch[:batch_size], w_rms_torch, w_gatedup_torch, w_down_proj_torch)
    def mpk_run():
        mpk(batch_size)
        
    graph, ref_output = TorchRef.compile_capture(ref_run, is_compile=False)
    mpk_output = out_torch[:batch_size]
    ###
    
    if (args.profiling):
        mpk(batch_size)
        print("Finish profiling.")
        exit()
        
    reporter.generate_report(mpk_run, mpk_output, 1, 
                            graph.replay, ref_output, 
                            warnup_iter=100, test_iter=200, 
                            allclose_iter=5, print_all=False)

