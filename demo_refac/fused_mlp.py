
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
    # reporter.memory_footprint_simulation(rank)
    
    splitk = 1 # 8
    hidden_size = 2560
    intermediate_size = 9728
    # hidden_size = 1024
    # intermediate_size = 3072
    x_torch = torch.randn((max_batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")
    w_gatedup_torch = torch.randn((intermediate_size*2, hidden_size), dtype=torch.bfloat16, device="cuda")
    w_down_proj_torch = torch.randn((hidden_size, intermediate_size), dtype=torch.bfloat16, device="cuda")
    out_torch = torch.zeros((max_batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")
    
    # (38, 19, 20) => 512, 512, 128 => 19456/38, 9728/19, 2560/20
    # (76, 38, 40) => 256, 256, 64 => 19456/76, 9728/38, 2560/40
    gridsize = [152, 76, 20] # 19456/128, 9728/128, 2560/128, 128是TILE_DIM_X
    x = mpk.attach_input(torch_tensor=x_torch, name="in")
    w_gatedup = mpk.attach_input(torch_tensor=w_gatedup_torch, name="w_gatedup")
    w_down_proj = mpk.attach_input(torch_tensor=w_down_proj_torch, name="w_down_proj")
    mlp_out = mpk.attach_input(torch_tensor=out_torch, name="mlp_out")
    
    # mlp_mid_torch = torch.zeros((max_batch_size, intermediate_size*2), dtype=torch.bfloat16, device="cuda")
    # mlp_mid = mpk.attach_input(torch_tensor=mlp_mid_torch, name="mlp_mid")
    mlp_mid = mpk.new_tensor(dims=(max_batch_size, intermediate_size*2), dtype=mi.bfloat16, name="mlp_mid", io_category="cuda_tensor")
    mpk.linear_layer(
        input=x,
        weight=w_gatedup,
        output=mlp_mid,
        grid_dim=(gridsize[0], 1, 1),
        block_dim=(128, 1, 1),
    )
    
    # silu_mul_out_torch = torch.zeros((max_batch_size, intermediate_size), dtype=torch.bfloat16, device="cuda")
    # silu_mul_out = mpk.attach_input(torch_tensor=silu_mul_out_torch, name="silu_mul_out")
    # mlp_out_torch = silu_mul_out_torch
    silu_mul_out = mpk.new_tensor(dims=(max_batch_size, intermediate_size), dtype=mi.bfloat16, name="silu_mul_out", io_category="cuda_tensor")
    mpk.silu_mul_layer(
        input=mlp_mid,
        output=silu_mul_out,
        grid_dim=(gridsize[1], 1, 1),
        block_dim=(128, 1, 1),
    )

    mpk.linear_layer(
        input=silu_mul_out,
        weight=w_down_proj,
        output=mlp_out,
        grid_dim=(gridsize[2], 1, 1),    # (2560) / 128 = 40 / 20
        block_dim=(128, 1, 1),
    )

    layers.compile_load(args.nc, args.output_dir)
    
    ###
    def ref_run():
        return TorchRef.mlp(x_torch[:batch_size], w_gatedup_torch, w_down_proj_torch)
    graph, ref_output = TorchRef.compile_capture(ref_run, is_compile=False)
    
    def mpk_run():
        mpk(batch_size)
        
    ref_output = ref_run()
    mpk_output = out_torch[:batch_size]
    
    # mpk_run()
    # for _ in range(100):
    #     graph.replay()
    ###
    
    reporter.generate_report(mpk_run, mpk_output, splitk, 
                            graph.replay, ref_output, 
                            warnup_iter=100, test_iter=200, 
                            allclose_iter=5, print_all=False)