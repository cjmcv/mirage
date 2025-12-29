
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
    
    loop = 10
    splitk = 1 # 8
    hidden_size = 2560
    intermediate_size = 9728
    x_torch = torch.randn((max_batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")
    out_torch = torch.zeros((max_batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")
    
    x = mpk.attach_input(torch_tensor=x_torch, name="in")
    mlp_out = mpk.attach_input(torch_tensor=out_torch, name="mlp_out")
    
    temp = []
    w_gatedup_torch = []
    w_down_proj_torch = []
    w_gatedup = []
    w_down_proj = []
    # 模拟？
    for i in range(loop*2):
        temp.append(torch.randn((intermediate_size*2, hidden_size), dtype=torch.bfloat16, device="cuda"))
        
    for i in range(loop):
        w_gatedup_torch.append(torch.randn((intermediate_size*2, hidden_size), dtype=torch.bfloat16, device="cuda"))
        w_down_proj_torch.append(torch.randn((hidden_size, intermediate_size), dtype=torch.bfloat16, device="cuda"))
        
        w_gatedup.append(mpk.attach_input(torch_tensor=w_gatedup_torch[i], name=f"w_gatedup_{i}"))
        w_down_proj.append(mpk.attach_input(torch_tensor=w_down_proj_torch[i], name=f"w_down_proj_{i}"))

    
    for i in range(loop):
        gridsize = [76, 38, 40]

        # mlp_mid_torch = torch.zeros((max_batch_size, intermediate_size*2), dtype=torch.bfloat16, device="cuda")
        # mlp_mid = mpk.attach_input(torch_tensor=mlp_mid_torch, name="mlp_mid")    
        mlp_mid = mpk.new_tensor(dims=(max_batch_size, intermediate_size*2), dtype=mi.bfloat16, name=f"mlp_mid_{i}", io_category="cuda_tensor")
        mpk.linear_layer(
            input=x,
            weight=w_gatedup[i],
            output=mlp_mid,
            grid_dim=(gridsize[0], 1, 1),
            block_dim=(128, 1, 1),
        )
        
        # silu_mul_out_torch = torch.zeros((max_batch_size, intermediate_size), dtype=torch.bfloat16, device="cuda")
        # silu_mul_out = mpk.attach_input(torch_tensor=silu_mul_out_torch, name="silu_mul_out")
        # mlp_out_torch = silu_mul_out_torch
        silu_mul_out = mpk.new_tensor(dims=(max_batch_size, intermediate_size), dtype=mi.bfloat16, name=f"silu_mul_out_{i}", io_category="cuda_tensor")
        mpk.silu_mul_layer(
            input=mlp_mid,
            output=silu_mul_out,
            grid_dim=(gridsize[1], 1, 1),
            block_dim=(128, 1, 1),
        )
        mpk.linear_layer( # [1, 9728] * [2560, 9728] = [1, 2560]
            input=silu_mul_out,
            weight=w_down_proj[i],
            output=mlp_out,
            grid_dim=(gridsize[2], 1, 1),    # (2560) / 128 = 40 / 20
            block_dim=(128, 1, 1),
        )
        
        x = mlp_out

    layers.compile_load(args.nc, args.output_dir)
    
    ###
    def ref_run():
        input = x_torch[:batch_size]
        for i in range(loop):
            output = TorchRef.mlp(input, w_gatedup_torch[i], w_down_proj_torch[i])
            input = output
        return output
    
    graph, ref_output = TorchRef.compile_capture(ref_run, is_compile=False)
    
    def mpk_run():
        mpk(batch_size)
        
    ref_output = ref_run()
    mpk_output = out_torch[:batch_size]
    ###
    
    reporter.generate_report(mpk_run, mpk_output, splitk, 
                            graph.replay, ref_output, 
                            warnup_iter=100, test_iter=200, 
                            allclose_iter=5, print_all=False)