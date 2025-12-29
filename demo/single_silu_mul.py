
import torch
import argparse
import mirage as mi

from common.pkt_util import TorchRef, MpkReporter, TestUtil
from common.mpk_layers import MpkLayers

if __name__ == "__main__":
    max_batch_size = 128
    batch_size = 128
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

    layers = MpkLayers(0,1, world_size, rank, max_batch_size, args.trace_name, args.profiling)
    mpk = layers.get_mpk()
    reporter = MpkReporter() 
    # reporter.memory_footprint_simulation(rank)
    
    splitk = 1 # 8
    hidden_size = 2560
    intermediate_size = 9728
    x_torch = torch.randn((batch_size, intermediate_size*2), dtype=torch.bfloat16, device="cuda")
    out_torch = torch.zeros((batch_size, intermediate_size), dtype=torch.bfloat16, device="cuda")
    
    x = mpk.attach_input(torch_tensor=x_torch, name="in")
    silu_mul_out = mpk.attach_input(torch_tensor=out_torch, name="silu_mul_out")
    mpk.silu_mul_layer(
        input=x,
        output=silu_mul_out,
        grid_dim=(16, 1, 1),
        block_dim=(128, 1, 1),
    )
    
    layers.compile_load(args.nc, args.output_dir)
    
    ##
    def ref_run():
        return TorchRef.silu_and_mul(x_torch)
    def mpk_run():
        mpk(batch_size)

    ref_output = ref_run()
    ###
    
    reporter.generate_report(mpk_run, out_torch, splitk, 
                            ref_run, ref_output, 
                            warnup_iter=100, test_iter=200, 
                            allclose_iter=5, print_all=False)