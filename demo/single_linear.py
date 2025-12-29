
import torch
import argparse
from torch import nn
import mirage as mi

from common.pkt_util import TorchRef, MpkReporter, TestUtil
from common.mpk_layers import MpkLayers

if __name__ == "__main__":
    # batch_size只支持8的倍数，gridSize切分后，每个block的N也需要是8的倍数
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
    
    reporter = MpkReporter() 
    model, tokenizer = reporter.memory_footprint_simulation(rank)
    w_rms_torch, w_gatedup_torch, w_down_proj_torch = reporter.get_weight_qwen3_mlp(layer_id=0)
    
    layers = MpkLayers(0, 1, world_size, rank, max_batch_size, args.trace_name, args.profiling)
    mpk = layers.get_mpk()
    
    splitk = 1 # 8
    # hidden_size = 2560        # K
    # intermediate_size = 9728 # torch.randn / ones / TestUtil.create_matrix_arange_col /
    hidden_size = 1024
    intermediate_size = 3072
    
    w_torch = w_gatedup_torch 
    # w_torch = TestUtil.create_matrix_arange_row((intermediate_size*2, hidden_size), dtype=torch.bfloat16, device="cuda")
    # a = torch.ones((100000, 30000), dtype=torch.bfloat16, device="cuda")
    x_torch = torch.ones((max_batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")
    out_torch = torch.zeros((max_batch_size, intermediate_size*2), dtype=torch.bfloat16, device="cuda")
    print("x: ", x_torch.data_ptr(), "w: ", w_torch.data_ptr(), "o: ", out_torch.data_ptr())
    
    # x_torch2 = nn.Parameter(x_torch)
    # w_torch2 = nn.Parameter(w_torch)
    # out_torch2 = nn.Parameter(out_torch)
    
    x = mpk.attach_input(torch_tensor=x_torch, name="in")
    w = mpk.attach_input(torch_tensor=w_torch, name="w")
    linear_out = mpk.attach_input(torch_tensor=out_torch, name="linear_out")

    if splitk == 1:
        mpk.linear_layer(
            input=x,
            weight=w,
            output=linear_out,
            grid_dim=(32, 1, 1),  # (9728 * 2) / 8 = 2432 / ... / 76 / 38 / 19 / 8
            block_dim=(128, 1, 1),
        )
    else:
        mpk.linear_postfix_layer( # [1, 9728] * [2560, 9728] = [1, 2560]
            input=x,
            weight=w,
            output=linear_out,
            grid_dim=(splitk, 8, 1), # (64, 1, 1)
            block_dim=(128, 1, 1),
        )
    
    layers.compile_load(args.nc, args.output_dir)
    
    
    # ###
    def ref_run():
        return TorchRef.linear_o(x_torch[:batch_size], w_torch, out=out_torch)
    
    def mpk_run():
        mpk(batch_size)
    
    mpk(batch_size)    
    ref_output = ref_run()
    mpk_output = out_torch[:batch_size]
    # print("ref_output", ref_output)
    # mpk(batch_size)
    # print("out_torch", out_torch)
    # if (torch.allclose(out_torch, ref_output, rtol=1e-2, atol=0)):
    #     print("allclose: True")
    ###
    
    reporter.generate_report(mpk_run, mpk_output, splitk, 
                            ref_run, ref_output, 
                            warnup_iter=100, test_iter=200, 
                            allclose_iter=5, print_all=False)