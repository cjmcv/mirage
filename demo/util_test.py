
import torch
from torch import nn
import argparse
import mirage as mi

from pkt_util import TorchRef

if __name__ == "__main__":
    batch_size = 1
    splitk = 1 # 8
    hidden_size = 2560
    intermediate_size = 9728
    x_torch = torch.randn((batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")
    w_rms_torch = torch.randn((1, hidden_size), dtype=torch.bfloat16, device="cuda")
    w_gatedup_torch = torch.randn((intermediate_size*2, hidden_size), dtype=torch.bfloat16, device="cuda")
    w_down_proj_torch = torch.randn((hidden_size, intermediate_size), dtype=torch.bfloat16, device="cuda")
    mlp_out_torch = torch.zeros((splitk, hidden_size), dtype=torch.bfloat16, device="cuda")
    
    ###
    weight = nn.Parameter(torch.ones(hidden_size))
    print(type(weight.data.data_ptr()))
    print(type(w_gatedup_torch.data_ptr()))
    
    warnup_iter = 100
    test_iter = 200
            
    def ref_run():
        return TorchRef.norm_mlp(x_torch, w_rms_torch, w_gatedup_torch, w_down_proj_torch) + x_torch
    
    graph, output = TorchRef.compile_capture(ref_run, is_compile=True) 
    output.zero_()
    graph.replay()
    print(output)    
    ###

