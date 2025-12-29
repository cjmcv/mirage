
import torch
import mirage as mi

class MpkLayers:
    def __init__(self, instance_id, kernel_num, world_size, rank, max_batch_size, trace_name, profiling):
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
            instance_id=instance_id,
            kernel_num=kernel_num,
            mode="offline",
            world_size=world_size,
            mpi_rank=rank,
            num_workers=num_workers,
            num_local_schedulers=num_schedulers,
            num_remote_schedulers=0,
            meta_tensors={}, #  meta_tensors={"qo_indptr_buffer": self.qo_indptr_buffer,},
            profiler_tensor=self.profiler_tensor,
            trace_name=trace_name,
            use_cutlass_kernel=True,
        )
        self.max_batch_size = max_batch_size
    
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
                
    def create_qwen3_oproj_norm_mlp(self, gridsize, total_head_dims, hidden_size, intermediate_size, 
                                    w_o_proj_torch, w_rms_torch, w_gatedup_torch, w_down_proj_torch):
        self.x_torch = torch.randn((self.max_batch_size, total_head_dims), dtype=torch.bfloat16, device="cuda")
        self.x_residual_torch = torch.randn((self.max_batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")
        self.out_torch = torch.zeros((self.max_batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")
    
        # gridsize = [max_batch_size, 76, 38, 40]
        self.x = self.mpk.attach_input(torch_tensor=self.x_torch, name="in")
        self.x_residual = self.mpk.attach_input(torch_tensor=self.x_residual_torch, name="x_residual")
        self.w_o_proj = self.mpk.attach_input(torch_tensor=w_o_proj_torch, name="w_o_proj")
        self.w_rms = self.mpk.attach_input(torch_tensor=w_rms_torch, name="w_rms")
        self.w_gatedup = self.mpk.attach_input(torch_tensor=w_gatedup_torch, name="w_gatedup")
        self.w_down_proj = self.mpk.attach_input(torch_tensor=w_down_proj_torch, name="w_down_proj")
        self.mlp_out = self.mpk.attach_input(torch_tensor=self.out_torch, name="mlp_out")
        
        self.o_proj_out = self.mpk.new_tensor(dims=(self.max_batch_size, hidden_size), dtype=mi.bfloat16, name="o_proj_out", io_category="cuda_tensor")
        self.mpk.linear_with_residual_layer(
            input=self.x,
            weight=self.w_o_proj,
            residual=self.x_residual,
            output=self.o_proj_out,
            grid_dim=(gridsize[0], 1, 1),
            block_dim=(128, 1, 1),
        )
        
        ## self.o_proj_out
         
        self.rmsnorm_out = self.mpk.new_tensor(dims=(self.max_batch_size, hidden_size), dtype=mi.bfloat16, name="rmsnorm_out", io_category="cuda_tensor")
        self.mpk.rmsnorm_layer(
            input=self.o_proj_out,
            weight=self.w_rms,
            output=self.rmsnorm_out,
            grid_dim=(gridsize[1], 1, 1),
            block_dim=(128, 1, 1),
        )
        
        # mlp_mid_torch = torch.zeros((max_batch_size, intermediate_size*2), dtype=torch.bfloat16, device="cuda")
        # mlp_mid = mpk.attach_input(torch_tensor=mlp_mid_torch, name="mlp_mid")    
        self.mlp_mid = self.mpk.new_tensor(dims=(self.max_batch_size, intermediate_size*2), dtype=mi.bfloat16, name="mlp_mid", io_category="cuda_tensor")
        self.mpk.linear_layer(
            input=self.rmsnorm_out,
            weight=self.w_gatedup,
            output=self.mlp_mid,
            grid_dim=(gridsize[2], 1, 1),
            block_dim=(128, 1, 1),
        )
        
        # silu_mul_out_torch = torch.zeros((max_batch_size, intermediate_size), dtype=torch.bfloat16, device="cuda")
        # silu_mul_out = mpk.attach_input(torch_tensor=silu_mul_out_torch, name="silu_mul_out")
        # out_torch = silu_mul_out_torch
        self.silu_mul_out = self.mpk.new_tensor(dims=(self.max_batch_size, intermediate_size), dtype=mi.bfloat16, name="silu_mul_out", io_category="cuda_tensor")
        self.mpk.silu_mul_layer(
            input  = self.mlp_mid,
            output = self.silu_mul_out,
            grid_dim  = (gridsize[3], 1, 1),
            block_dim = (128, 1, 1),
        )
        self.mpk.linear_with_residual_layer( # [1, 9728] * [2560, 9728] = [1, 2560]
            input = self.silu_mul_out,
            weight = self.w_down_proj,
            residual = self.o_proj_out,
            output = self.mlp_out,
            grid_dim = (gridsize[4], 1, 1), # (64, 1, 1)
            block_dim = (128, 1, 1),
        )
            
        return self.x_torch, self.x_residual_torch, self.out_torch
    
    def create_qwen3_norm_mlp(self, gridsize, hidden_size, intermediate_size, w_rms_torch, w_gatedup_torch, w_down_proj_torch):
        self.x_torch = torch.randn((self.max_batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")
        self.out_torch = torch.zeros((self.max_batch_size, hidden_size), dtype=torch.bfloat16, device="cuda")
    
        # gridsize = [max_batch_size, 76, 38, 40]
        self.x = self.mpk.attach_input(torch_tensor=self.x_torch, name="in")
        self.w_rms = self.mpk.attach_input(torch_tensor=w_rms_torch, name="w_rms")
        self.w_gatedup = self.mpk.attach_input(torch_tensor=w_gatedup_torch, name="w_gatedup")
        self.w_down_proj = self.mpk.attach_input(torch_tensor=w_down_proj_torch, name="w_down_proj")
        self.mlp_out = self.mpk.attach_input(torch_tensor=self.out_torch, name="mlp_out")
        
        self.rmsnorm_out = self.mpk.new_tensor(dims=(self.max_batch_size, hidden_size), dtype=mi.bfloat16, name="rmsnorm_out", io_category="cuda_tensor")
        self.mpk.rmsnorm_layer(
            input=self.x,
            weight=self.w_rms,
            output=self.rmsnorm_out,
            grid_dim=(gridsize[0], 1, 1),
            block_dim=(128, 1, 1),
        )
        
        # mlp_mid_torch = torch.zeros((max_batch_size, intermediate_size*2), dtype=torch.bfloat16, device="cuda")
        # mlp_mid = mpk.attach_input(torch_tensor=mlp_mid_torch, name="mlp_mid")    
        self.mlp_mid = self.mpk.new_tensor(dims=(self.max_batch_size, intermediate_size*2), dtype=mi.bfloat16, name="mlp_mid", io_category="cuda_tensor")
        self.mpk.linear_layer(
            input=self.rmsnorm_out,
            weight=self.w_gatedup,
            output=self.mlp_mid,
            grid_dim=(gridsize[1], 1, 1),
            block_dim=(128, 1, 1),
        )
        
        # silu_mul_out_torch = torch.zeros((max_batch_size, intermediate_size), dtype=torch.bfloat16, device="cuda")
        # silu_mul_out = mpk.attach_input(torch_tensor=silu_mul_out_torch, name="silu_mul_out")
        # out_torch = silu_mul_out_torch
        self.silu_mul_out = self.mpk.new_tensor(dims=(self.max_batch_size, intermediate_size), dtype=mi.bfloat16, name="silu_mul_out", io_category="cuda_tensor")
        self.mpk.silu_mul_layer(
            input  = self.mlp_mid,
            output = self.silu_mul_out,
            grid_dim  = (gridsize[2], 1, 1),
            block_dim = (128, 1, 1),
        )
        self.mpk.linear_with_residual_layer( # [1, 9728] * [2560, 9728] = [1, 2560]
            input = self.silu_mul_out,
            weight = self.w_down_proj,
            residual = self.x,
            output = self.mlp_out,
            grid_dim = (gridsize[3], 1, 1), # (64, 1, 1)
            block_dim = (128, 1, 1),
        )
            
        return self.x_torch, self.out_torch