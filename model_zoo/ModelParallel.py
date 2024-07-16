from typing import Tuple

import sys
import os
import torch
import torch.nn as nn
import torch.distributed as dist

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/..")

import torch_function as OPMX

def setup(use_cpu: bool = True) -> Tuple[int, int]:
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    if use_cpu:
        dist.init_process_group("gloo")
    else:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)

    # seed must be the same in all processes
    torch.manual_seed(1)
    return local_rank, world_size


class ParallelEmbedding(torch.nn.Module):
    def __init__(
        self,
        proc_group: dist.ProcessGroup,
        num_embeddings: int,
        embedding_dim: int,
        padding_idx: int = -1,
        max_norm: float = 0,
        norm_type: float = 2) -> None:
        super().__init__()

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx
        self.max_norm = max_norm
        self.norm_type = norm_type
        self.proc_group = proc_group

        world_size = 1 if proc_group is None else proc_group.size()
        assert embedding_dim % world_size == 0, "{} is not divisible by {}".format(embedding_dim, world_size)

        self.embedding_dim_per_partition = embedding_dim // world_size

        self.weight = nn.Parameter(torch.ones(self.num_embeddings, self.embedding_dim_per_partition))

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
            return OPMX.parallel_embedding(
                ids, self.weight, self.proc_group,
                self.num_embeddings, self.embedding_dim,
                self.padding_idx, self.max_norm,
                self.norm_type)


class ColumnParallelLinear(torch.nn.Module):
    def __init__(
        self,
        proc_group: dist.ProcessGroup,
        in_features: int,
        out_features: int,
        bias_term: bool = True,
        gather_output: bool = True) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.proc_group = proc_group

        world_size = 1 if proc_group is None else proc_group.size()
        assert out_features % world_size == 0, "{} is not divisible by {}".format(out_features, world_size)

        self.out_features_per_partition = out_features // world_size

        self.weight = nn.Parameter(torch.ones(self.out_features_per_partition, self.in_features))
        if bias_term:
            self.bias = nn.Parameter(torch.zeros(self.out_features_per_partition))
        else:
            self.register_parameter("bias", None)


    def forward(self, X: torch.Tensor):
        return OPMX.column_parallel_linear(
            X, self.weight, self.bias, self.proc_group,
            self.in_features, self.out_features, self.gather_output)


class RowParallelLinear(torch.nn.Module):
    def __init__(
        self,
        proc_group: dist.ProcessGroup,
        in_features: int,
        out_features: int,
        bias_term: bool = True,
        input_is_parallel: bool = False) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self.proc_group = proc_group

        world_size = 1 if proc_group is None else proc_group.size()
        assert in_features % world_size == 0, "{} is not divisible by {}".format(in_features, world_size)

        self.in_features_per_partition = in_features // world_size

        self.weight = nn.Parameter(torch.ones(self.out_features, self.in_features_per_partition))
        if bias_term:
            self.bias = nn.Parameter(torch.zeros(self.out_features))
        else:
            self.register_parameter("bias", None)


    def forward(self, X: torch.Tensor):
        return OPMX.row_parallel_linear(
            X, self.weight, self.bias, self.proc_group,
            self.in_features, self.out_features, self.input_is_parallel)


class MoeColumnParallelLinear(torch.nn.Module):
    def __init__(
        self,
        proc_group: dist.ProcessGroup,
        num_experts: int,
        in_features: int,
        out_features: int,
        bias_term: bool = True,
        gather_output: bool = True) -> None:
        super().__init__()
        
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.proc_group = proc_group
        
        world_size = 1 if proc_group is None else proc_group.size()
        assert out_features % world_size == 0, "{} is not divisible by {}".format(out_features, world_size)

        self.out_features_per_partition = out_features // world_size

        self.weight = nn.Parameter(torch.ones(self.num_experts, self.out_features_per_partition, self.in_features))
        if bias_term:
            self.bias = nn.Parameter(torch.zeros(self.num_experts, self.out_features_per_partition))
        else:
            self.register_parameter("bias", None)
            
    def forward(self, X: torch.Tensor, expert_offset: torch.Tensor):
        return OPMX.moe_column_parallel_linear(
            X, expert_offset, self.weight, self.bias,
            self.proc_group, self.num_experts, self.in_features,
            self.out_features, self.gather_output)


class MoeRowParallelLinear(torch.nn.Module):
    def __init__(
        self, 
        proc_group: dist.ProcessGroup,
        num_experts: int,
        in_features: int,
        out_features: int,
        bias_term: bool = True,
        input_is_parallel: bool = False) -> None:
        super().__init__()
        
        self.num_experts = num_experts
        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self.proc_group = proc_group

        world_size = 1 if proc_group is None else proc_group.size()
        assert in_features % world_size == 0, "{} is not divisible by {}".format(in_features, world_size)

        self.in_features_per_partition = in_features // world_size
        
        self.weight = nn.Parameter(torch.ones(self.num_experts, self.out_features, self.in_features_per_partition))
        if bias_term:
            self.bias = nn.Parameter(torch.zeros(self.num_experts, self.out_features))
        else:
            self.register_parameter("bias", None)
            
        
            
    def forward(self, X: torch.Tensor, expert_offset: torch.Tensor):
        return OPMX.moe_row_parallel_linear(
            X, expert_offset, self.weight, self.bias,
            self.proc_group, self.num_experts,
            self.in_features, self.out_features, self.input_is_parallel)

## Quant Using
class WoquColumnParallelLinear(torch.nn.Module):
    def __init__(
        self,
        proc_group: dist.ProcessGroup,
        in_features: int,
        out_features: int,
        bias_term: bool = True,
        gather_output: bool = True,
        quant_data_type: str = "int4",
        quant_method: str = "weight_only",
        quant_axis: int = 1,
        group_size: int = 128,
        storage_bits: int = 32,
        has_zeropoint = False,
        float_zeropoint = False) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.gather_output = gather_output
        self.proc_group = proc_group
        self.quant_data_type = quant_data_type
        self.quant_method = quant_method
        self.quant_axis = quant_axis
        self.group_size = group_size
        self.has_zeropoint = has_zeropoint
        self.float_zeropoint = float_zeropoint
        self.storage_bits = storage_bits

        world_size = 1 if proc_group is None else proc_group.size()
        assert out_features % world_size == 0, "{} is not divisible by {}".format(out_features, world_size)

        self.out_features_per_partition = out_features // world_size
        if quant_data_type == "int4":
            if self.storage_bits == 16:
                self.register_buffer('qweight', torch.ones(self.out_features_per_partition // (storage_bits // 4), self.in_features, dtype=torch.int16))
            elif self.storage_bits == 32:
                self.register_buffer('qweight', torch.ones(self.out_features_per_partition // (storage_bits // 4), self.in_features, dtype=torch.int32))
        elif quant_data_type == "int8":
            self.register_buffer('qweight', torch.ones(self.out_features_per_partition, self.in_features, dtype=torch.int8))
        else:
            raise ValueError("quant_data_type must be one of int4 or int8")
        
        if bias_term:
            self.bias = nn.Parameter(torch.zeros(self.out_features_per_partition))
        else:
            self.register_parameter("bias", None)

        if self.has_zeropoint:
            dtype = torch.float32 if self.float_zeropoint else torch.int32
            self.register_buffer(
                'zeropoint', 
                torch.ones(
                    self.out_features_per_partition, 
                    self.in_features // self.group_size, 
                    dtype=dtype, 
                    requires_grad=False
                )
            )
        else:
            self.register_parameter("zeropoint", None)
        
        self.scale = nn.Parameter(torch.ones(self.out_features_per_partition, self.in_features // self.group_size, dtype=torch.float32, requires_grad=False))

    def forward(self, X: torch.Tensor):
        return OPMX.woqu_column_parallel_linear(
            X, self.qweight, self.scale, self.zeropoint, self.bias, self.proc_group, self.quant_data_type,
            self.in_features, self.out_features, self.gather_output, self.quant_method, self.quant_axis,
            self.group_size, self.has_zeropoint, self.float_zeropoint)


class WoquRowParallelLinear(torch.nn.Module):
    def __init__(
        self,
        proc_group: dist.ProcessGroup,
        in_features: int,
        out_features: int,
        bias_term: bool = True,
        input_is_parallel: bool = True,
        quant_data_type: str = "int4",
        quant_method: str = "weight_only",
        quant_axis: int = 1,
        group_size: int = 128,
        storage_bits: int = 32,
        has_zeropoint = False,
        float_zeropoint = False) -> None:
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.input_is_parallel = input_is_parallel
        self.proc_group = proc_group
        self.quant_data_type = quant_data_type
        self.quant_method = quant_method
        self.quant_axis = quant_axis
        self.group_size = group_size
        self.has_zeropoint = has_zeropoint
        self.float_zeropoint = float_zeropoint
        self.storage_bits = storage_bits

        world_size = 1 if proc_group is None else proc_group.size()
        assert out_features % world_size == 0, "{} is not divisible by {}".format(out_features, world_size)

        self.out_features_per_partition = out_features // world_size
        if quant_data_type == "int4":
            if self.storage_bits == 16:
                self.register_buffer('qweight', torch.ones(self.out_features_per_partition // (storage_bits // 4), self.in_features, dtype=torch.int16))
            elif self.storage_bits == 32:
                self.register_buffer('qweight', torch.ones(self.out_features_per_partition // (storage_bits // 4), self.in_features, dtype=torch.int32))
        elif quant_data_type == "int8":
            self.register_buffer('qweight', torch.ones(self.out_features_per_partition, self.in_features, dtype=torch.int8))
        else:
            raise ValueError("quant_data_type must be one of int4 or int8")
        
        if bias_term:
            self.bias = nn.Parameter(torch.zeros(self.out_features_per_partition))
        else:
            self.register_parameter("bias", None)

        if self.has_zeropoint:
            dtype = torch.float32 if self.float_zeropoint else torch.int32
            self.register_buffer(
                'zeropoint', 
                torch.ones(
                    self.out_features_per_partition, 
                    self.in_features // self.group_size, 
                    dtype=dtype, 
                    requires_grad=False
                )
            )
        else:
            self.register_parameter("zeropoint", None)
        
        
        self.scale = nn.Parameter(torch.ones(self.out_features_per_partition, self.in_features // self.group_size, dtype=torch.float32, requires_grad=False))

    def forward(self, X: torch.Tensor):
        return OPMX.woqu_row_parallel_linear(
            X, self.qweight, self.scale, self.zeropoint, self.bias, self.proc_group, self.quant_data_type,
            self.in_features, self.out_features, self.input_is_parallel, self.quant_method, self.quant_axis,
            self.group_size, self.has_zeropoint, self.float_zeropoint)