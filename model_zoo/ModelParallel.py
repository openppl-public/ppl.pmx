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

class DistMapping(object):
    '''
    A node with 8 GPUs, tp_size = 4, pp_size = 2

    2 tp groups:

    - [0, 1, 2, 3]
    - [4, 5, 6, 7]

    4 pp groups:

    - [0, 4]
    - [1, 5]
    - [2, 6]
    - [3, 7]
    '''
    def __init__(self,
                 world_size=1,
                 rank=0,
                 tp_size=1,
                 pp_size=1):
        self.tp_size = tp_size
        self.pp_size = pp_size
        self.world_size = world_size
        self.rank = rank

        if pp_size * tp_size != world_size:
            raise ValueError("world_size must equal to pp_size * tp_size")
        self.pp_groups = []
        self.tp_groups = []

        # init pp group
        for i in range(tp_size):
            ranks = range(i, world_size, tp_size)
            self.pp_groups.append(list(ranks))

        # init tp group
        for i in range(pp_size):
            ranks = range(i * tp_size, (i + 1) * tp_size)
            self.tp_groups.append(list(ranks))

        self.pp_rank = self.rank // self.tp_size
        self.tp_rank = self.rank % self.tp_size

        self.tp_group = self.tp_groups[self.pp_rank]
        self.pp_group = self.pp_groups[self.tp_rank]

        self.tp_proc_group = None
        self.pp_proc_group = None

        print(f"rank: {self.rank}, tp_groups: {self.tp_groups}, pp_groups: {self.pp_groups}")  

    def is_last_pp_rank(self):
        return self.pp_rank == self.pp_size - 1

    def is_first_pp_rank(self):
        return self.pp_rank == 0

    def has_pp(self):
        return self.pp_size > 1

    def prev_pp_rank(self):
        p = self.rank - self.tp_size
        if p < 0:
            p = p + self.world_size
        return p

    def next_pp_rank(self):
        p = self.rank + self.tp_size
        if p >= self.world_size:
            p = p - self.world_size
        return p

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
