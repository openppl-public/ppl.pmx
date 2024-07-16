import os
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional

from .WeightOnlyQuantUtils import Int4QuantUtils


class WoquRowParallelLinear(torch.autograd.Function):
    @staticmethod
    def symbolic(
        g, X: torch.Value, W: torch.Value, Scale: torch.Value, ZeroPoint: Optional[torch.Value],
        B: Optional[torch.Value], proc_group: torch.Value, quant_data_type: str,
        in_features: int, out_features: int, input_is_parallel: bool = False,
        quant_method: str='', quant_axis: int=1, group_size: int=128,
        has_zeropoint: bool=False, float_zeropoint: bool=False):
        if B is not None:
            Y = g.op("opmx::WoquRowParallelLinear", X, W, Scale, ZeroPoint, B,
                    quant_data_type_s = quant_data_type,
                    in_features_i = in_features,
                    out_features_i = out_features,
                    bias_term_i = True,
                    input_is_parallel_i = input_is_parallel,
                    quant_method_s = quant_method,
                    quant_axis_i = quant_axis,
                    group_size_i = group_size,
                    has_zeropoint_i = has_zeropoint,
                    float_zeropoint_i = float_zeropoint)
        elif ZeroPoint is not None:
            Y = g.op("opmx::WoquRowParallelLinear", X, W, Scale, ZeroPoint,
                    quant_data_type_s = quant_data_type,
                    in_features_i = in_features,
                    out_features_i = out_features,
                    bias_term_i = False,
                    input_is_parallel_i = input_is_parallel,
                    quant_method_s = quant_method,
                    quant_axis_i = quant_axis,
                    group_size_i = group_size,
                    has_zeropoint_i = has_zeropoint,
                    float_zeropoint_i = float_zeropoint)
        else:
            Y = g.op("opmx::WoquRowParallelLinear", X, W, Scale,
                     quant_data_type_s = quant_data_type,
                     in_features_i = in_features,
                     out_features_i = out_features,
                     bias_term_i = False,
                     input_is_parallel_i = input_is_parallel,
                     quant_method_s = quant_method,
                     quant_axis_i = quant_axis,
                     group_size_i = group_size,
                     has_zeropoint_i = has_zeropoint,
                     float_zeropoint_i = False)
        return Y


    @staticmethod
    def forward(
        self, X: torch.Tensor, W: torch.Tensor, Scale: torch.Value, ZeroPoint: Optional[torch.Value],
        B: Optional[torch.Value], proc_group: dist.ProcessGroup, quant_data_type: str,
        in_features: int, out_features: int, input_is_parallel: bool = False, quant_method: str='',
        quant_axis: int=1, group_size: int=128, has_zeropoint: bool=False, float_zeropoint: bool=False):

        # for row parallel, if input_is_parallel eq false, we need to split
        if torch.onnx.is_in_onnx_export():
            if W.dtype == torch.int32:
                output_parallel = torch.zeros(*X.shape[:-1], W.shape[0] * 8, dtype=W.dtype).to(X.device)
            elif W.dtype == torch.int16:
                output_parallel = torch.zeros(*X.shape[:-1], W.shape[0] * 4, dtype=W.dtype).to(X.device)
            else:
                output_parallel = torch.zeros(*X.shape[:-1], W.shape[0], dtype=W.dtype).to(X.device)
                
            if not input_is_parallel and proc_group is not None \
                and torch.distributed.get_world_size(proc_group) > 1:
                rank = torch.distributed.get_rank(group=proc_group)
                world_size = torch.distributed.get_world_size(group=proc_group)
                Y = torch.split(output_parallel, world_size, dim=-1)[0]
            else:
                Y = output_parallel
            return Y
        else:
            if not input_is_parallel and proc_group is not None \
                and torch.distributed.get_world_size(proc_group) > 1:
                rank = torch.distributed.get_rank(group=proc_group)
                world_size = torch.distributed.get_world_size(group=proc_group)
                x_parallel = torch.split(X, world_size, dim=-1)[rank]
            else:
                x_parallel = X
            # Matrix multiply.
            assert quant_data_type == 'int4', 'int8 dequantize is not implemented'
            if W.dtype == torch.int32:
                unpacked_int4_w = Int4QuantUtils.unpack(W, 32, 4)
            elif W.dtype == torch.int16:
                unpacked_int4_w = Int4QuantUtils.unpack(W, 16, 4)

            dequant_fp16_w = Int4QuantUtils.dequantize_int4_to_fp16(unpacked_int4_w, Scale, ZeroPoint, group_size)
            output_parallel = F.linear(x_parallel, dequant_fp16_w, B)
            Y = output_parallel
        return Y


def woqu_row_parallel_linear(
    X: torch.Tensor, W: torch.Tensor, Scale: torch.Value, ZeroPoint: Optional[torch.Value],
    B: Optional[torch.Value], proc_group: dist.ProcessGroup, quant_data_type: str, in_features: int,
    out_features: int, input_is_parallel: bool = False, quant_method: str='', quant_axis: int=1,
    group_size: int=128, has_zeropoint: bool=False, float_zeropoint: bool=False) -> torch.Tensor:

    if B is not None:
        _ZeroPoint = torch.empty(0, device=X.device)
    else:
        _ZeroPoint = ZeroPoint

    return WoquRowParallelLinear.apply(X, W, Scale, _ZeroPoint, B, proc_group, quant_data_type,
                                       in_features, out_features, input_is_parallel, quant_method,
                                       quant_axis, group_size, has_zeropoint, float_zeropoint)


if __name__ == "__main__":
    class TestModule1(torch.nn.Module):
        def __init__(
            self,
            proc_group: dist.ProcessGroup,
            in_features: int,
            out_features: int,
            quant_data_type: str = 'int4',
            quant_method: str = 'weight only',
            quant_axis: int = 1,
            group_size: int = 128,
            storage_bits: int = 16,
            has_zeropoint: bool=False,
            float_zeropoint: bool=False,
            bias_term: bool = True,
            input_is_parallel: bool = False) -> None:
            super().__init__()

            self.in_features = in_features
            self.out_features = out_features
            self.input_is_parallel = input_is_parallel
            self.proc_group = proc_group

            self.quant_data_type = quant_data_type
            self.quant_method= quant_method
            self.quant_axis = quant_axis
            self.group_size = group_size
            self.has_zeropoint = has_zeropoint
            self.float_zeropoint = float_zeropoint

            world_size = 1 if proc_group is None else proc_group.size()
            assert out_features % world_size == 0, "{} is not divisible by {}".format(out_features, world_size)

            self.out_features_per_partition = out_features // world_size

            # pack int4 to int16
            if self.quant_data_type == 'int4':
                self.register_buffer('weight', torch.ones( self.out_features_per_partition // (storage_bits // 4), self.in_features, dtype=torch.int16))
            elif self.quant_data_type == 'int8' and self.has_zeropoint == False:
                self.weight = self.register_buffer('weight', torch.ones(self.out_features_per_partition, self.in_features, dtype=torch.int8))

            if bias_term:
                self.bias = nn.Parameter(torch.zeros(self.out_features_per_partition, dtype=torch.float16))
            else:
                self.register_parameter("bias", None)

            if self.has_zeropoint:
                self.register_buffer('ZeroPoint', torch.ones(self.out_features_per_partition, self.in_features // self.group_size, dtype=torch.int8))
            else:
                self.register_buffer('ZeroPoint', None)
            self.Scale = nn.Parameter(torch.ones(self.out_features_per_partition, self.in_features // self.group_size, dtype=torch.float16))


        def forward(self, X: torch.Tensor):
            return woqu_row_parallel_linear(
                X, self.weight, self.Scale, self.ZeroPoint, self.bias, self.proc_group, self.quant_data_type,
                self.in_features, self.out_features, self.input_is_parallel, self.quant_method, self.quant_axis,
                self.group_size, self.has_zeropoint, self.float_zeropoint)


    test_op1 = TestModule1(None, 512, 2048, has_zeropoint=False, bias_term=True)

    input = torch.ones([8, 512], dtype=torch.float16)
    model_str1 = torch.onnx.export_to_pretty_string(
        test_op1, (input), "WoquRowParallelLinear1.onnx", opset_version=11)

    print(model_str1)
    torch.onnx.export(
        test_op1,
        (input),
        "WoquRowParallelLinear1.onnx",
        input_names=['input'],
        output_names=['out'],
        opset_version=11,)
