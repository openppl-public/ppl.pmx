import torch

class GroupNorm(torch.autograd.Function):
    @staticmethod
    def symbolic(
        g, X: torch.Value, weight: torch.Value, bias: torch.Value,
        elementwise_affine: bool = False, num_groups: int = 1, eps: float = 1e-5):
        if elementwise_affine:
            Y = g.op("pmx::GroupNorm", X, weight, bias,
                        elementwise_affine_i = True,
                        num_groups_i = num_groups,
                        eps_f = eps)
        else:
            Y = g.op("pmx::GroupNorm", X,
                        elementwise_affine_i = False,
                        num_groups_i = num_groups,
                        eps_f = eps)
        return Y.setTypeAs(X)


    @staticmethod
    def forward(
        self, X: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor,
            elementwise_affine: bool = False, num_groups: int = 1, eps: float = 1e-5):
        if torch.onnx.is_in_onnx_export():
            return X

        N, C = X.shape[0], X.shape[1] # input shape -> (N, C, *)
        assert C % num_groups == 0, "Number of channels must be divisible by number of groups"

        x = X.float()
        x = x.reshape(N, num_groups, -1)
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True)

        Y = (x - mean) * torch.rsqrt(var + eps)
        Y = Y.reshape(X.shape) # output shape -> (N, C, *), same shape as input
        if elementwise_affine:
            broadcast_shape = [C if dim==1 else 1 for dim in range(X.dim())]
            weight = weight.reshape(broadcast_shape) # weight shape -> (1, C, 1, ...)
            bias = bias.reshape(broadcast_shape)
            Y = Y * weight + bias
        return Y.type_as(X)


def group_norm(X: torch.Tensor, weight: torch.Tensor = None, bias: torch.Tensor = None,
               num_groups: int = -1, eps: float = 1e-5) -> torch.Tensor:
    if weight is None or bias is None:
        elementwise_affine = False
    else:
        elementwise_affine = True
    return GroupNorm.apply(X, weight, bias, elementwise_affine, num_groups, eps)


if __name__ == "__main__":
    class TestModule1(torch.nn.Module):
        def __init__(self, num_groups: int, num_channels: int = 1, eps: float = 1e-5, ) -> None:
            super().__init__()
            self.num_groups =  num_groups
            self.eps = eps
            self.weight = torch.nn.Parameter(torch.ones(num_channels))
            self.bias = torch.nn.Parameter(torch.zeros(num_channels))


        def forward(self, X: torch.Tensor):
            return group_norm(X, self.weight, self.bias, self.num_groups, self.eps)


    test_op = TestModule1(8, 32, 1e-6)
    input = torch.ones([1, 32, 4096])

    model_str1 = torch.onnx.export_to_pretty_string(
        test_op, (input), "GroupNorm.onnx", opset_version=11)
    print(model_str1)

    torch.onnx.export(
         test_op,
         (input),
         "groupnorm.onnx",
         input_names=['input'],
         output_names=['out'],
         opset_version=11,
     )

