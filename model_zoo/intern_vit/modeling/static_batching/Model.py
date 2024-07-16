import sys
import os

import torch
from torch import nn
import torch.distributed as dist

from typing import Mapping, Any, Optional

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../..")

import torch_function as OPMX

import clip_vit.modeling.Params as Params
import ModelUtils
from ModelParallel import ColumnParallelLinear, RowParallelLinear
from ModelLayers import Linear, RMSNorm, SkipRMSNorm, LayerNorm


TensorDumper = ModelUtils.__TensorDumper__()


class InternVL_MLP(nn.Module):
    def __init__(self,
                 args: Params.ViTParams,
                 linear_bias_term: bool,
                 proc_group: dist.ProcessGroup):
        super().__init__()

        self.vit_hidden_dim = args.hidden_dim
        self.llm_hidden_dim = args.llm_hidden_dim
        self.downsample_ratio = args.downsample_ratio
        self.h_w = args.image_size // args.patch_size
        self.input_dim = self.vit_hidden_dim * self.downsample_ratio ** 2

        self.layernorm = LayerNorm(self.input_dim, eps=args.norm_eps)
        self.w1 = ColumnParallelLinear(proc_group, self.input_dim, self.llm_hidden_dim,
                                       bias_term=linear_bias_term, gather_output=False)
        self.w2 = RowParallelLinear(proc_group, self.llm_hidden_dim, self.llm_hidden_dim,
                                    bias_term=linear_bias_term, input_is_parallel=True)

    def forward(self, x):
        #h = w = int(x.shape[1] ** 0.5)
        # transform x to nhwc data layout
        x = OPMX.reshape(x, (0, self.h_w, self.h_w, self.vit_hidden_dim))
        x = OPMX.pixel_unshuffle(x, self.downsample_ratio, 'nhwc')
        x = OPMX.reshape(x, (0, -1, self.input_dim))

        x = self.w1(x)
        x = OPMX.gelu(x, approximate=False)
        output = self.w2(x)
        return output


class VisionEmbeddings(torch.nn.Module):
    def __init__(self, hidden_dim: int, image_size: int, patch_size: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_positions = (self.image_size // self.patch_size) ** 2 + 1

        self.cls_emb_weight = nn.Parameter(torch.randn(self.hidden_dim))
        self.patch_emb_weight  = nn.Parameter(torch.randn([self.hidden_dim, 3, patch_size, patch_size]))
        self.patch_emb_bias  = nn.Parameter(torch.randn([self.hidden_dim]))
        self.pos_emb_weight = nn.Parameter(torch.randn(self.num_positions, self.hidden_dim))

    def forward(self, pixel_values: torch.Tensor):
        return OPMX.vision_embedding(pixel_values, self.cls_emb_weight, self.patch_emb_weight, self.pos_emb_weight, self.patch_emb_bias, self.hidden_dim, self.patch_size)


class Attention(nn.Module):
    def __init__(
            self,
            args: Params.ViTParams,
            layer_id: int,
            fused_qkv: bool,
            attn_wqkv_bias_term: bool,
            attn_wo_bias_term: bool,
            proc_group: dist.ProcessGroup):
        super().__init__()

        world_size = 1 if proc_group is None else proc_group.size()

        self.num_kv_heads = args.num_kv_heads
        self.num_local_heads = args.num_heads // world_size
        self.num_local_kv_heads = self.num_kv_heads // world_size
        self.num_local_kv_repeats = self.num_local_heads // self.num_local_kv_heads
        self.head_dim = args.hidden_dim // args.num_heads
        self.num_layers = args.num_layers
        self.layer_id = layer_id
        self.fused_qkv = fused_qkv

        if self.fused_qkv:
            self.wqkv = ColumnParallelLinear(
                proc_group, args.hidden_dim, args.hidden_dim + 2 * self.num_kv_heads * self.head_dim,
                bias_term=attn_wqkv_bias_term, gather_output=False)
        else:
            self.wq = ColumnParallelLinear(
                proc_group, args.hidden_dim, args.hidden_dim,
                bias_term=attn_wqkv_bias_term, gather_output=False)
            self.wk = ColumnParallelLinear(
                proc_group, args.hidden_dim, self.num_kv_heads * self.head_dim,
                bias_term=attn_wqkv_bias_term, gather_output=False)
            self.wv = ColumnParallelLinear(
                proc_group, args.hidden_dim, self.num_kv_heads * self.head_dim,
                bias_term=attn_wqkv_bias_term, gather_output=False)
        self.wo = RowParallelLinear(
            proc_group, args.hidden_dim, args.hidden_dim,
            bias_term=attn_wo_bias_term, input_is_parallel=True)

        self.q_norm = RMSNorm(args.hidden_dim, eps=args.norm_eps)
        self.k_norm = RMSNorm(args.hidden_dim, eps=args.norm_eps)


    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]):
        expanded_shape = (0, 0, -1, self.head_dim)
        if self.fused_qkv:
            xqkv = self.wqkv(x)
            # xqkv = OPMX.reshape(xqkv, expanded_shape)
            # TensorDumper.dump(xqkv, "layer{}_reshaped_xqkv".format(self.layer_id))
            split_size = (self.num_local_heads * self.head_dim,
                          self.num_local_kv_heads * self.head_dim,
                          self.num_local_kv_heads * self.head_dim)
            xq, xk, xv = torch.split(xqkv, split_size, -1)
        else:
            xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
            #xq = OPMX.reshape(xq, expanded_shape)
            #xk = OPMX.reshape(xk, expanded_shape)
            #xv = OPMX.reshape(xv, expanded_shape)
        # TensorDumper.dump(xq, "layer{}_reshaped_xq".format(self.layer_id))
        # TensorDumper.dump(xk, "layer{}_reshaped_xk".format(self.layer_id))
        # TensorDumper.dump(xv, "layer{}_reshaped_xv".format(self.layer_id))

        xq = self.q_norm(xq)
        xk = self.k_norm(xk)

        xq = OPMX.reshape(xq, expanded_shape)
        xk = OPMX.reshape(xk, expanded_shape)
        xv = OPMX.reshape(xv, expanded_shape)

        attn = OPMX.multi_head_attention(xq, xk, xv,
                                        attn_mask=attn_mask,
                                        num_heads=self.num_local_heads,
                                        head_dim=self.head_dim,
                                        is_causal=False,
                                        num_kv_heads=self.num_local_kv_heads)
        # TensorDumper.dump(attn, "layer{}_multi_head_attention_out".format(self.layer_id))

        output = self.wo(OPMX.reshape(attn, (0, 0, -1)))
        # TensorDumper.dump(output, "layer{}_reshaped_wo_out".format(self.layer_id))

        return output


class FeedForward(nn.Module):
    def __init__(
        self,
        args: Params.ViTParams,
        layer_id: int,
        linear_bias_term: bool,
        proc_group: dist.ProcessGroup
    ):
        super().__init__()
        self.layer_id = layer_id

        self.w1 = ColumnParallelLinear(
            proc_group, args.hidden_dim, args.intermediate_dim,
            bias_term=linear_bias_term, gather_output=False)
        self.w2 = RowParallelLinear(
            proc_group, args.intermediate_dim, args.hidden_dim,
            bias_term=linear_bias_term, input_is_parallel=True)


    def forward(self, x):
        x1 = self.w1(x)
        x1 = OPMX.gelu(x1, approximate=False)
        # TensorDumper.dump(x1, "layer{}_ffn_w1".format(self.layer_id))
        output = self.w2(x1)
        # TensorDumper.dump(output, "layer{}_ffn_w2".format(self.layer_id))
        return output


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int,
                 args: Params.ViTParams,
                 fused_qkv: bool,
                 attn_wqkv_bias_term: bool,
                 attn_wo_bias_term: bool,
                 ffn_linear_bias_term: bool,
                 proc_group: dist.ProcessGroup):
        super().__init__()
        self.attention = Attention(args,
                                   layer_id,
                                   fused_qkv,
                                   attn_wqkv_bias_term,
                                   attn_wo_bias_term,
                                   proc_group=proc_group)
        self.feed_forward = FeedForward(args,
                                        layer_id,
                                        ffn_linear_bias_term,
                                        proc_group=proc_group)

        self.layer_id = layer_id
        self.ls1 = nn.Parameter(torch.ones(args.hidden_dim))
        self.ls2 = nn.Parameter(torch.ones(args.hidden_dim))
        self.attention_norm = SkipRMSNorm(args.hidden_dim, eps=args.norm_eps)
        self.ffn_norm = SkipRMSNorm(args.hidden_dim, eps=args.norm_eps)


    def forward(self, x: torch.Tensor, skip: torch.Tensor, attn_mask: Optional[torch.Tensor]):
        norm, res1 = self.attention_norm(x, skip) # res1 = input_x when skip == None
        # TensorDumper.dump(norm, "layer{}_attention_norm_out".format(self.layer_id))
        # TensorDumper.dump(res1, "layer{}_attention_norm_skip_out".format(self.layer_id))
        attn = self.attention.forward(norm, attn_mask) * self.ls1
        norm, res2 = self.ffn_norm(attn, res1) # res2 = attn + res1
        # TensorDumper.dump(norm, "layer{}_ffn_norm_out".format(self.layer_id))
        # TensorDumper.dump(res2, "layer{}_ffn_norm_skip_out".format(self.layer_id))
        ffn = self.feed_forward.forward(norm) * self.ls2
        return ffn, res2


class VitTransformer(nn.Module):
    def __init__(self, params: Params.ViTParams,
                 with_proj_head: bool,
                 fused_qkv: bool,
                 attn_wqkv_bias_term: bool,
                 attn_wo_bias_term: bool,
                 ffn_linear_bias_term: bool,
                 proc_group: dist.ProcessGroup):
        super().__init__()
        self.params = params
        self.n_layers = params.num_layers
        self.proc_group = proc_group
        self.with_proj_head = with_proj_head
        self.fused_qkv = fused_qkv

        world_size = 1 if proc_group is None else proc_group.size()
        num_kv_heads = params.num_heads if params.num_kv_heads is None else params.num_kv_heads
        num_local_heads = params.num_heads // world_size
        num_local_kv_heads = num_kv_heads // world_size
        head_dim = params.hidden_dim // params.num_heads
        self.local_q_dim = num_local_heads * head_dim
        self.local_kv_dim = num_local_kv_heads * head_dim

        self.vision_embeddings = VisionEmbeddings(params.hidden_dim, params.image_size, params.patch_size)
        #self.pre_layernorm = LayerNorm(params.hidden_dim, eps=params.norm_eps)
        #self.post_layernorm = LayerNorm(params.hidden_dim, eps=params.norm_eps)

        if self.with_proj_head:
            self.vision_projection = InternVL_MLP(params, linear_bias_term=True, proc_group=self.proc_group)
            #self.vision_projection = ColumnParallelLinear(proc_group, params.hidden_dim, params.hidden_dim, bias_term=False, gather_output=True)


        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.num_layers):
            self.layers.append(TransformerBlock(
                layer_id, params,
                fused_qkv,
                attn_wqkv_bias_term,
                attn_wo_bias_term,
                ffn_linear_bias_term,
                proc_group=proc_group))


    @torch.inference_mode()
    def forward(self, pixel_values: torch.Tensor, attn_mask: Optional[torch.Tensor]):
        TensorDumper.dump(pixel_values, "pixel_values")
        if attn_mask is not None:
            TensorDumper.dump(attn_mask, "attn_mask")

        h = self.vision_embeddings(pixel_values)
        # TensorDumper.dump(h, "emb_out")
        # h = self.pre_layernorm(h)
        # TensorDumper.dump(h, "pre_layernorm_out")

        norm = None
        for layer in self.layers:
            h, norm = layer(h, norm, attn_mask)

        # output = (norm + h)[:, 0, :] # get cls token
        # TensorDumper.dump(pooled_output, "pooled_output")

        if self.with_proj_head:
            output = self.vision_projection((norm+h)[:, 1:, :])
            # TensorDumper.dump(output, "vision_proj_out")
        TensorDumper.dump(output, "logits")
        return output

    @torch.no_grad()
    def load_state_dict(self, state_dict: Mapping[str, Any]):
        loaded_params = set()
        model_params = {key: value for key, value in self.named_parameters()}

        for key, value in state_dict.items():
            module_name, param_name = key.rsplit(".", 1)

            if key in model_params:
                self.get_submodule(module_name)._parameters[param_name][:] = value
                loaded_params.add(key)
                print(f'Loaded: {key} -> {key}[{value.shape}]')

            try:
                if self.fused_qkv:
                    if 'attention.wq' in key:
                        loaded_params.add(key)
                        module_name = module_name.replace('wq', 'wqkv')
                        self.get_submodule(module_name)._parameters[param_name][
                            :self.local_q_dim] = value
                        replaced_key = module_name + '.' + param_name
                        print(f'Loaded: {key} -> {replaced_key}[{value.shape}]')
                    elif 'attention.wk' in key:
                        loaded_params.add(key)
                        module_name = module_name.replace('wk', 'wqkv')
                        self.get_submodule(module_name)._parameters[param_name][
                            self.local_q_dim:self.local_q_dim + self.local_kv_dim] = value
                        replaced_key = module_name + '.' + param_name
                        print(f'Loaded: {key} -> {replaced_key}[{value.shape}]')
                    elif 'attention.wv' in key:
                        loaded_params.add(key)
                        module_name = module_name.replace('wv', 'wqkv')
                        self.get_submodule(module_name)._parameters[param_name][
                            self.local_q_dim + self.local_kv_dim:
                            self.local_q_dim + self.local_kv_dim * 2] = value
                        replaced_key = module_name + '.' + param_name
                        print(f'Loaded: {key} -> {replaced_key}[{value.shape}]')
            except AttributeError as e:
                raise Exception(f'Failed to inject model weight {key}, can not find corresponding layer.')

        for key in state_dict:
            if key not in loaded_params:
                print(f'{key} is not loaded.')
