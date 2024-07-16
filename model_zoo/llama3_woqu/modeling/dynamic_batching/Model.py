import sys
import os

from tqdm import tqdm
import torch
from torch import nn
import torch.distributed as dist

from typing import Mapping, Any, Optional

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../..")

import torch_function as OPMX
from ModelParams import ModelParams
import ModelUtils
from ModelParallel import WoquColumnParallelLinear, ColumnParallelLinear, WoquRowParallelLinear, ParallelEmbedding
from ModelLayers import SkipRMSNorm

TensorDumper = ModelUtils.__TensorDumper__()

class Attention(nn.Module):
    def __init__(
            self,
            args: ModelParams,
            layer_id: int,
            friendly_gqa: bool,
            fused_qkv: bool,
            fused_kvcache: bool,
            fused_alibi: bool,
            with_rope: bool,
            with_alibi: bool,
            attn_wqkv_bias_term: bool,
            attn_wo_bias_term: bool,
            rotary_dim: int,
            # quant
            quant_data_type: str,
            quant_method: str,
            quant_axis: int,
            group_size: int,
            storage_bits: int,
            has_zeropoint: bool, 
            float_zeropoint: bool,
            proc_group: dist.ProcessGroup):
        super().__init__()

        world_size = 1 if proc_group is None else proc_group.size()

        self.num_kv_heads = args.num_heads if args.num_kv_heads is None else args.num_kv_heads
        self.num_local_heads = args.num_heads // world_size
        self.num_local_kv_heads = self.num_kv_heads // world_size
        self.num_local_kv_repeats = self.num_local_heads // self.num_local_kv_heads
        self.head_dim = args.hidden_dim // args.num_heads
        self.num_layers = args.num_layers
        self.layer_id = layer_id
        self.cache_quant_bit = args.cache_quant_bit
        self.cache_quant_group = args.cache_quant_group
        self.cache_layout = args.cache_layout
        self.cache_mode = args.cache_mode
        self.page_size = args.page_size

        self.friendly_gqa = friendly_gqa
        self.fused_qkv = fused_qkv
        self.fused_kvcache = fused_kvcache
        self.auto_causal = args.auto_causal

        self.with_alibi = with_alibi
        self.fused_alibi = fused_alibi

        self.with_rope = with_rope
        self.rotary_dim = rotary_dim
        self.rope_theta = args.rope_theta
        self.rope_scaling_type = args.rope_scaling_type
        self.rope_scaling_factor = args.rope_scaling_factor
        self.max_position_embeddings = args.max_position_embeddings
        
        if self.fused_qkv:
            self.wqkv = WoquColumnParallelLinear(
                proc_group, args.hidden_dim, args.hidden_dim + 2 * self.num_kv_heads * self.head_dim,
                bias_term=attn_wqkv_bias_term, quant_data_type=quant_data_type, quant_method=quant_method, 
                quant_axis=quant_axis, group_size=group_size, storage_bits=storage_bits, has_zeropoint=has_zeropoint, 
                float_zeropoint=float_zeropoint, gather_output=False)
        else:
            self.wq = WoquColumnParallelLinear(
                proc_group, args.hidden_dim, args.hidden_dim,
                bias_term=attn_wqkv_bias_term, quant_data_type=quant_data_type, quant_method=quant_method, 
                quant_axis=quant_axis, group_size=group_size, storage_bits=storage_bits, has_zeropoint=has_zeropoint, 
                float_zeropoint=float_zeropoint, gather_output=False)
            self.wk = WoquColumnParallelLinear(
                proc_group, args.hidden_dim, self.num_kv_heads * self.head_dim,
                bias_term=attn_wqkv_bias_term, quant_data_type=quant_data_type, quant_method=quant_method, 
                quant_axis=quant_axis, group_size=group_size, storage_bits=storage_bits, has_zeropoint=has_zeropoint, 
                float_zeropoint=float_zeropoint, gather_output=False)
            self.wv = WoquColumnParallelLinear(
                proc_group, args.hidden_dim, self.num_kv_heads * self.head_dim,
                bias_term=attn_wqkv_bias_term, quant_data_type=quant_data_type, quant_method=quant_method, 
                quant_axis=quant_axis, group_size=group_size, storage_bits=storage_bits, has_zeropoint=has_zeropoint, 
                float_zeropoint=float_zeropoint, gather_output=False)
            
        self.wo = WoquRowParallelLinear(
            proc_group, args.hidden_dim, args.hidden_dim,
            bias_term=attn_wo_bias_term, quant_data_type=quant_data_type, quant_method=quant_method, 
            quant_axis=quant_axis, group_size=group_size, storage_bits=storage_bits, has_zeropoint=has_zeropoint, 
            float_zeropoint=float_zeropoint,input_is_parallel=True)


    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor],
                seqstarts: torch.Tensor, kvstarts: torch.Tensor, cachestarts: torch.Tensor,
                decoding_batches: torch.Tensor, start_pos: torch.Tensor,
                max_seqlen: torch.Tensor, max_kvlen: torch.Tensor,
                kv_cache: torch.Tensor, kv_scale: torch.Tensor = None):
        expanded_shape = (0, -1, self.head_dim)
        if self.fused_qkv:
            xqkv = self.wqkv(x)
            xqkv = OPMX.reshape(xqkv, expanded_shape)
            split_size = (self.num_local_heads, self.num_local_kv_heads, self.num_local_kv_heads)
            # TensorDumper.dump(xqkv, "layer{}_reshaped_xqkv".format(self.layer_id))
            xq, xk, xv = torch.split(xqkv, split_size, -2)
        else:
            xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
            xq = OPMX.reshape(xq, expanded_shape)
            xk = OPMX.reshape(xk, expanded_shape)
            xv = OPMX.reshape(xv, expanded_shape)
        # TensorDumper.dump(xq, "layer{}_reshaped_xq".format(self.layer_id))
        # TensorDumper.dump(xk, "layer{}_reshaped_xk".format(self.layer_id))
        # TensorDumper.dump(xv, "layer{}_reshaped_xv".format(self.layer_id))

        if self.with_rope:
            xq, xk = OPMX.dynamic_batching.rotary_position_embedding(
                                            xq, xk, seqstarts,
                                            start_pos, max_seqlen,
                                            rotary_dim=self.rotary_dim,
                                            max_position_embeddings=self.max_position_embeddings,
                                            theta=self.rope_theta, scaling_type=self.rope_scaling_type,
                                            scaling_factor=self.rope_scaling_factor)
            # TensorDumper.dump(xq, "layer{}_rotary_position_embedding_out_xq".format(self.layer_id))
            # TensorDumper.dump(xk, "layer{}_rotary_position_embedding_out_xk".format(self.layer_id))

        if self.fused_kvcache:
            attn = OPMX.dynamic_batching.multi_head_cache_attention(
                xq, xk, xv, seqstarts, kvstarts,
                cachestarts, start_pos,
                decoding_batches,
                max_seqlen, max_kvlen,
                kv_cache, kv_scale,
                attn_mask=attn_mask,
                num_heads=self.num_local_heads,
                head_dim=self.head_dim,
                is_causal=self.auto_causal,
                is_alibi=self.with_alibi and self.fused_alibi,
                num_kv_heads=self.num_local_kv_heads,
                num_layer=self.num_layers,
                layer_idx=self.layer_id,
                quant_bit=self.cache_quant_bit,
                quant_group=self.cache_quant_group,
                cache_mode=self.cache_mode,
                cache_layout=self.cache_layout,
                page_size=self.page_size)
        else:
            keys, values = OPMX.dynamic_batching.key_value_cache(
                                            xk, xv, seqstarts, kvstarts,
                                            cachestarts, start_pos,
                                            max_seqlen, max_kvlen,
                                            kv_cache, kv_scale,
                                            num_layer=self.num_layers,
                                            layer_idx=self.layer_id,
                                            quant_bit=self.cache_quant_bit,
                                            quant_group=self.cache_quant_group,
                                            num_repeat=self.num_local_kv_repeats if self.friendly_gqa else 1,
                                            cache_mode=self.cache_mode,
                                            cache_layout=self.cache_layout,
                                            page_size=self.page_size)
            # TensorDumper.dump(kv_cache, "layer{}_modified_kv_cache".format(self.layer_id))
            # TensorDumper.dump(kv_scale, "layer{}_modified_kv_scale".format(self.layer_id))
            # TensorDumper.dump(keys, "layer{}_key_value_cache_out_keys".format(self.layer_id))
            # TensorDumper.dump(values, "layer{}_key_value_cache_out_values".format(self.layer_id))
            attn = OPMX.dynamic_batching.multi_head_attention(
                                            xq, keys, values,
                                            seqstarts, kvstarts,
                                            decoding_batches,
                                            max_seqlen, max_kvlen,
                                            attn_mask=attn_mask,
                                            num_heads=self.num_local_heads,
                                            head_dim=self.head_dim,
                                            is_causal=self.auto_causal,
                                            is_alibi=self.with_alibi and self.fused_alibi,
                                            num_kv_heads=0 if self.friendly_gqa else self.num_local_kv_heads)
        attn = OPMX.reshape(attn, (0, -1))
        # TensorDumper.dump(attn, "layer{}_multi_head_attention_out".format(self.layer_id))

        output = self.wo(attn)
        # TensorDumper.dump(output, "layer{}_reshaped_wo_out".format(self.layer_id))

        return output


class FeedForward(nn.Module):
    def __init__(
        self,
        args: ModelParams,
        layer_id: int,
        fused_ffn_glu: bool,
        linear_bias_term: bool,
        # quant
        quant_data_type: str,
        quant_method: str,
        quant_axis: int,
        group_size: int,
        storage_bits: int,
        has_zeropoint: bool, 
        float_zeropoint: bool,
        proc_group: dist.ProcessGroup
    ):
        super().__init__()
        self.layer_id = layer_id
        self.fused_ffn_glu = fused_ffn_glu

        if self.fused_ffn_glu:
            self.wu = WoquColumnParallelLinear(
                proc_group, args.hidden_dim, 2 * args.intermediate_dim,
                bias_term=linear_bias_term, quant_data_type=quant_data_type, quant_method=quant_method,
                quant_axis=quant_axis, group_size=group_size, storage_bits=storage_bits, has_zeropoint=has_zeropoint, 
                float_zeropoint=float_zeropoint, gather_output=False)
        else:
            self.w1 = WoquColumnParallelLinear(
                proc_group, args.hidden_dim, args.intermediate_dim,
                bias_term=linear_bias_term, quant_data_type=quant_data_type, quant_method=quant_method,
                quant_axis=quant_axis, group_size=group_size, storage_bits=storage_bits, has_zeropoint=has_zeropoint, 
                float_zeropoint=float_zeropoint, gather_output=False)
            self.w3 = WoquColumnParallelLinear(
                proc_group, args.hidden_dim, args.intermediate_dim,
                bias_term=linear_bias_term, quant_data_type=quant_data_type, quant_method=quant_method,
                quant_axis=quant_axis, group_size=group_size, storage_bits=storage_bits, has_zeropoint=has_zeropoint, 
                float_zeropoint=float_zeropoint, gather_output=False)
            
        self.w2 = WoquRowParallelLinear(
            proc_group, args.intermediate_dim, args.hidden_dim,
            bias_term=linear_bias_term, quant_data_type=quant_data_type, quant_method=quant_method,
            quant_axis=quant_axis, group_size=group_size, storage_bits=storage_bits, has_zeropoint=has_zeropoint, 
            float_zeropoint=float_zeropoint, input_is_parallel=True)


    def forward(self, x):
        if self.fused_ffn_glu:
            x13 = self.wu(x)
            # TensorDumper.dump(x13, "layer{}_ffn_wu".format(self.layer_id))
            x13 = OPMX.swiglu(x13)
        else:
            x1 = self.w1(x)
            # TensorDumper.dump(x1, "layer{}_ffn_w1".format(self.layer_id))
            x3 = self.w3(x)
            # TensorDumper.dump(x3, "layer{}_ffn_w3".format(self.layer_id))
            x13 = OPMX.silu(x1, x3)
        # TensorDumper.dump(x13, "layer{}_ffn_mul_silu".format(self.layer_id))
        output = self.w2(x13)
        # TensorDumper.dump(output, "layer{}_ffn_w2".format(self.layer_id))
        return output


class TransformerBlock(nn.Module):
    def __init__(self, layer_id: int,
                 args: ModelParams,
                 friendly_gqa: bool,
                 fused_qkv: bool,
                 fused_kvcache: bool,
                 fused_ffn_glu: bool,
                 fused_alibi: bool,
                 with_rope: bool,
                 with_alibi: bool,
                 attn_wqkv_bias_term: bool,
                 attn_wo_bias_term: bool,
                 ffn_linear_bias_term: bool,
                 rotary_dim: int,
                 # quant
                 quant_data_type: str,
                 quant_method: str,
                 quant_axis: int,
                 group_size: int,
                 storage_bits: int,
                 has_zeropoint: bool, 
                 float_zeropoint: bool,
                 proc_group: dist.ProcessGroup):
        super().__init__()
        self.attention = Attention(args,
                                   layer_id,
                                   friendly_gqa,
                                   fused_qkv,
                                   fused_kvcache,
                                   fused_alibi,
                                   with_rope,
                                   with_alibi,
                                   attn_wqkv_bias_term,
                                   attn_wo_bias_term,
                                   rotary_dim=rotary_dim,
                                   # quant
                                   quant_data_type=quant_data_type,
                                   quant_method=quant_method,
                                   quant_axis=quant_axis,
                                   group_size=group_size,
                                   storage_bits=storage_bits,
                                   has_zeropoint=has_zeropoint, 
                                   float_zeropoint=float_zeropoint,
                                   proc_group=proc_group)
        self.feed_forward = FeedForward(args,
                                        layer_id,
                                        fused_ffn_glu,
                                        ffn_linear_bias_term,
                                        # quant
                                        quant_data_type=quant_data_type,
                                        quant_method=quant_method,
                                        quant_axis=quant_axis,
                                        group_size=group_size,
                                        storage_bits=storage_bits,
                                        has_zeropoint=has_zeropoint, 
                                        float_zeropoint=float_zeropoint,
                                        proc_group=proc_group)

        self.layer_id = layer_id
        self.attention_norm = SkipRMSNorm(args.hidden_dim, eps=args.norm_eps)
        self.ffn_norm = SkipRMSNorm(args.hidden_dim, eps=args.norm_eps)


    def forward(self, x: torch.Tensor, skip: torch.Tensor, attn_mask: Optional[torch.Tensor],
                seqstarts: torch.Tensor, kvstarts: torch.Tensor, cachestarts: torch.Tensor,
                decoding_batches: torch.Tensor, start_pos: torch.Tensor,
                max_seqlen: torch.Tensor, max_kvlen: torch.Tensor,
                kv_cache: torch.Tensor, kv_sacle: torch.Tensor = None):
        norm, x = self.attention_norm(x, skip)
        # TensorDumper.dump(norm, "layer{}_attention_norm_out".format(self.layer_id))
        # TensorDumper.dump(x, "layer{}_attention_norm_skip_out".format(self.layer_id))
        attn = self.attention.forward(norm, attn_mask, seqstarts, kvstarts,
                                      cachestarts, decoding_batches,
                                      start_pos, max_seqlen, max_kvlen,
                                      kv_cache, kv_sacle)
        norm, h = self.ffn_norm(x, attn)
        # TensorDumper.dump(norm, "layer{}_ffn_norm_out".format(self.layer_id))
        # TensorDumper.dump(h, "layer{}_ffn_norm_skip_out".format(self.layer_id))
        ffn = self.feed_forward.forward(norm)
        return h, ffn


class Transformer(nn.Module):
    def __init__(self, params: ModelParams,
                 friendly_gqa: bool,
                 fused_qkv: bool,
                 fused_kvcache: bool,
                 fused_ffn_glu: bool,
                 fused_alibi: bool,
                 with_rope: bool,
                 with_alibi: bool,
                 attn_wqkv_bias_term: bool,
                 attn_wo_bias_term: bool,
                 ffn_linear_bias_term: bool,
                 rotary_dim: int,
                 # quant
                 quant_data_type: str,
                 quant_method: str,
                 quant_axis: int,
                 group_size: int,
                 storage_bits: int,
                 has_zeropoint: bool, 
                 float_zeropoint: bool,
                 proc_group: dist.ProcessGroup):
        super().__init__()
        self.params = params
        self.vocab_size = params.vocab_size
        self.n_layers = params.num_layers
        self.proc_group = proc_group
        self.fused_qkv = fused_qkv
        self.fused_kvcache = fused_kvcache
        self.fused_ffn_glu = fused_ffn_glu

        self.with_alibi = with_alibi
        self.fused_alibi = fused_alibi

        world_size = 1 if proc_group is None else proc_group.size()
        num_kv_heads = params.num_heads if params.num_kv_heads is None else params.num_kv_heads
        num_local_heads = params.num_heads // world_size
        num_local_kv_heads = num_kv_heads // world_size
        head_dim = params.hidden_dim // params.num_heads
        self.local_q_dim = num_local_heads * head_dim
        self.local_kv_dim = num_local_kv_heads * head_dim
        self.local_imm_dim = params.intermediate_dim // world_size
        if quant_data_type == "int4":
            self.local_q_quant_dim = num_local_heads * head_dim // (storage_bits // 4)
            self.local_kv_quant_dim = num_local_kv_heads * head_dim // (storage_bits // 4)
            self.local_imm_quant_dim = params.intermediate_dim // world_size // (storage_bits // 4)
        elif quant_data_type == "int8":
            self.local_q_quant_dim = num_local_heads * head_dim // (storage_bits // 8)
            self.local_kv_quant_dim = num_local_kv_heads * head_dim // (storage_bits // 8)
            self.local_imm_quant_dim = params.intermediate_dim // world_size // (storage_bits // 8)
        else:
            raise ValueError("quant_data_type must be one of int4 or int8")


        self.tok_embeddings = ParallelEmbedding(proc_group, params.vocab_size, params.hidden_dim)

        self.layers = torch.nn.ModuleList()
        for layer_id in range(params.num_layers):
            self.layers.append(TransformerBlock(
                layer_id, params,
                friendly_gqa,
                fused_qkv,
                fused_kvcache,
                fused_ffn_glu,
                fused_alibi,
                with_rope,
                with_alibi,
                attn_wqkv_bias_term,
                attn_wo_bias_term,
                ffn_linear_bias_term,
                rotary_dim,
                # quant
                quant_data_type,
                quant_method,
                quant_axis,
                group_size,
                storage_bits,
                has_zeropoint, 
                float_zeropoint,
                proc_group=proc_group))

        self.norm = SkipRMSNorm(params.hidden_dim, eps=params.norm_eps)
        self.output = ColumnParallelLinear(proc_group, params.hidden_dim, params.vocab_size, bias_term=False)


    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, attn_mask: Optional[torch.Tensor],
                seqstarts: torch.Tensor, kvstarts: torch.Tensor,
                cachestarts: torch.Tensor, decoding_batches: torch.Tensor,
                start_pos: torch.Tensor, max_seqlen: torch.Tensor,  max_kvlen: torch.Tensor,
                kv_cache: torch.Tensor, kv_scale: torch.Tensor = None):
        h = self.tok_embeddings(tokens)
        # TensorDumper.dump(h, "emb_out")

        _kv_scale = kv_scale
        TensorDumper.dump(tokens, "token_ids")
        if attn_mask is not None:
            TensorDumper.dump(attn_mask, "attn_mask")
        if self.fused_kvcache and attn_mask is not None:
            if kv_scale is None: # mount an empty scale for friendly exporting
                _kv_scale = torch.empty(0, dtype=h.dtype)
        TensorDumper.dump(seqstarts, "seqstarts")
        TensorDumper.dump(kvstarts, "kvstarts")
        TensorDumper.dump(cachestarts, "cachestarts")
        TensorDumper.dump(decoding_batches, "decoding_batches")
        TensorDumper.dump(start_pos, "start_pos")
        TensorDumper.dump(max_seqlen, "max_seqlen")
        TensorDumper.dump(max_kvlen, "max_kvlen")
        TensorDumper.dump(kv_cache, "kv_cache")
        if kv_scale is not None:
            TensorDumper.dump(kv_scale, "kv_scale")

        if self.with_alibi and not self.fused_alibi:
            attn_mask = OPMX.dynamic_batching.alibi_mask(seqstarts, kvstarts, attn_mask, self.params.num_heads, h.dtype)
            # TensorDumper.dump(attn_mask, "alibi_mask")

        norm = None
        for layer in self.layers:
            h, norm = layer(h, norm, attn_mask, seqstarts, kvstarts, cachestarts,
                            decoding_batches, start_pos, max_seqlen, max_kvlen,
                            kv_cache, _kv_scale)

        h, norm = self.norm(h, norm)
        # TensorDumper.dump(h, "last_rms_norm")
        gathered_h = torch.index_select(h, 0, seqstarts[1:] - 1)
        # TensorDumper.dump(gathered_h, "gathered_h")
        output = self.output(gathered_h)  # only compute last logits
        # TensorDumper.dump(output, "logits_before_cast")
        output = output.float()
        TensorDumper.dump(output, "logits")
        return output


    @torch.no_grad()
    def load_state_dict(self, state_dict: Mapping[str, Any]):
        loaded = set()
        model_params = {key: value for key, value in self.named_parameters()}
        model_buffers = {key: value for key, value in self.named_buffers()}
            
        for key, value in tqdm(state_dict.items(), desc="Processing layers"):
            module_name, param_name = key.rsplit(".", 1)
            if key in model_params:
                self.get_submodule(module_name)._parameters[param_name][:] = value
                loaded.add(key)
                # print(f'Loaded: {key} -> {key}[{value.shape}]')
            elif key in model_buffers:
                self.get_submodule(module_name)._buffers[param_name][:] = value
                loaded.add(key)
                # print(f'Loaded: {key} -> {key}[{value.shape}]')

            try:
                if self.fused_qkv:
                    if 'attention.wq' in key:
                        loaded.add(key)
                        module_name = module_name.replace('wq', 'wqkv')
                        if param_name == "qweight":
                            self.get_submodule(module_name)._buffers[param_name][
                                :self.local_q_quant_dim] = value
                            replaced_key = module_name + '.' + param_name
                            # print(f'Loaded: {key} -> {replaced_key}[{value.shape}]')
                        elif param_name == "zeropoint":
                            self.get_submodule(module_name)._buffers[param_name][
                                :self.local_q_dim] = value
                            replaced_key = module_name + '.' + param_name
                            # print(f'Loaded: {key} -> {replaced_key}[{value.shape}]')
                        else:
                            self.get_submodule(module_name)._parameters[param_name][
                                :self.local_q_dim] = value
                            replaced_key = module_name + '.' + param_name
                            # print(f'Loaded: {key} -> {replaced_key}[{value.shape}]')
                    elif 'attention.wk' in key:
                        loaded.add(key)
                        module_name = module_name.replace('wk', 'wqkv')
                        if param_name == "qweight":
                            self.get_submodule(module_name)._buffers[param_name][
                                self.local_q_quant_dim:self.local_q_quant_dim + self.local_kv_quant_dim] = value
                            replaced_key = module_name + '.' + param_name
                            # print(f'Loaded: {key} -> {replaced_key}[{value.shape}]')
                        elif param_name == "zeropoint":
                            self.get_submodule(module_name)._buffers[param_name][
                                self.local_q_dim:self.local_q_dim + self.local_kv_dim] = value
                            replaced_key = module_name + '.' + param_name
                            # print(f'Loaded: {key} -> {replaced_key}[{value.shape}]')
                        else:
                            self.get_submodule(module_name)._parameters[param_name][
                                self.local_q_dim:self.local_q_dim + self.local_kv_dim] = value
                            replaced_key = module_name + '.' + param_name
                            # print(f'Loaded: {key} -> {replaced_key}[{value.shape}]')
                    elif 'attention.wv' in key:
                        loaded.add(key)
                        module_name = module_name.replace('wv', 'wqkv')
                        if param_name == "qweight":
                            self.get_submodule(module_name)._buffers[param_name][
                                self.local_q_quant_dim + self.local_kv_quant_dim:
                                self.local_q_quant_dim + self.local_kv_quant_dim * 2] = value
                            replaced_key = module_name + '.' + param_name
                            # print(f'Loaded: {key} -> {replaced_key}[{value.shape}]')
                        elif param_name == "zeropoint":
                            self.get_submodule(module_name)._buffers[param_name][
                                self.local_q_dim + self.local_kv_dim:
                                self.local_q_dim + self.local_kv_dim * 2] = value
                            replaced_key = module_name + '.' + param_name
                            # print(f'Loaded: {key} -> {replaced_key}[{value.shape}]')
                        else:
                            self.get_submodule(module_name)._parameters[param_name][
                                self.local_q_dim + self.local_kv_dim:
                                self.local_q_dim + self.local_kv_dim * 2] = value
                            replaced_key = module_name + '.' + param_name
                            # print(f'Loaded: {key} -> {replaced_key}[{value.shape}]')
                if self.fused_ffn_glu:
                    if 'feed_forward.w1' in key:
                        loaded.add(key)
                        if param_name == "qweight":
                            module_name = module_name.replace('w1', 'wu')
                            self.get_submodule(module_name)._buffers[param_name][
                                :self.local_imm_quant_dim] = value
                            replaced_key = module_name + '.' + param_name
                            # print(f'Loaded: {key} -> {replaced_key}[{value.shape}]')
                        elif param_name == "zeropoint":
                            module_name = module_name.replace('w1', 'wu')
                            self.get_submodule(module_name)._buffers[param_name][
                                :self.local_imm_dim] = value
                            replaced_key = module_name + '.' + param_name
                            # print(f'Loaded: {key} -> {replaced_key}[{value.shape}]')
                        else:
                            module_name = module_name.replace('w1', 'wu')
                            self.get_submodule(module_name)._parameters[param_name][
                                :self.local_imm_dim] = value
                            replaced_key = module_name + '.' + param_name
                            # print(f'Loaded: {key} -> {replaced_key}[{value.shape}]')
                    if 'feed_forward.w3' in key:
                        loaded.add(key)
                        module_name = module_name.replace('w3', 'wu')
                        if param_name == "qweight":
                            self.get_submodule(module_name)._buffers[param_name][
                                self.local_imm_quant_dim:] = value
                            replaced_key = module_name + '.' + param_name
                            # print(f'Loaded: {key} -> {replaced_key}[{value.shape}]')
                        elif param_name == "zeropoint":
                            self.get_submodule(module_name)._buffers[param_name][
                                self.local_imm_dim:] = value
                            replaced_key = module_name + '.' + param_name
                            # print(f'Loaded: {key} -> {replaced_key}[{value.shape}]')
                        else:
                            self.get_submodule(module_name)._parameters[param_name][
                                self.local_imm_dim:] = value
                            replaced_key = module_name + '.' + param_name
                            # print(f'Loaded: {key} -> {replaced_key}[{value.shape}]')
            except AttributeError as e:
                raise Exception(f'Failed to inject model weight {key}, can not find corresponding layer.')

        for key in state_dict:
            if key not in loaded:
                print(f'{key} is not loaded.')

    @torch.no_grad()
    def random_weight(self):
        model_params = {key: value for key, value in self.named_parameters()}
        model_buffers = {key: value for key, value in self.named_buffers()}

        for key, value in model_params.items():
            module_name, param_name = key.rsplit(".", 1)

            self.get_submodule(module_name)._parameters[param_name] = torch.randn_like(value)
            print(f'Random: {key} -> {key}[{value.shape}]')
            
        for key, value in model_buffers.items():
            module_name, param_name = key.rsplit(".", 1)

            self.get_submodule(module_name)._buffers[param_name] = torch.randn_like(value)
            print(f'Random: {key} -> {key}[{value.shape}]')
