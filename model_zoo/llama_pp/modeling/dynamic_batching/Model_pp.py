import sys
import os

import torch
from torch import nn
import torch.distributed as dist

from typing import Mapping, Any, Optional

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../..")

import torch_function as PMX
from ModelParams import ModelParams
import ModelUtils
from ModelParallel import ColumnParallelLinear, RowParallelLinear, ParallelEmbedding, DistMapping
from llama.modeling.dynamic_batching.Model import RMSNorm, Attention, FeedForward

TensorDumper = ModelUtils.__TensorDumper__()

class TransformerBlockPP(nn.Module):
    def __init__(self, layer_id: int,
                 args: ModelParams,
                 friendly_gqa: bool,
                 fused_qkv: bool,
                 fused_kvcache: bool,
                 fused_ffn_glu: bool,
                 attn_wqkv_bias_term: bool,
                 attn_wo_bias_term: bool,
                 ffn_linear_bias_term: bool,
                 rotary_dim: int,
                 tp_proc_group: dist.ProcessGroup):
        super().__init__()
        self.attention = Attention(args,
                                   layer_id,
                                   friendly_gqa,
                                   fused_qkv,
                                   fused_kvcache,
                                   attn_wqkv_bias_term,
                                   attn_wo_bias_term,
                                   rotary_dim=rotary_dim,
                                   proc_group=tp_proc_group)
        self.feed_forward = FeedForward(args, 
                                        layer_id,
                                        fused_ffn_glu,
                                        ffn_linear_bias_term,
                                        proc_group=tp_proc_group)

        self.layer_id = layer_id
        self.attention_norm = RMSNorm(args.hidden_dim, eps=args.norm_eps)
        self.ffn_norm = RMSNorm(args.hidden_dim, eps=args.norm_eps)


    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor],
                seqstarts: torch.Tensor, kvstarts: torch.Tensor, cachestarts: torch.Tensor,
                decoding_batches: torch.Tensor, start_pos: torch.Tensor,
                max_seqlen: torch.Tensor, max_kvlen: torch.Tensor,
                kv_cache: torch.Tensor, kv_sacle: torch.Tensor = None):
        residual = x
        norm = self.attention_norm(x)
        TensorDumper.dump(norm, "layer{}_attention_norm_out".format(self.layer_id))
        attn = self.attention.forward(norm, attn_mask, seqstarts, kvstarts,
                                      cachestarts, decoding_batches,
                                      start_pos, max_seqlen, max_kvlen,
                                      kv_cache, kv_sacle)
        hidden_states = residual + attn
        residual = hidden_states

        norm = self.ffn_norm(hidden_states)
        TensorDumper.dump(norm, "layer{}_ffn_norm_out".format(self.layer_id))
        ffn = self.feed_forward.forward(norm)

        hidden_states = residual + ffn
        TensorDumper.dump(hidden_states, "layer{}_block_output".format(self.layer_id))
        return hidden_states

class Transformer(nn.Module):
    def __init__(self, params: ModelParams,
                 friendly_gqa: bool,
                 fused_qkv: bool,
                 fused_kvcache: bool,
                 fused_ffn_glu: bool,
                 attn_wqkv_bias_term: bool,
                 attn_wo_bias_term: bool,
                 ffn_linear_bias_term: bool,
                 rotary_dim: int,
                 dist_mapping: DistMapping):
        super().__init__()
        self.params = params
        self.dist_mapping = dist_mapping
        self.vocab_size = params.vocab_size
        
        self.num_layers = params.num_layers if not self.dist_mapping.has_pp() else (params.num_layers) // self.dist_mapping.pp_size
        self.layer_range = list(range(dist_mapping.pp_rank * self.num_layers, (dist_mapping.pp_rank + 1) * self.num_layers, 1))

        self.tp_proc_group = dist_mapping.tp_proc_group
        self.fused_qkv = fused_qkv
        self.fused_kvcache = fused_kvcache
        self.fused_ffn_glu = fused_ffn_glu

        tp_size = 1 if self.tp_proc_group is None else self.tp_proc_group.size()
        num_kv_heads = params.num_heads if params.num_kv_heads is None else params.num_kv_heads
        self.hidden_dim = params.hidden_dim
        self.num_local_heads = params.num_heads // tp_size
        self.num_local_kv_heads = num_kv_heads // tp_size
        self.head_dim = params.hidden_dim // params.num_heads
        self.local_q_dim = self.num_local_heads * self.head_dim
        self.local_kv_dim = self.num_local_kv_heads * self.head_dim
        self.local_imm_dim = params.intermediate_dim // tp_size

        if self.dist_mapping.is_first_pp_rank():
            self.tok_embeddings = ParallelEmbedding(self.tp_proc_group, params.vocab_size, params.hidden_dim)
        self.layers = torch.nn.ModuleList()
        
        # for layer_id in self.layer_range:
        for layer_id in range(self.num_layers):
            self.layers.append(TransformerBlockPP(
                layer_id, params,
                friendly_gqa,
                fused_qkv,
                fused_kvcache,
                fused_ffn_glu,
                attn_wqkv_bias_term,
                attn_wo_bias_term,
                ffn_linear_bias_term,
                rotary_dim,
                self.tp_proc_group))

        if self.dist_mapping.is_last_pp_rank():
            self.norm = RMSNorm(params.hidden_dim, eps=params.norm_eps)
            self.output = ColumnParallelLinear(self.tp_proc_group, params.hidden_dim, params.vocab_size, bias_term=False)

    @torch.inference_mode()
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor],
                seqstarts: torch.Tensor, kvstarts: torch.Tensor,
                cachestarts: torch.Tensor, decoding_batches: torch.Tensor,
                start_pos: torch.Tensor, max_seqlen: torch.Tensor,  max_kvlen: torch.Tensor,
                kv_cache: torch.Tensor, kv_scale: torch.Tensor = None):
        # if is first rank, x is tokens, else hidden states
        # print(f"rank: {self.dist_mapping.rank} forward")
        TensorDumper.dump(x, "model_input")
        if self.dist_mapping.is_first_pp_rank():
            h = self.tok_embeddings(x)
        else:
            h = x
        _kv_scale = kv_scale
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
        for i, layer in enumerate(self.layers):
        # for layer in self.layers:
            h = layer(h, attn_mask, seqstarts, kvstarts, cachestarts,
                            decoding_batches, start_pos, max_seqlen, max_kvlen,
                            kv_cache, _kv_scale)
        # print(f"rank: {self.dist_mapping.rank} after layer forward")

        if self.dist_mapping.is_last_pp_rank():
            h = self.norm(h)
            TensorDumper.dump(h, "last_rms_norm")
            gathered_h = torch.index_select(h, 0, seqstarts[1:] - 1)
            TensorDumper.dump(gathered_h, "gathered_h")
            output = self.output(gathered_h)  # only compute last logits
            TensorDumper.dump(output, "logits_before_cast")
            output = output.float()
            TensorDumper.dump(output, "logits")
            # print(f"rank: {self.dist_mapping.rank} after output")
        else:
            output = h
        return output
    
    @torch.no_grad()
    def load_state_dict(self, state_dict: Mapping[str, Any]):
        loaded_params = set()
        model_params = {key: value for key, value in self.named_parameters()}

        pp_state_dict = filter_pp_state_dict(state_dict, self.layer_range, self.dist_mapping)

        tp_rank, tp_size = self.dist_mapping.tp_rank, self.dist_mapping.tp_size

        for key, value in pp_state_dict.items():            
            try:
                if 'attention.wq.weight' in key:
                    value = value.reshape(
                        self.num_local_heads * tp_size, self.head_dim, self.hidden_dim).split(
                            [self.num_local_heads] * tp_size, dim=0)[tp_rank].reshape(-1, self.hidden_dim)
                if 'attention.wk.weight' in key or 'attention.wv.weight' in key:
                    value = value.reshape(
                        self.num_local_kv_heads * tp_size, self.head_dim, self.hidden_dim).split(
                            [self.num_local_kv_heads] * tp_size, dim=0)[tp_rank].reshape(-1, self.hidden_dim)
                if 'attention.wo.weight' in key:
                    value = value.split([self.hidden_dim // tp_size] * tp_size, dim=1)[tp_rank]
                if 'feed_forward.w1.weight' in key or 'feed_forward.w3.weight' in key:
                    value = value.split([self.local_imm_dim] * tp_size, dim=0)[tp_rank]
                if 'feed_forward.w2.weight' in key:
                    value = value.split([self.local_imm_dim]*tp_size, dim=1)[tp_rank]
                if 'tok_embeddings.weight' in key:
                    value = value.split([self.hidden_dim // tp_size] * tp_size, dim=1)[tp_rank]
                if 'output.weight' in key:
                    value = value.split([self.vocab_size // tp_size] * tp_size, dim=0)[tp_rank]
                # split ColParaelleLinear bias
                if 'attention.wq.bias' in key or 'attention.wk.bias' in key or 'attention.wv.bias' in key or \
                     'attention.w1.bias' in key or 'attention.w3.bias' in key or \
                     'tok_embeddings.bias' in key or 'output.bias' in key:
                    bias_dim = value.shape[0]
                    value = value.split([bias_dim // tp_size] * tp_size)[tp_rank]

                module_name, param_name = key.rsplit(".", 1)
                if key in model_params:
                    self.get_submodule(module_name)._parameters[param_name][:] = value
                    loaded_params.add(key)
                    print(f'Loaded: {key} -> {key}[{value.shape}]')

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
                if self.fused_ffn_glu:
                    if 'feed_forward.w1' in key:
                        loaded_params.add(key)
                        module_name = module_name.replace('w1', 'wu')
                        self.get_submodule(module_name)._parameters[param_name][
                            :self.local_imm_dim] = value
                        replaced_key = module_name + '.' + param_name
                        print(f'Loaded: {key} -> {replaced_key}[{value.shape}]')
                    if 'feed_forward.w3' in key:
                        loaded_params.add(key)
                        module_name = module_name.replace('w3', 'wu')
                        self.get_submodule(module_name)._parameters[param_name][
                            self.local_imm_dim:] = value
                        replaced_key = module_name + '.' + param_name
                        print(f'Loaded: {key} -> {replaced_key}[{value.shape}]')
            except AttributeError as e:
                raise Exception(f'Failed to inject model weight {key}, can not find corresponding layer.')
        
        for key in pp_state_dict:
            if key not in loaded_params:
                print(f'{key} is not loaded.')


def filter_pp_state_dict(state_dict: Mapping[str, Any], layer_range: list, dist_mapping: DistMapping):
    if dist_mapping.pp_size == 1:
        return state_dict

    pp_state_dict = {}

    for key, value in state_dict.items():                
        if "tok_embeddings" in key:
            if dist_mapping.is_first_pp_rank():
                pp_state_dict.update({key: value})
            continue
        if "norm.weight" == key or "norm.bias" == key or "output" in key:
            if dist_mapping.is_last_pp_rank():
                pp_state_dict.update({key: value})
            continue
        prefix, layer_idx, param = key.split(".", 2)
        if int(layer_idx) not in layer_range:
            continue
        
        remapped_layer_idx = str(int(layer_idx) - layer_range[0])
        pp_key = ".".join([prefix, remapped_layer_idx, param])

        pp_state_dict.update({pp_key: value})
    return pp_state_dict