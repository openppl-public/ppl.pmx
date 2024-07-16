from typing import List
import sys
import os
import torch
import json

from .Model import Transformer, TensorDumper

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../..")

from ModelUtils import __TextGenerator__


class LLaMA(__TextGenerator__):
    def __init__(self, model: Transformer):
        self.model = model
        self.context_chunking = False

    def generate(
        self,
        prompts_ids: List[List[int]],
        eos_id: int,
        pad_id: int,
        max_gen_len: int,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> List[List[int]]:
        def sample_top_p(probs, p):
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            next_token = torch.multinomial(probs_sort, num_samples=1)
            next_token = torch.gather(probs_idx, -1, next_token)
            return next_token


        def padto16(x: torch.Tensor):
            last_dim = x.shape[-1]
            padded_dim = (last_dim + 15) // 16 * 16
            return torch.nn.functional.pad(x, (0, padded_dim - last_dim), "constant", 0)


        if self.context_chunking == True:
            raise Exception("static batching does not support context chunking")

        bsz = len(prompts_ids)

        min_prompt_size = min([len(t) for t in prompts_ids])
        max_prompt_size = max([len(t) for t in prompts_ids])

        total_len = max_gen_len + max_prompt_size

        head_dim = self.model.params.hidden_dim // self.model.params.num_heads
        num_local_kv_heads = self.model.params.num_kv_heads // torch.distributed.get_world_size(group=self.model.proc_group)
        num_layers = self.model.params.num_layers

        if self.model.params.cache_layout == 0:
            cache_shape_prefix = (bsz, num_layers, 2, total_len, num_local_kv_heads)
        elif self.model.params.cache_layout == 1:
            cache_shape_prefix = (num_layers, bsz, 2, num_local_kv_heads, total_len)
        else:
            raise Exception("unsupported cache_layout: {}".format(self.model.params.cache_layout))

        if self.model.params.cache_quant_bit == 8:
            scale_head_dim = head_dim // self.model.params.cache_quant_group
            kv_cache = torch.zeros(cache_shape_prefix + (head_dim,), dtype=torch.int8).cuda()
            kv_scale = torch.zeros(cache_shape_prefix + (scale_head_dim,), dtype=torch.float16).cuda()
        else:
            kv_cache = torch.zeros(cache_shape_prefix + (head_dim,), dtype=torch.float16).cuda()
            kv_scale = torch.empty(0)

        tokens_ids = torch.full((bsz, total_len), pad_id).cuda().long()
        for k, t in enumerate(prompts_ids):
            tokens_ids[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens_ids != pad_id
        prev_pos = 0
        for cur_pos in range(min_prompt_size, total_len):
            token_ids = tokens_ids[:, prev_pos:cur_pos]
            start_pos = torch.tensor([prev_pos])
            TensorDumper.step = cur_pos - min_prompt_size

            attn_mask = torch.empty(0, dtype=torch.float16)
            if self.model.params.auto_causal == False and token_ids.shape[1] > 1:
                attn_mask = torch.full((min_prompt_size, min_prompt_size), float("-inf")).cuda()
                attn_mask = torch.triu(attn_mask, diagonal=1).to(torch.float16)
                attn_mask = padto16(attn_mask)

            logits = self.model.forward(token_ids, attn_mask, start_pos, kv_cache, kv_scale)
            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits, dim=-1)
            next_token = next_token.reshape(-1)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens_ids[:, cur_pos], next_token
            )
            tokens_ids[:, cur_pos] = next_token
            prev_pos = cur_pos

        response_ids = []
        for i, t in enumerate(tokens_ids.tolist()):
            # cut to max gen len
            t = t[: len(prompts_ids[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(eos_id)]
            except ValueError:
                pass
            response_ids.append(t)
        return response_ids


    def export(
        self,
        export_path: str
    ):
        bsz = 4
        total_len = 16

        head_dim = self.model.params.hidden_dim // self.model.params.num_heads
        num_local_kv_heads = self.model.params.num_kv_heads // torch.distributed.get_world_size(group=self.model.proc_group)
        num_layers = self.model.params.num_layers

        if self.model.params.cache_layout == 0:
            cache_shape_prefix = (bsz, num_layers, 2, total_len, num_local_kv_heads)
            cache_batch_idx = 0
            cache_seqlen_idx = 3
        elif self.model.params.cache_layout == 1:
            cache_shape_prefix = (num_layers, bsz, 2, num_local_kv_heads, total_len)
            cache_batch_idx = 1
            cache_seqlen_idx = 4
        else:
            raise Exception("unsupported cache_layout: {}".format(self.model.params.cache_layout))

        if self.model.params.cache_quant_bit == 8:
            scale_head_dim = head_dim // self.model.params.cache_quant_group
            kv_cache = torch.zeros(cache_shape_prefix + (head_dim,), dtype=torch.int8)
            kv_scale = torch.zeros(cache_shape_prefix + (scale_head_dim,), dtype=torch.float16)
        else:
            kv_cache = torch.zeros(cache_shape_prefix + (head_dim,), dtype=torch.float16)
            kv_scale = torch.empty(0)

        start_pos = torch.tensor([0])
        tokens_ids = torch.ones(bsz, total_len // 2).long()
        attn_mask = torch.empty(0, dtype=torch.float16)

        input_names = ["token_ids", "attn_mask", "start_pos", "kv_cache", "kv_scale"]
        output_names = ["logits"]

        dynamic_axes = {
            'token_ids': {
                0:'batch',
                1:'seqlen'
            },
            'kv_cache': {
                cache_batch_idx: 'max_batch',
                cache_seqlen_idx: 'max_seqlen'
            },
            'kv_scale': {
                cache_batch_idx: 'max_batch',
                cache_seqlen_idx: 'max_seqlen'
            },
            'logits': {
                0: 'batch',
                1: 'vocab_size'
            },
        }

        if self.model.params.cache_quant_bit == 0:
            dynamic_axes.pop('kv_scale')
            input_names.pop()

        local_rank = torch.distributed.get_rank(group=self.model.proc_group)
        model_path = os.path.join(export_path, "model_slice_{}".format(local_rank))

        if not os.path.exists(model_path):
            os.makedirs(model_path)

        torch.onnx.export(
            self.model.cpu(),
            (tokens_ids, attn_mask,
             start_pos, kv_cache, kv_scale),
            os.path.join(model_path, "model.onnx"),
            input_names=input_names,
            output_names=output_names,
            do_constant_folding=True,
            opset_version=11,
            dynamic_axes=dynamic_axes)

        if local_rank == 0:
            with open(os.path.join(export_path, "params.json"), "w") as f:
                json.dump(self.model.params.__dict__, f)
