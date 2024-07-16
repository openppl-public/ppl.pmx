from typing import List
import sys
import os
import torch
import json

from .Model import Transformer, TensorDumper

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../..")

from ModelUtils import __Tokenizer__, __TextGenerator__


class BatchState:
    def __init__(self):
        self.tid = 0
        self.input_tokens = []
        self.start_pos = 0
        self.cache_starts = 0
        self.output_tokens = []
        self.is_decoding = False


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


        def round_up_to_page(seqlen):
            page_size = self.model.params.page_size
            return (seqlen + page_size - 1) // page_size * page_size


        def gen_page_list(start, seqlen, max_len):
            page_size = self.model.params.page_size
            start_page = round_up_to_page(start) // page_size
            max_page_count = round_up_to_page(max_len) // page_size
            eff_page_count = round_up_to_page(seqlen) // page_size
            page_list = [i * page_size for i in range(start_page, start_page + eff_page_count)]
            if eff_page_count < max_page_count:
                page_list = page_list + [-1 for i in range(eff_page_count, max_page_count)]
            return page_list


        unprocessed_prompt_tokens_ids = []
        total_cache_len = 0
        for i, p in enumerate(prompts_ids):
            unprocessed_prompt_tokens_ids.append(p.copy())
            if self.model.params.cache_mode == 0:
                total_cache_len += len(p) + max_gen_len
            if self.model.params.cache_mode == 1:
                total_cache_len += round_up_to_page(len(p) + max_gen_len)

        head_dim = self.model.params.hidden_dim // self.model.params.num_heads
        num_local_kv_heads = self.model.params.num_kv_heads // torch.distributed.get_world_size(group=self.model.proc_group)
        num_layers = self.model.params.num_layers

        if self.model.params.cache_layout == 0:
            cache_prefix_shape = (total_cache_len, num_layers, 2, num_local_kv_heads)
        elif self.model.params.cache_layout == 1:
            cache_prefix_shape = (num_layers, total_cache_len, 2, num_local_kv_heads)
        elif self.model.params.cache_layout == 2:
            cache_prefix_shape = (num_layers, 2, total_cache_len, num_local_kv_heads)
        elif self.model.params.cache_layout == 3:
            cache_prefix_shape = (num_layers, 2, num_local_kv_heads, total_cache_len)
        else:
            raise Exception("unsupported cache_layout: {}".format(self.model.params.cache_layout))

        if self.model.params.cache_quant_bit == 8:
            scale_head_dim = head_dim // self.model.params.cache_quant_group
            kv_cache = torch.zeros(cache_prefix_shape + (head_dim,), dtype=torch.int8).cuda()
            kv_scale = torch.zeros(cache_prefix_shape + (scale_head_dim,), dtype=torch.float16).cuda()
        else:
            kv_cache = torch.zeros(cache_prefix_shape + (head_dim,), dtype=torch.float16).cuda()
            kv_scale = torch.empty(0)

        max_prompt_len = max([len(t) for t in unprocessed_prompt_tokens_ids])
        max_cache_len = round_up_to_page(max_prompt_len + max_gen_len)

        batch_states = []
        allocated_cache_len = 0
        processed_batches = 0
        TensorDumper.step = 0
        finished_tokens = [[] for _ in unprocessed_prompt_tokens_ids]
        while True:
            # if len(unprocessed_prompt_tokens_ids) > 0:
            while len(unprocessed_prompt_tokens_ids) > 0:
                state = BatchState()
                state.tid = processed_batches

                state.input_tokens = unprocessed_prompt_tokens_ids.pop(0)
                if self.context_chunking:
                    state.input_tokens.reverse()
                input_len = len(state.input_tokens)

                if self.model.params.cache_mode == 0:
                    state.cache_starts = allocated_cache_len
                    allocated_cache_len += input_len + max_gen_len
                if self.model.params.cache_mode == 1:
                    # paged attetion. We must align cache len to page size to avoid overlap
                    cache_len = round_up_to_page(input_len + max_gen_len)
                    state.cache_starts = gen_page_list(allocated_cache_len, cache_len, max_cache_len)
                    allocated_cache_len += cache_len

                processed_batches += 1
                batch_states.append(state)

            current_batches = len(batch_states)
            decoding_batches = sum([1 if s.is_decoding else 0 for s in batch_states])
            seqstarts = torch.zeros(current_batches + 1, dtype=torch.int64)
            kvstarts = torch.zeros(current_batches + 1, dtype=torch.int64)

            seqlens = []
            token_ids = []
            # context chunking only take 4 token at once
            if self.context_chunking:
                for b, s in enumerate(batch_states):
                    seqlens.append(len(s.input_tokens[-4:]) if not s.is_decoding else 1)
                    token_ids.extend(s.input_tokens[-4:][::-1] if not s.is_decoding else [s.output_tokens[-1]])
            else:
                for b, s in enumerate(batch_states):
                    seqlens.append(len(s.input_tokens) if not s.is_decoding else 1)
                    token_ids.extend(s.input_tokens if not s.is_decoding else [s.output_tokens[-1]])

            kvlens = [s.start_pos + l for (s, l) in zip(batch_states, seqlens)]
            seqstarts[1:] = torch.tensor(seqlens, dtype=torch.int64)
            kvstarts[1:] = torch.tensor(kvlens, dtype=torch.int64)
            seqstarts = seqstarts.cumsum(0)
            kvstarts = kvstarts.cumsum(0)
            cachestarts = torch.tensor([s.cache_starts for s in batch_states], dtype=torch.int64).cuda()
            start_pos = torch.tensor([s.start_pos for s in batch_states], dtype=torch.int64).cuda()
            token_ids = torch.tensor(token_ids, dtype=torch.int64).cuda()

            # generate attention mask [sum(seqlens), pad(sum(kvlens), 16)]
            attn_mask = torch.empty(0, dtype=torch.float16)
            if self.model.params.auto_causal == False and decoding_batches < current_batches:
                attn_mask = torch.zeros((seqstarts[-1], (kvstarts[-1] + 15) // 16 * 16), dtype=torch.float16).cuda()
                for b in range(decoding_batches, current_batches):
                    seqbeg = seqstarts[b]
                    seqend = seqstarts[b+1]
                    kvbeg = kvstarts[b]
                    kvend = kvstarts[b+1]

                    attn_mask[seqbeg:seqend, kvbeg:kvend] = (
                        torch.triu(
                            torch.full_like(attn_mask[seqbeg:seqend, kvbeg:kvend], float("-inf")),
                            diagonal=1
                        )
                    )

            seqstarts = seqstarts.cuda()
            kvstarts = kvstarts.cuda()

            max_seqlen = torch.tensor([max(seqlens)], dtype=torch.int64)
            max_kvlen = torch.tensor([max(kvlens)], dtype=torch.int64)
            decoding_batches = torch.tensor([decoding_batches], dtype=torch.int64)

            logits = self.model.forward(token_ids, attn_mask, seqstarts, kvstarts,
                                        cachestarts, decoding_batches, start_pos,
                                        max_seqlen, max_kvlen, kv_cache, kv_scale)
            TensorDumper.step += 1

            if temperature > 0:
                probs = torch.softmax(logits / temperature, dim=-1)
                next_tokens = sample_top_p(probs, top_p)
            else:
                next_tokens = torch.argmax(logits, dim=-1)

            next_tokens = next_tokens.reshape(-1)
            next_tokens = next_tokens.tolist()

            removed_batch = []
            for b, s in enumerate(batch_states):
                s.start_pos += seqlens[b]
                if self.context_chunking:
                    s.input_tokens = s.input_tokens[:-4]
                    s.is_decoding = len(s.input_tokens) == 0
                else:
                    s.input_tokens = []
                    s.is_decoding = True
                
                if s.is_decoding:
                    s.output_tokens.append(next_tokens[b])

                if len(s.output_tokens) >= max_gen_len or next_tokens[b] == eos_id:
                    removed_batch.append(b)

            removed_batch.reverse()
            for b in removed_batch:
                s = batch_states[b]
                finished_tokens[s.tid] = s.output_tokens
                batch_states.pop(b)
            if len(batch_states) == 0:
                break

        response_ids = []
        for i, t in enumerate(finished_tokens):
            # cut to eos tok if any
            try:
                t = t[: t.index(eos_id)]
            except ValueError:
                pass
            response_ids.append(prompts_ids[i] + t)
        return response_ids


    def export(
        self,
        export_path: str,
    ):
        bsz = 4
        total_len = 16
        page_size = self.model.params.page_size

        total_cache_len = bsz * total_len
        head_dim = self.model.params.hidden_dim // self.model.params.num_heads
        num_local_kv_heads = self.model.params.num_kv_heads // torch.distributed.get_world_size(group=self.model.proc_group)
        num_layers = self.model.params.num_layers

        if self.model.params.cache_layout == 0:
            cache_prefix_shape = (total_cache_len, num_layers, 2, num_local_kv_heads)
            max_tokens_idx = 0
        elif self.model.params.cache_layout == 1:
            cache_prefix_shape = (num_layers, total_cache_len, 2, num_local_kv_heads)
            max_tokens_idx = 1
        elif self.model.params.cache_layout == 2:
            cache_prefix_shape = (num_layers, 2, total_cache_len, num_local_kv_heads)
            max_tokens_idx = 2
        elif self.model.params.cache_layout == 3:
            cache_prefix_shape = (num_layers, 2, num_local_kv_heads, total_cache_len)
            max_tokens_idx = 3
        else:
            raise Exception("unsupported cache_layout: {}".format(self.model.params.cache_layout))

        if self.model.params.cache_quant_bit == 8:
            scale_head_dim = head_dim // self.model.params.cache_quant_group
            kv_cache = torch.zeros(cache_prefix_shape + (head_dim,), dtype=torch.int8)
            kv_scale = torch.zeros(cache_prefix_shape + (scale_head_dim,), dtype=torch.float16)
        else:
            kv_cache = torch.zeros(cache_prefix_shape + (head_dim,), dtype=torch.float16)
            kv_scale = torch.empty(0)

        seqlen = total_len // 2
        token_ids = torch.ones(bsz * seqlen, dtype=torch.int64)
        start_pos = torch.zeros(bsz, dtype=torch.int64)
        seqstarts = torch.arange(0, seqlen * (bsz + 1), seqlen, dtype=torch.int64)
        kvstarts = torch.arange(0, seqlen * (bsz + 1), seqlen, dtype=torch.int64)
        decoding_batches = torch.tensor([0], dtype=torch.int64)
        max_seqlen = torch.tensor([seqlen])
        attn_mask = torch.empty(0, dtype=torch.float16)

        if self.model.params.cache_mode == 0:
            cachestarts = torch.arange(0, total_len * bsz, total_len, dtype=torch.int64)
            cachestarts_axes = {
                0:'batch'
            }
        elif self.model.params.cache_mode == 1:
            cachestarts = torch.tensor([[b * page_size] for b in range(bsz)], dtype=torch.int64)
            cachestarts_axes = {
                0:'batch',
                1:'max_pages'
            }
        else:
            raise Exception("unsupported cache_mode: {}".format(self.model.params.cache_mode))

        input_names = ["token_ids", "attn_mask", "seqstarts", "kvstarts",
                       "cachestarts", "decoding_batches",
                       "start_pos", "max_seqlen", "max_kvlen",
                       "kv_cache", "kv_scale"]
        output_names = ["logits"]

        dynamic_axes = {
            'token_ids': {
                0:'total_seqlen'
            },
            'seqstarts': {
                0:'batch + 1'
            },
            'kvstarts': {
                0:'batch + 1'
            },
            'cachestarts': cachestarts_axes,
            'start_pos': {
                0:'batch'
            },
            'kv_cache': {
                max_tokens_idx: 'max_tokens'
            },
            'kv_scale': {
                max_tokens_idx: 'max_tokens'
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
            (token_ids, attn_mask, 
             seqstarts, kvstarts,
             cachestarts, decoding_batches,
             start_pos, max_seqlen, max_seqlen,
             kv_cache, kv_scale),
            os.path.join(model_path, "model.onnx"),
            input_names=input_names,
            output_names=output_names,
            do_constant_folding=True,
            opset_version=11,
            dynamic_axes=dynamic_axes)

        if local_rank == 0:
            with open(os.path.join(export_path, "params.json"), "w") as f:
                json.dump(self.model.params.__dict__, f)
