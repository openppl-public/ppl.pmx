from typing import List
import sys
import os
import torch
import os
import json
import torch.multiprocessing as mp
import torch.distributed as dist

from .Model_pp import Transformer, TensorDumper

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../../..")

from ModelUtils import __Tokenizer__, __TextGenerator__
from ModelParallel import DistMapping

import time
from torch.cuda import synchronize

def sample_top_p(probs, p):
    probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = torch.gather(probs_idx, -1, next_token)
    return next_token

def recv_tensor_info(dist_mapping: DistMapping, hidden_dim: int, step: int):
    # print(f"step [{step}] start recv")
    tmp_tensor = torch.empty((2), dtype=torch.int64).cuda()
    dist.recv(tmp_tensor, src=dist_mapping.prev_pp_rank())
    _batches, _decoding_batches = tmp_tensor.chunk(2, dim=0)
    tmp_tensor2 = torch.empty((_batches.item() * 4), dtype=torch.int64).cuda()
    dist.recv(tmp_tensor2, src=dist_mapping.prev_pp_rank())
    _seqlens, _start_pos, _cache_starts, _tid_list = tmp_tensor2.chunk(4, dim=0)

    total_seqlen = torch.sum(_seqlens)
    _hs = torch.empty((total_seqlen, hidden_dim), dtype=torch.half).cuda()
    dist.recv(_hs, src=dist_mapping.prev_pp_rank())
    torch.cuda.synchronize()
    # print("recv end")
    return _decoding_batches, _seqlens, _start_pos, _cache_starts, _tid_list.tolist(), _hs

def send_tensor_info(dist_mapping: DistMapping, decoding_batches: int, _seqlens: torch.tensor, 
                     _start_pos: torch.tensor, _cache_starts: torch.tensor, _tid_list: torch.tensor, _hidden_states: torch.tensor, step: int):
    if dist_mapping.is_last_pp_rank():
        return
    # print(f"step [{step}] start send")
    batches = _seqlens.shape[0]
    tmp_tensor = torch.tensor([batches, decoding_batches], dtype=torch.int64).cuda()
    dist.send(tensor=tmp_tensor, dst=dist_mapping.next_pp_rank())
    tmp_tensor2 = torch.cat([_seqlens, _start_pos, _cache_starts, _tid_list], dim=0)
    dist.send(tensor=tmp_tensor2, dst=dist_mapping.next_pp_rank())
    dist.send(tensor=_hidden_states, dst=dist_mapping.next_pp_rank())
    # print("end send")

def post_process(queue: mp.Queue, logits: torch.tensor, temperature: float, top_p: float, start_pos, tid_list, current_steps, max_gen_len, output_ids: torch.tensor):
    if temperature > 0:
        probs = torch.softmax(logits / temperature, dim=-1)
        next_tokens = sample_top_p(probs, top_p)
    else:
        next_tokens = torch.argmax(logits, dim=-1)
    next_tokens = next_tokens.reshape(-1)
    output_ids[tid_list, start_pos] = next_tokens    # send 
    
    # push to queue and early finish check
    for idx, tid, next_token in zip(range(len(tid_list)), tid_list, next_tokens):
        # early finish check
        if current_steps[idx] >= max_gen_len:
            continue
        # print(f"queue.put(({tid}, [{next_token}])")
        queue.put((tid, [next_token]))
    return output_ids


def remove_finished_task(tid, tid_list, start_pos, seqlens, cachestarts, current_steps):
    idx = tid_list.index(tid)
    tid_list.pop(idx)
    seqlens.pop(idx)
    start_pos.pop(idx)
    cachestarts.pop(idx)
    current_steps.pop(idx)

def update_decode_input(seqlens: List[int], start_pos: List[int]):
    for idx in range(len(seqlens)):
        start_pos[idx] += seqlens[idx]    
        seqlens[idx] = 1

class Profiler():
    def __init__(self):
        self.recv_start = []
        self.recv_end = []
        self.forward_start = []
        self.forward_end = []
        self.send_start = []
        self.send_end = []

class LLaMA(__TextGenerator__):
    def __init__(self, model: Transformer):
        self.model = model
        self.profiler = Profiler()

    def generate(
        self,
        prompts_ids: List[List[int]],
        eos_id: int,
        pad_id: int,
        max_gen_len: int,
        temperature: float,
        top_k: int,
        top_p: float,
        queue: mp.Queue,
        global_start = None
    ) -> List[List[int]]:    

        global_step = 0

        total_prompt_len = 0
        for i, p in enumerate(prompts_ids):
            total_prompt_len = total_prompt_len + len(p)
             
        total_cache_len = total_prompt_len + len(prompts_ids) * max_gen_len
        head_dim = self.model.params.hidden_dim // self.model.params.num_heads
        num_local_kv_heads = self.model.params.num_kv_heads // self.model.dist_mapping.tp_size
        hidden_dim = self.model.params.hidden_dim
        local_num_layer = self.model.params.num_layers

        dist_mapping = self.model.dist_mapping

        if self.model.params.cache_layout == 0:
            cache_prefix_shape = (total_cache_len, local_num_layer, 2, num_local_kv_heads)
        elif self.model.params.cache_layout == 1:
            cache_prefix_shape = (local_num_layer, total_cache_len, 2, num_local_kv_heads)
        elif self.model.params.cache_layout == 2:
            cache_prefix_shape = (local_num_layer, 2, total_cache_len, num_local_kv_heads)
        elif self.model.params.cache_layout == 3:
            cache_prefix_shape = (local_num_layer, 2, num_local_kv_heads, total_cache_len)
        else:
            raise Exception("unsupported cache_layout: {}".format(self.model.params.cache_layout))

        if self.model.params.cache_quant_bit == 8:
            scale_head_dim = head_dim // self.model.params.cache_quant_group
            kv_cache = torch.zeros(cache_prefix_shape + (head_dim,), dtype=torch.int8).cuda()
            kv_scale = torch.zeros(cache_prefix_shape + (scale_head_dim,), dtype=torch.float16).cuda()
        else:
            kv_cache = torch.zeros(cache_prefix_shape + (head_dim,), dtype=torch.float16).cuda()
            kv_scale = None

        max_prompt_len = max([len(t) for t in prompts_ids])
        output_ids = torch.full((len(prompts_ids), max_prompt_len + max_gen_len), pad_id).cuda().long()
        for k, t in enumerate(prompts_ids):
            output_ids[k, : len(t)] = torch.tensor(t).long()

        pp_rank = self.model.dist_mapping.pp_rank

        def myprint(string: str):
            print(f"step[{global_step}]-rank[{pp_rank}]: {string}")
            

        # send in queue
        tid = 0
        if pp_rank == 0:
            # token_ids = []
            for prompt_id in prompts_ids:
                queue.put((tid, prompt_id))   # 0 means new req, 1 means decoding req
                # print(f" [x] Sent {tid}: {prompt_id}")
                tid += 1

        TensorDumper.step = 0

        if pp_rank == 0:
            allocated_cache_len = 0
            seqlens = []
            start_pos = []
            cachestarts = []
        
        tid_list = []

        max_seqlen = 0
        current_steps = []
        total_tid_cnt = len(prompts_ids)

        while True:
            # rank 0，从msgq里取数据
            # print(f"---------- step: {global_step}-------pp_rank: {pp_rank} ---------------- ")
            if dist_mapping.is_first_pp_rank():
                tokens_ids = []
                current_batches = 0
                decoding_batches = torch.tensor([0])
                prev_tid_set = set(tid_list)
                tid_set = set()

                self.profiler.recv_start.append(time.time() - global_start)
                while True:
                    item = queue.get()  # 期望是block wait
                    if item is None:
                        print("item is none")
                        break
                    tid, prompt_id = item
                    # print(f"queue.get(({tid}, [{prompt_id}])")
                    current_batches += 1
                    tid_set.add(tid)

                    if tid not in prev_tid_set: # new req
                        tokens_ids.extend(prompt_id)
                        l = len(prompt_id)
                        start_pos.append(0)
                        cachestarts.append(allocated_cache_len)
                        seqlens.append(l)
                        tid_list.append(tid)
                        allocated_cache_len += l + max_gen_len
                        current_steps.append(0)
                    else:   # decode
                        tokens_ids.extend(prompt_id)
                        decoding_batches += 1
                    
                    if queue.qsize() == 0:
                        break
                self.profiler.recv_end.append(time.time() - global_start)

                # early 检测与处理处理
                tid_finished = []

                for tid in prev_tid_set:
                    if tid not in tid_set:
                        tid_finished.append(tid)
                        remove_finished_task(tid, tid_list, start_pos, seqlens, cachestarts, current_steps)

                if len(tid_finished) > 0:
                    print("early finish tid list: ", tid_finished)

                _tokens_ids = torch.tensor(tokens_ids, dtype=torch.int64).cuda()
                if self.model.params.cache_mode == 0:
                    _cachestarts = torch.tensor(cachestarts, dtype=torch.int64).cuda()
                else:
                    raise Exception("unsupported cache_mode: {}".format(self.model.params.cache_mode))

                _start_pos = torch.tensor(start_pos, dtype=torch.int64).cuda()
                _seqlens = torch.tensor(seqlens, dtype=torch.int64).cuda()
                model_input0 = _tokens_ids
            else:   # other pp_rank
                prev_tid_list = tid_list
                prev_tid_set = set(prev_tid_list)

                self.profiler.recv_start.append(time.time() - global_start)
                
                decoding_batches, _seqlens, _start_pos, _cachestarts, tid_list, _hs = recv_tensor_info(dist_mapping, hidden_dim, global_step)
                
                self.profiler.recv_end.append(time.time() - global_start)
                
                tid_set = set(tid_list)
                
                all_tid_set = prev_tid_set | tid_set
                new_tid_set = all_tid_set - prev_tid_set
                finished_tid_set = all_tid_set - tid_set
                
                # update current_steps
                for tid in finished_tid_set:    # delete finish tid
                    idx = prev_tid_list.index(tid)
                    current_steps.pop(idx)
                current_steps.extend([0 for _ in range(len(new_tid_set))])   # add new tid
                
                current_batches = _seqlens.shape[0]
                model_input0 = _hs
            
            _kvlens = _start_pos + _seqlens
            max_seqlen = _seqlens.max().cpu().unsqueeze(0)
            max_kvlen = _kvlens.max().cpu().unsqueeze(0)
            _seqstarts = torch.zeros(current_batches + 1, dtype=torch.int64).cuda()
            _seqstarts[1:] = _seqlens
            _seqstarts = _seqstarts.cumsum(0)
            _kvstarts = torch.zeros(current_batches + 1, dtype=torch.int64).cuda()
            _kvstarts[1:] = _kvlens
            _kvstarts = _kvstarts.cumsum(0)

            attn_mask = torch.empty(0, dtype=torch.float16).cuda()
            if self.model.params.auto_causal == False and decoding_batches < current_batches:
                padded_last_dim = (_kvstarts[-1] + 15) // 16 * 16
                attn_mask = torch.zeros((_seqstarts[-1], padded_last_dim), dtype=torch.float16).cuda()
                for b in range(decoding_batches, current_batches):
                    seqbeg = _seqstarts[b]
                    seqend = _seqstarts[b+1]
                    kvbeg = _kvstarts[b]
                    kvend = _kvstarts[b+1]
                    attn_mask[seqbeg:seqend, kvbeg:kvend] = \
                        torch.triu(torch.full_like(attn_mask[seqbeg:seqend, kvbeg:kvend], float("-inf")), diagonal=1)

            self.profiler.forward_start.append(time.time() - global_start)
            synchronize()
            model_output = self.model.forward(model_input0, attn_mask, _seqstarts, _kvstarts,
                                        _cachestarts, decoding_batches,
                                        _start_pos, max_seqlen, max_kvlen,
                                        kv_cache, kv_scale)
            synchronize()
            self.profiler.forward_end.append(time.time() - global_start)


            current_steps = [i + 1 for i in current_steps]
            global_step += 1
            # update input 参数
            if pp_rank == 0:
                update_decode_input(seqlens, start_pos)

            self.profiler.send_start.append(time.time() - global_start)
            if dist_mapping.is_last_pp_rank():
                next_token_idx = (_start_pos + _seqlens).tolist()
                post_process(queue, model_output, temperature, top_p, next_token_idx, tid_list, current_steps, max_gen_len, output_ids)
            else:
                send_tensor_info(dist_mapping, decoding_batches, _seqlens, _start_pos, _cachestarts, torch.tensor(tid_list).cuda(), model_output, global_step)

            self.profiler.send_end.append(time.time() - global_start)

            if max(current_steps) >= max_gen_len:
                # myprint("break")
                break

        # print(f"rank [{pp_rank}]:       recv start | recv end | forward start | forward end | send start | send end ")
        # for i in range(max_gen_len):
        #     print(f"rank[{pp_rank}] step[{i}]: {round(self.profiler.recv_start[i] * 1000)} | {round(self.profiler.recv_end[i] * 1000)} | {round(self.profiler.forward_start[i] * 1000)} | {round(self.profiler.forward_end[i] * 1000)} | {round(self.profiler.send_start[i] * 1000)} | {round(self.profiler.send_end[i] * 1000)} ")

        return output_ids.tolist()
    

    # python export method
    def export(
        self,
        export_path: str
    ):
        bsz = 4
        total_len = 16

        total_cache_len = bsz * total_len
        head_dim = self.model.params.hidden_dim // self.model.params.num_heads
        num_local_kv_heads = self.model.params.num_kv_heads // self.model.dist_mapping.tp_size
        num_layers = self.model.params.num_layers

        if self.model.params.cache_layout == 0:
            cache_prefix_shape = (total_cache_len, num_layers, 2, num_local_kv_heads)
            max_tokenlen_idx = 0
        elif self.model.params.cache_layout == 1:
            cache_prefix_shape = (num_layers, total_cache_len, 2, num_local_kv_heads)
            max_tokenlen_idx = 1
        elif self.model.params.cache_layout == 2:
            cache_prefix_shape = (num_layers, 2, total_cache_len, num_local_kv_heads)
            max_tokenlen_idx = 2
        elif self.model.params.cache_layout == 3:
            cache_prefix_shape = (num_layers, 2, num_local_kv_heads, total_cache_len)
            max_tokenlen_idx = 3
        else:
            raise Exception("unsupported cache_layout: {}".format(self.model.params.cache_layout))

        if self.model.params.cache_quant_bit == 8:
            scale_head_dim = head_dim // self.model.params.cache_quant_group
            kv_cache = torch.zeros(cache_prefix_shape + (head_dim,), dtype=torch.int8)
            kv_scale = torch.zeros(cache_prefix_shape + (scale_head_dim,), dtype=torch.float16)
        else:
            kv_cache = torch.zeros(cache_prefix_shape + (head_dim,), dtype=torch.float16)
            kv_scale = None

        seqlen = total_len // 2
        token_ids = torch.ones(bsz * seqlen, dtype=torch.int64)
        hidden_states = torch.ones(bsz * seqlen, self.model.params.hidden_dim)
        start_pos = torch.zeros(bsz, dtype=torch.int64)
        seqstarts = torch.arange(0, seqlen * (bsz + 1), seqlen, dtype=torch.int64)
        kvstarts = torch.arange(0, seqlen * (bsz + 1), seqlen, dtype=torch.int64)
        decoding_batches = torch.tensor([0], dtype=torch.int64)
        max_seqlen = torch.tensor([seqlen])
        attn_mask = torch.empty(0, dtype=torch.float16)

        if self.model.params.cache_mode == 0:
            cachestarts = torch.arange(0, total_len * bsz, total_len, dtype=torch.int64)
            cachestarts_dim_name = 'batch'
        elif self.model.params.cache_mode == 1:
            cachestarts = torch.arange(0, total_len * bsz, dtype=torch.int64)
            cachestarts_dim_name = 'total_kvlen'
        else:
            raise Exception("unsupported cache_mode: {}".format(self.model.params.cache_mode))
        
        # set input
        if self.model.dist_mapping.is_first_pp_rank():
            input_names = ["token_ids", "attn_mask", "seqstarts", "kvstarts",
            "cachestarts", "decoding_batches",
            "start_pos", "max_seqlen", "max_kvlen",
            "kv_cache", "kv_scale"]
            input_tensors = [token_ids, attn_mask, 
                seqstarts, kvstarts,
                cachestarts, decoding_batches,
                start_pos, max_seqlen, max_seqlen,
                kv_cache, kv_scale
            ]
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
                'cachestarts': {
                    0:cachestarts_dim_name
                },
                'start_pos': {
                    0:'batch'
                },
                'kv_cache': {
                    max_tokenlen_idx: 'max_tokenlen'
                },
                'kv_scale': {
                    max_tokenlen_idx: 'max_tokenlen'
                },
            }
        else:
            input_names = ["hidden_states", "attn_mask", "seqstarts", "kvstarts",
            "cachestarts", "decoding_batches",
            "start_pos", "max_seqlen", "max_kvlen",
            "kv_cache", "kv_scale"]
            input_tensors = [hidden_states, attn_mask, 
                seqstarts, kvstarts,
                cachestarts, decoding_batches,
                start_pos, max_seqlen, max_seqlen,
                kv_cache, kv_scale
            ]

            dynamic_axes = {
                'hidden_states': {
                    0:'total_seqlen'
                },
                'seqstarts': {
                    0:'batch + 1'
                },
                'kvstarts': {
                    0:'batch + 1'
                },
                'cachestarts': {
                    0:cachestarts_dim_name
                },
                'start_pos': {
                    0:'batch'
                },
                'kv_cache': {
                    max_tokenlen_idx: 'max_tokenlen'
                },
                'kv_scale': {
                    max_tokenlen_idx: 'max_tokenlen'
                },
            }

        # set output
        if self.model.dist_mapping.is_last_pp_rank():
            output_names = ["logits"]
            dynamic_axes.update({
                "logits": {
                    0: 'batch',
                    1: 'vocab_size'}
            })

        else:
            output_names = ["hidden_states"]
            dynamic_axes.update({
                'hidden_states': {
                    0: 'total_seqlen',
                    1: 'hidden_dim'}
            })

        if self.model.params.cache_quant_bit == 0:
            dynamic_axes.pop('kv_scale')
            input_names.pop()
            input_tensors.pop()

        local_rank = self.model.dist_mapping.rank
        model_path = os.path.join(export_path, "model_slice_{}".format(local_rank))
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        torch.onnx.export(
            self.model.cpu(),
            tuple(input_tensors),
            os.path.join(model_path, "model.onnx"),
            input_names=input_names,
            output_names=output_names,
            do_constant_folding=True,
            opset_version=11,
            dynamic_axes=dynamic_axes)
        
        if local_rank == 0:
            with open(os.path.join(export_path, "params.json"), "w") as f:
                json.dump(self.model.params.__dict__, f)