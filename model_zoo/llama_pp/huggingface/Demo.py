import sys
import os
import json

from pathlib import Path
from typing import List

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../..")

import llama_pp.modeling.Loader as Loader
from llama.huggingface.Tokenizer import Tokenizer
from ModelParams import ModelParams

import torch.multiprocessing as mp
import argparse

import signal
import traceback
import os
import time

def ParseCommandLineArgs():
    parser = argparse.ArgumentParser()
    # basic command
    parser.add_argument(
        "--nnodes",
        type=int,
        default=1,
        help="Number of nodes, or the range of nodes in form <minimum_nodes>:<maximum_nodes>.",
    )
    parser.add_argument(
        "--nproc_per_node",
        type=int,
        default=1,
        help="Number of workers per node;"
    )

    parser.add_argument(
        "--master_port",
        type=str,
        default="29500"
    )

    parser.add_argument(
        "--local_addr",
        type=str,
        default="localhost"
    )

    # llm param
    parser.add_argument("--ckpt_dir", type=str)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--batch", type=int, default=4)

    parser.add_argument("--seqlen_scale_up", type=int, default=1)
    parser.add_argument("--unaligned_batch", type=int, default=False)
    parser.add_argument("--max_gen_len", type=int, default=256)
    parser.add_argument("--friendly_gqa", type=int, default=False)
    parser.add_argument("--fused_qkv", type=int, default=False)

    parser.add_argument("--fused_kvcache", type=int, default=True)
    parser.add_argument("--fused_ffn_glu", type=int, default=True)
    parser.add_argument("--auto_causal", type=int, default=True)
    parser.add_argument("--quantized_cache", type=int, default=True)
    parser.add_argument("--cache_layout", type=int, default=0)

    parser.add_argument("--cache_mode", type=int, default=0)
    parser.add_argument("--dynamic_batching", type=int, default=True)
    parser.add_argument("--pp_size", type=int, default=1)
    # parser.add_argument("--tp_size", type=int, default=1)
    parser.add_argument("--dump_tensor_path", type=str, default=None)
    parser.add_argument("--dump_steps", type=str, default=None)

    args = parser.parse_args()
    if args.dump_steps:
        args.dump_steps = [int(s) for s in args.dump_steps.split(",")]

    args.unaligned_batch = bool(args.unaligned_batch)
    args.friendly_gqa = bool(args.friendly_gqa)
    args.fused_qkv = bool(args.fused_qkv)
    args.fused_kvcache = bool(args.fused_kvcache)
    args.fused_ffn_glu = bool(args.fused_ffn_glu)
    args.auto_causal = bool(args.auto_causal)
    args.quantized_cache = bool(args.quantized_cache)
    args.dynamic_batching = bool(args.dynamic_batching)

    args.world_size = args.nproc_per_node * args.nnodes
    return args


def set_dist_env_var(rank: int, world_size: int, local_addr: str, master_port: str):
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["MASTER_ADDR"] = local_addr
    os.environ["MASTER_PORT"] = master_port

def main(rank: int, args: argparse.Namespace, queue: mp.Queue, global_start=None):
    set_dist_env_var(rank, args.world_size, args.local_addr, args.master_port)

    tokenizer = Tokenizer(model_path=args.tokenizer_path)

    with open(Path(args.ckpt_dir) / "pmx_params.json", "r") as f:
        params = json.loads(f.read())
    params: ModelParams = ModelParams(**params)

    generator = Loader.load(
        args.ckpt_dir, params, args.friendly_gqa,
        args.fused_qkv, args.fused_kvcache, args.fused_ffn_glu,
        args.auto_causal, args.quantized_cache, args.cache_layout,
        args.cache_mode, args.dynamic_batching,
        False, False, False, False,
        0, pp_size=args.pp_size,
        dump_tensor_path=args.dump_tensor_path, dump_steps=args.dump_steps
    )

    if args.unaligned_batch:
        test_prompt = [        # For these prompts, the expected answer is the natural continuation of the prompt
        "I believe the meaning of life is",
        "Simply put, the theory of relativity states that ",
        """A brief message congratulating the team on the launch:

        Hi everyone,

        I just """,
        # Few shot prompt (providing a few examples before asking model to complete more);
        """Translate English to French:

        sea otter => loutre de mer
        peppermint => menthe poivrée
        plush girafe => girafe peluche
        cheese =>""",
        ]
        test_prompt = [tokenizer.encode(t, bos=True, eos=False) for t in test_prompt]

        prompt_tokens = test_prompt.copy()
        for _ in range((args.batch - 1) // len(test_prompt)):
            prompt_tokens.extend(test_prompt)
    else:
        test_prompt = "I believe the meaning of life is"
        test_prompt = tokenizer.encode(test_prompt, bos=True, eos=False)

        _scale_up_prompt = []
        for _ in range(args.seqlen_scale_up):
            _scale_up_prompt.extend(test_prompt)
        test_prompt = _scale_up_prompt

        prompt_tokens = [test_prompt for _ in range(args.batch)]

    print(f"prepared {len(prompt_tokens)} prompts")
    results = generator.generate(
        prompt_tokens[:args.batch], tokenizer.get_eos_id(), tokenizer.get_pad_id(),
        max_gen_len=args.max_gen_len, temperature=args.temperature, top_p=args.top_p, top_k=0, 
        queue=queue, global_start=global_start
    )
    if generator.model.dist_mapping.is_last_pp_rank():
        for result in results:
            print(result)
            print(tokenizer.decode(result))
            print("\n==================================\n")


if __name__ == "__main__":
    args = ParseCommandLineArgs()
    print(args)
    mp.set_start_method('spawn')
    queue = mp.Queue()
    global_start = time.time()
    # tid_dict = mp.Manager().dict()
    # lock = mp.Lock()

    mp.spawn(main, nprocs=args.world_size, args=(args, queue, global_start), join=True)