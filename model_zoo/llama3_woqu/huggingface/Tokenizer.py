import sys
import os
from tokenizers import Tokenizer as TokenizerFast
from logging import getLogger
from typing import List

sys.path.append(os.path.dirname(os.path.realpath(__file__)) + "/../..")

from ModelUtils import __Tokenizer__

logger = getLogger()

class Tokenizer(__Tokenizer__):
    def __init__(self, model_path: str):
        # reload tokenizer
        assert os.path.isfile(model_path), model_path
        self.sp_model = TokenizerFast.from_file(model_path)

        self.bos_token = "<|begin_of_text|>"
        self.eos_token = "<|end_of_text|>"
        self.pad_token = "<|end_of_text|>"

        # BOS / EOS token IDs
        self.n_words: int = self.sp_model.get_vocab_size()
        self.bos_id: int = self.sp_model.token_to_id(self.bos_token)
        self.eos_id: int = self.sp_model.token_to_id(self.eos_token)
        self.pad_id: int = self.sp_model.token_to_id(self.pad_token)

    def encode(self, s: str, bos: bool, eos: bool) -> List[int]:
        assert type(s) is str
        t = self.sp_model.encode(s).ids
        if bos:
            t = [self.bos_id] + t
        if eos:
            t = t + [self.eos_id]
        return t

    def decode(self, t: List[int]) -> str:
        return self.sp_model.decode(t)

    def vocab_size(self):
        return self.n_words

    def get_bos_id(self):
        return self.bos_id

    def get_eos_id(self):
        return self.eos_id

    def get_pad_id(self):
        return self.pad_id
