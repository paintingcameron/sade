import sys
from typing import List, Union, Optional
from itertools import product
from random import shuffle, choice

import torch
from torch import Tensor
from torch.nn import Embedding

from diffusers.models import ModelMixin
from diffusers.configuration_utils import ConfigMixin, register_to_config

class CharacterTokenizer(ConfigMixin):
    config_name = 'CharacterTokenizer'
    added_tokens_encoder = []

    @register_to_config
    def __init__(self, alphabet: str, max_length: int, default_char: str = ' '):
        
        self.alphabet = list(set(alphabet))
        self.alphabet.sort()
        self.max_length = max_length
        self.default_char = default_char

        self.char2token = {c:i for i, c in enumerate(self.alphabet)}
        self.token2char = {i:c for i, c in enumerate(self.alphabet)}

    def random_word_gen(self, deterministic: Optional[bool] = False):
        if deterministic:
            words = ["".join(i) for i in product(self.alphabet, repeat=self.max_length)]
            shuffle(words)

            i = -1
            while True:
                i += 1
                yield words[i % len(words)]
        else:
            while True:
                yield "".join(choice(self.alphabet) for _ in range(self.max_length))

    def _get_token(self, c: str) -> int:
        if c in self.char2token:
            return self.char2token[c]
        else:
            return len(self.char2token)

    def _get_char(self, t: int) -> str:
        if t in self.token2char:
            return self.token2char[t]
        else:
            return self.default_char


    def detokenize(self, tokens: Tensor) -> str:
        if len(tokens.shape) == 2:
            sequences = []
            for token_sequence in tokens:
                print(f"token sequence: {token_sequence}")
                sequences.append(self.detokenize(token_sequence))
        else:
            return "".join([self._get_char(int(t)) for t in tokens])

    def tokenize(self, x: Union[str, List[str]]) -> Tensor:
        return self(x)

    def __call__(self, x: Union[str, List[str]], max_length: Optional[int] = None) -> Tensor:
        if isinstance(x, str):
            return torch.tensor([self._get_token(c) for c in x])
        else:
            if max_length is None:
                max_length = self.max_length

            padded_words = []
            for y in x:
                if len(y) > max_length:
                    raise Exception(f"Label longer than {max_length}: \"{y}\"")
                
                padded_words.append(y + self.default_char * max(0, max_length - len(y)))

            return torch.tensor([[self._get_token(c) for c in word] for word in padded_words])


class TextEncoder(ModelMixin, ConfigMixin):
    config_name = "TextEncoder"

    @register_to_config
    def __init__(self, alphabet: str, embed_dim: int):
        super().__init__()

        self.alphabet = list(set(alphabet))
        self.alphabet.sort()
        self.embed_dim = embed_dim
        self.embedding = Embedding(len(self.alphabet)+1, embedding_dim=embed_dim)

    def forward(self, y: Tensor) -> Tensor:
        y_emb = self.embedding(y)
        y_emb = y_emb.reshape((y_emb.shape[0], -1))

        return y_emb
