# GPT-2
import functools
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
import tiktoken
from tinygrad.nn import Embedding, Linear
from tinygrad.nn.state import load_state_dict, torch_load
from tinygrad.tensor import Tensor

from utils import create_arg_parser, fetch_as_file, get_url


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # From the GPT-2 Paper
    layers: int = 12
    heads: int = 12
    channels: int = 512
    dropout: float = 0.0
    bias: bool = False


class LayerNorm:
    def __init__(self, dim, eps=1e-5):
        self.eps = eps
        self.weight = Tensor.ones(dim)
        self.bias = Tensor.zeros(dim)

    def __call__(self, x: Tensor):
        return (x.layernorm(eps=self.eps)) * self.weight + self.bias


class MLP:
    def __init__(self, dim, hidden_dim):
        self.c_fc = Linear(dim, hidden_dim, bias=True)
        self.c_proj = Linear(hidden_dim, dim, bias=True)

    def __call__(self, x: Tensor) -> Tensor:
        return self.c_proj(self.c_fc(x).gelu())


class Attention:
    def __init__(self, channels, heads):
        self.c_attn = Linear(channels, 3 * channels, bias=True)
        self.c_proj = Linear(channels, channels, bias=True)
        self.heads = heads
        self.channels = channels
        self.head_size = channels // heads
        self.dropout = 0.0

    def __call__(
        self,
        x: Tensor,
        cache_k: Optional[Tensor],
        cache_v: Optional[Tensor],
        start_idx: int,
        mask: Optional[Tensor] = None,
    ) -> Tensor:
        qkv = self.c_attn(x)
        q, k, v = [
            qkv.slice([None, None, (i * self.channels, (i + 1) * self.channels)])
            for i in range(3)
        ]
        q, k, v = [
            x.reshape(x.shape[0], x.shape[1], self.heads, self.head_size)
            for x in (q, k, v)
        ]

        bsz, seqlen, _, _ = q.shape
        if start_idx != 0:
            assert cache_k, "no cache"
            assert (
                seqlen == k.shape[1] and seqlen == v.shape[1]
            ), "seqlen is wrong shape."
            k, v = cache_k.cat(k, dim=1), cache_v.cat(v, dim=1)

        # save the cache
        cache_k, cache_v = k.realize(), v.realize()
        q, keys, values = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        output = (
            q.scaled_dot_product_attention(keys, values, mask)
            .transpose(1, 2)
            .reshape(bsz, seqlen, -1)
        )
        return self.c_proj(output), cache_k, cache_v


class GPTBlock:
    def __init__(self, conf: GPTConfig):
        self.attn = Attention(conf.channels, conf.heads)
        self.mlp = MLP(conf.channels, 4 * conf.channels)
        self.ln_1 = LayerNorm(conf.channels, eps=1e-5)
        self.ln_2 = LayerNorm(conf.channels, eps=1e-5)
        self.cache_k, self.cache_v = None, None

    def inner(
        self,
        x: Tensor,
        cache_k: Optional[Tensor],
        cache_v: Optional[Tensor],
        start_pos: int,
        mask: Optional[Tensor],
    ):
        output, cache_k, cache_v = self.attn(
            self.ln_1(x), cache_k, cache_v, start_pos, mask
        )
        h = x + output
        return (h + self.mlp(self.ln_2(h))).realize(), cache_k, cache_v

    def __call__(self, x: Tensor, start_idx: int, mask: Optional[Tensor]):
        x, self.cache_k, self.cache_v = self.inner(
            x, self.cache_k, self.cache_v, start_idx, mask
        )
        return x


class GPT:
    def __init__(self, conf: GPTConfig):
        # token embeddings
        self.wte = Embedding(conf.vocab_size, conf.channels)
        # positional embeddings
        self.wpe = Embedding(conf.block_size, conf.channels)
        # Transformer Blocks
        self.h = [GPTBlock(conf) for _ in range(conf.layers)]
        # Layer Norm
        self.ln_f = LayerNorm(conf.channels, eps=1e-5)
        # final linear layer
        self.lm_head = Linear(conf.channels, conf.vocab_size, bias=False)

    def __call__(self, x: Tensor, start_idx: int = 0):
        """Forward pass of the transformer."""
        _, seqlen = x.shape
        pos = Tensor(np.arange(start_idx, start_idx + seqlen)).reshape(shape=(1, -1))
        h = self.wte(x) + self.wpe(pos)

        mask = (
            Tensor.full((1, 1, seqlen, start_idx + seqlen), float("-inf"))
            .triu(start_idx + 1)
            .realize()
            if seqlen > 1
            else None
        )
        h = h.sequential(
            [
                functools.partial(layer, start_idx=start_idx, mask=mask)
                for layer in self.h
            ]
        )
        return self.lm_head(self.ln_f(h))

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 100,
        temp: float = 0.8,
        top_k=None,
        tokenizer=None,
    ):
        """Generating sequence of words."""
        assert tokenizer is not None, "Please provide a tokenizer."
        toks = tokenizer.encode(prompt, allowed_special={"<|endoftext|>"})
        start_idx = 0
        for _ in range(max_new_tokens):
            x = Tensor([toks[start_idx:]])
            y = self(x, start_idx=start_idx)[:, -1, :].realize()

            probs = (y / temp).softmax()
            probs = probs.numpy().flatten()
            y = int(np.random.choice(len(probs), p=probs))

            start_idx = len(toks)
            toks.append(y)

            res = tokenizer.decode(toks)
        return res


if __name__ == "__main__":
    # Parse command line arguements.
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args(sys.argv[1:])

    if not (p := args.p):
        p = "Are you the problem?"

    model_type = "gpt2"
    # n_layer, n_head and n_embd are determined from model_type
    conf_args = {
        "gpt2": dict(layers=12, heads=12, channels=768),  # 124M params
        "gpt2-medium": dict(layers=24, heads=16, channels=1024),  # 350M params
        "gpt2-large": dict(layers=36, heads=20, channels=1280),  # 774M params
        "gpt2-xl": dict(layers=48, heads=25, channels=1600),  # 1558M params
    }[model_type]
    conf_args["vocab_size"] = 50257  # always 50257 for GPT model checkpoints
    conf_args["block_size"] = 1024  # always 1024 for GPT model checkpoints
    conf_args["bias"] = True  # always True for GPT model checkpoints

    conf = GPTConfig(**conf_args)
    gpt = GPT(conf)
    # load pretrained weights
    weights = torch_load(fetch_as_file(get_url(model_type)))

    transposed = [
        "attn.c_attn.weight",
        "attn.c_proj.weight",
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    ]
    for k in weights.keys():
        if any(k.endswith(w) for w in transposed):
            weights[k] = Tensor(weights[k].numpy().T)
    # lm head and wte are tied
    weights["lm_head.weight"] = Tensor(weights["wte.weight"].numpy())
    load_state_dict(gpt, weights)

    tokenizer = tiktoken.get_encoding("gpt2")
    # generate
    print("\nGPT2: \n")
    out = gpt.generate(p, 150, tokenizer=tokenizer)
    print(out, "\n")
