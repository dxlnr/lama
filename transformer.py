# GPT-X
import argparse
import sys
from dataclasses import dataclass
from typing import Optional

import numpy as np
from tinygrad.nn import Embedding, LayerNorm, Linear
from tinygrad.tensor import Tensor
from tinygrad.state import torch_load
from utils import fetch_as_file

import tiktoken

DEBUG = False


def create_arg_parser():
    """Get arguments from command lines."""
    parser = argparse.ArgumentParser(description="GPT-X")
    parser.add_argument(
        "-v",
        "--verbose",
        help="Be verbose",
        action="store_true",
        dest="loglevel",
    )
    return parser


def get_url(model_size: str = 'gpt2'):
    """Get model url from huggingface."""
    return f'https://huggingface.co/{model_size}/resolve/main/pytorch_model.bin'


def datasets(s: str = "data/tinyshakespeare/input.txt") -> tuple[list]:
    """Prepare and return dataset splits."""
    with open(s, "r") as f:
        d = f.read()
    # lut
    chars = {c: i for i, c in enumerate(sorted(set(d)))}
    # encode
    encode = lambda x: list(map(lambda c: chars[c], x))
    # train/val/test split of dataset
    d_tr = encode(d[: int(len(d) * 0.8)])
    d_v = encode(d[int(len(d) * 0.8) : int(len(d) * 0.9)])
    d_t = encode(d[int(len(d) * 0.9) :])

    return d_tr, d_v, d_t


def get_train_batch(d_tr: Tensor, block_size: int = 8, batch_size: int = 4):
    """Extract a single batch of training data."""
    idx = np.random.randint(0, len(d_tr) - block_size, (batch_size,))
    x = Tensor(np.stack([d_tr[i : i + block_size] for i in idx]))
    y = Tensor(np.stack([d_tr[i + 1 : i + block_size + 1] for i in idx]))
    return x, y


@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50304  # From the GPT-2 Paper
    layers: int = 12
    heads: int = 12
    channels: int = 512
    dropout: float = 0.0
    bias: bool = False


class TransformerBlock:
    def __init__(
        self,
        conf: GPTConfig,
        act=lambda x: x.gelu(),
    ):
        """."""
        self.channels = conf.channels
        self.heads = conf.heads
        self.head_size = int(conf.channels / conf.heads)
        self.dropout = conf.dropout
        self.act = act

        self.q = (
            Tensor.uniform(self.channels, self.channels),
            Tensor.zeros(self.channels),
        )
        self.k = (
            Tensor.uniform(self.channels, self.channels),
            Tensor.zeros(self.channels),
        )
        self.v = (
            Tensor.uniform(self.channels, self.channels),
            Tensor.zeros(self.channels),
        )

        self.la = Linear(self.channels, self.channels)
        self.l1 = Linear(self.channels, self.channels)
        self.l2 = Linear(self.channels, self.channels)

        self.mlp1 = Linear(self.channels, 4 * self.channels)
        self.mlp2 = Linear(4 * self.channels, self.channels)

    def __call__(self, x):
        """Forward pass of a single transformer block.

        :param x: input sequence. (B, T, C) where B: batch size, T: time and C: channels or embedded dim.
        """
        x = x + self.multi_head_attention(
            x,
            # mask=Tensor(
            #     np.ones((x.shape[1], x.shape[1])).astype(dtype=np.float32)
            # ).tril(),
        ).dropout(self.dropout)
        x = self.l1(x.layernorm())

        x = x + self.mlp2(self.act(self.mlp1(x))).dropout(self.dropout)
        x = self.l2(x.layernorm())
        return x

    def attention_layer(self, q, k, v, mask=None, dropout_p=0.1):
        """ "Scaled Dot-Product Attention."""
        x = q @ k.transpose(-2, -1)
        # scale
        x = x / np.sqrt(k.shape[-1])
        # mask
        if mask is not None:
            mask = (mask == 0).where(-float("inf"), mask)
            x = x + mask.transpose(-2, -1)
        # softmax
        x = x.softmax(-1)
        # dropout
        x = x.dropout(dropout_p)
        return x @ v

    def multi_head_attention(self, x, mask: Optional[Tensor] = None):
        """Multi-Head Attention."""
        # split heads: (B, T, C) -> (B, T, nh, C/H) -> (B, nh, T, hs)
        # (batch_size, sequence_length, num_heads, d_model/num_heads)
        q, k, v = [
            x.linear(*y)
            .reshape(shape=(x.shape[0], -1, self.heads, self.head_size))
            .transpose(1, 2)
            for y in [self.q, self.k, self.v]
        ]
        x = self.attention_layer(q, k, v, mask, self.dropout)
        x = self.la(x.reshape(shape=(x.shape[0], -1, self.heads * self.head_size)))
        # returns (B, T, C)
        return x


class Transformer:
    def __init__(self, conf: GPTConfig):
        # token embeddings
        self.te = Embedding(conf.vocab_size, conf.channels)
        # positional embeddings
        self.pe = Embedding(conf.block_size, conf.channels)
        # Transformer Blocks
        self.tbs = [
            TransformerBlock(conf) for _ in range(conf.layers)
        ]
        # Layer Norm
        self.ln_f = LayerNorm(conf.channels, eps=1e-5)
        # final linear layer
        self.lf = Linear(conf.channels, conf.vocab_size, bias=False)

    def __call__(self, x: Tensor, start_idx: int = 0):
        """Forward pass of the transformer."""
        _, seqlen = x.shape
        pos = Tensor(np.arange(start_idx, start_idx + seqlen)).reshape(
            shape=(1, -1)
        )
        x = self.te(x) + self.pe(pos)

        # mask = Tensor(np.ones((x.shape[1], x.shape[1])).astype(dtype=np.float32)).tril() if seqlen > 1 else None
        x = x.sequential(self.tbs)
        return self.lf(self.ln_f(x))


def train(d_tr: Tensor, block_size: int = 8, batch_size: int = 8):
    """."""
    for batch in range(batch_size):
        for t in range(block_size):
            pass


def main():
    """."""
    # Parse command line arguements.
    arg_parser = create_arg_parser()
    args = arg_parser.parse_args(sys.argv[1:])
    try:
        DEBUG = args.loglevel
    except ValueError:
        pass

    # dataset
    with open("data/tinyshakespeare/input.txt", "r") as f:
        d = f.read()
    # luts
    chars = {c: i for i, c in enumerate(sorted(set(d)))}
    tokens = {i: c for i, c in enumerate(sorted(set(d)))}
    # encode/decode
    encode = lambda x: list(map(lambda c: chars[c], x))
    decode = lambda x: "".join(map(lambda c: tokens[c], x))
    # There are alternatives to encode/decode tokens: e.g. tiktoken (OpenAI) or sentencepiece (Google)
    tokenizer = tiktoken.get_encoding("gpt2")

    if DEBUG:
        print("RAW DATASET")
        print(f"tokens: \n{tokens}")
        print(f"\nExample: encode sequence 'clara': {encode('clara')}.")
        print("")

    # train/val/test split of dataset
    d_tr = Tensor(encode(d[: int(len(d) * 0.8)]))
    if DEBUG:
        block_size = 8
        # This is referred to as the time dimension
        print("CONTEXT BLOCKS")
        x = d_tr[:block_size].numpy()
        y = d_tr[1 : block_size + 1].numpy()

        for i in range(block_size):
            print(f"{x[:i+1]} := {y[i]}")
            print(f"{decode(x[:i+1])} := {decode([y[i]])}")

    # Mathematical intuition behind the attention mechanism
    if DEBUG:
        print("\nATTENTION MECHANISM (IDEA). AVERAGING.")
        b, t, c = 4, 8, 2
        x = Tensor(np.random.rand(b, t, c).astype(dtype=np.float32))

        bow = np.zeros((b, t, c)).astype(dtype=np.float32)
        for b in range(x.shape[0]):
            for t in range(x.shape[1]):
                prev = x[b, : t + 1, :]
                bow[b, t, :] = np.mean(prev.numpy(), axis=0)

        print(
            "Random data sample: \n",
            x.numpy()[0],
            "\n\nAverage up to the previous tokens: \n",
            bow[0],
        )

    model_type = "gpt2"
    # n_layer, n_head and n_embd are determined from model_type
    conf_args = {
        'gpt2':         dict(layers=12, heads=12, channels=768),   # 124M params
        'gpt2-medium':  dict(layers=24, heads=16, channels=1024),  # 350M params
        'gpt2-large':   dict(layers=36, heads=20, channels=1280),  # 774M params
        'gpt2-xl':      dict(layers=48, heads=25, channels=1600),  # 1558M params
    }[model_type]
    conf_args['vocab_size'] = 50257  # always 50257 for GPT model checkpoints
    conf_args['block_size'] = 1024   # always 1024 for GPT model checkpoints
    conf_args['bias'] = True         # always True for GPT model checkpoints

    conf = GPTConfig(**conf_args)

    transformer = Transformer(conf)

    weights = torch_load(fetch_as_file(get_url(model_type)))
    # train
    train_d, _, _ = datasets()
    x, y = get_train_batch(train_d)
    if DEBUG:
        print("\nTRAINING")
        print(f"(data) single batch:  x: {x.shape}, y: {y.shape}")

    # forward pass
    # x = Embedding(conf.block_size, 128)(x)
    out = transformer(x)
    if DEBUG:
        print("(TransformerBlock) out", out.shape)


if __name__ == "__main__":
    main()
