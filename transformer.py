# GPT-X
import argparse
import sys
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear, Embedding

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


class TransformerBlock:
    def __init__(
        self,
        channels: int = 512,
        heads: int = 8,
        d_ff: int = 2048,
        act=lambda x: x.relu(),
        dropout_p: float = 0.1,
    ):
        """."""
        self.channels = channels
        self.heads = heads
        self.dropout = dropout_p
        self.act = act

        self.q = (Tensor.uniform(channels, channels), Tensor.zeros(channels))
        self.k = (Tensor.uniform(channels, channels), Tensor.zeros(channels))
        self.v = (Tensor.uniform(channels, channels), Tensor.zeros(channels))

        self.la = Linear(channels, channels)
        self.l1 = Linear(channels, d_ff)
        self.l2 = Linear(d_ff, channels)

    def __call__(self, x):
        """Forward pass of the transformer.

        :param x: input sequence. (B, T, C) where B: batch size, T: time and C: channels or embedded dim.
        """
        x = x + self.multi_head_attention(
            x,
            # mask=Tensor(
            #     np.ones((x.shape[1], x.shape[1])).astype(dtype=np.float32)
            # ).tril(),
        ).dropout(self.dropout)
        x = x.layernorm()

        x = x + self.act(self.l2(self.act(self.l1(x))))
        x = x.layernorm()
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

    def multi_head_attention(
        self, x, heads: int = 8, head_size: int = 8, mask=None, dropout_p: float = 0.1
    ):
        """Multi-Head Attention."""
        # split heads: (B, T, C) -> (B, T, H, C/H)
        # (batch_size, sequence_length, num_heads, d_model/num_heads)
        q, k, v = [
            x.linear(*y)
            .reshape(shape=(x.shape[0], -1, heads, head_size))
            .transpose(1, 2)
            for y in [self.q, self.k, self.v]
        ]
        x = self.attention_layer(q, k, v, mask, self.dropout)
        x = self.la(x.reshape(shape=(x.shape[0], -1, x.shape[-1] * x.shape[-2])))
        # returns (B, T, C)
        return x


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

    # Transformer Model Instatiation
    transformer = TransformerBlock()
    # train
    train_d, _, _ = datasets()
    x, y = get_train_batch(train_d)
    if DEBUG:
        print("\nTRAINING")
        print(f"(data) single batch:  x: {x.shape}, y: {y.shape}")

    # forward pass
    x = Embedding(16, 512)(x)
    out = transformer(x)
    if DEBUG:
        print("(TransformerBlock) out", out.shape)


if __name__ == "__main__":
    main()
