# GPT-X
import argparse
import sys
import numpy as np
from tinygrad.tensor import Tensor
from tinygrad.nn import Linear

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


def datasets(
    s: str = "data/tinyshakespeare/input.txt", block_size: int = 8, batch_size: int = 8
):
    """Prepare and return dataset splits."""
    with open(s, "r") as f:
        d = f.read()
    # lut
    chars = {c: i for i, c in enumerate(sorted(set(d)))}
    # encode
    encode = lambda x: list(map(lambda c: chars[c], x))
    # train/val/test split of dataset
    d_tr = Tensor(encode(d[: int(len(d) * 0.8)]))
    d_v = Tensor(encode(d[int(len(d) * 0.8) : int(len(d) * 0.9)]))
    d_t = Tensor(encode(d[int(len(d) * 0.9) :]))

    return d_tr, d_v, d_t


def get_train_batch(d_tr: Tensor, block_size: int = 8, batch_size: int = 8):
    """Extract a single batch of training data."""
    idx = np.random.randint(0, d_tr.shape[0] - block_size, (batch_size,))
    x = np.stack([d_tr[i : i + block_size] for i in idx])
    y = np.stack([d_tr[i + 1 : i + block_size + 1] for i in idx])
    return x, y


def attention_layer(q, k, v, mask=None):
    """ "Scaled Dot-Product Attention."""
    # matmul
    x = q.matmul(k)
    # scale
    x = x / np.sqrt(k.shape[1])
    # mask
    if mask is not None:
        pass
    # softmax
    x = x.softmax()
    # matmul
    return x.matmul(v)


class Transformer:
    def __init__(self, channels: int = 256, heads: int = 8, depth: int = 6):
        """."""
        self.q = Linear(channels, heads)
        self.k = Linear(channels, heads)

    def forward(self, x):
        """Forward pass of the transformer.

        :param x: input sequence. (B, T, C)
        """
        return attention_layer(self.q(x), self.k(x), x)


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
    d_v = Tensor(encode(d[int(len(d) * 0.8) : int(len(d) * 0.9)]))
    d_t = Tensor(encode(d[int(len(d) * 0.9) :]))

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

    transformer = Transformer()

    # train
    x, y = get_train_batch(d_tr)
    out = transformer.forward(x)

    print(out.shape)


if __name__ == "__main__":
    main()
