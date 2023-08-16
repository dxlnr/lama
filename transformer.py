# GPT-X
import numpy as np
from tinygrad.tensor import Tensor
# from tinygrad.nn import Linear

DEBUG = True


def attention_layer(q, k, v, mask=None):
    """ "Scaled Dot-Product Attention."""
    # matmul
    x = q.matmul(k.transpose())
    # scale
    x = x / np.sqrt(k.shape[1])
    # mask
    if mask is not None:
        pass
    # softmax
    x = x.softmax()
    # matmul
    return x.matmul(v)


def transformer():
    """."""
    pass


def train(d_tr: Tensor, block_size: int = 8, batch_size: int = 8):
    """."""
    for batch in range(batch_size):
        for t in range(block_size): 
            pass


def main():
    """."""
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

    block_size = 8
    if DEBUG:
        # This is referred to as the time dimension
        print("CONTEXT BLOCKS")
        x = d_tr[:block_size].numpy()
        y = d_tr[1:block_size + 1].numpy()

        for i in range(block_size):
            print(f"{x[:i+1]} := {y[i]}")
            print(f"{decode(x[:i+1])} := {decode([y[i]])}")

    # x = attention_layer(q, k, v)
    # print(x)


if __name__ == "__main__":
    main()
