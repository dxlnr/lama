# An Intuitive Introduction to the Transformer Architecture

As the transformer architecture is the State-of-the-Art archtitecture that is actually leading almost all the 
grounding breaking results in Deep Learning lately, this article tries to clarify what a transformer is, how it helped improve 
common architectures and how it can be intuitively understood.

## Outline 

> What I cannot create, I do not understand
> 
> <cite>Richard P. Feynman</cite>

I strongly believe that building something will actually force you to gain a profound understanding of it, otherwise you will fail.
This is probably true in almost all cases as the citation of Feynman indicates. Please prove me wrong by contradiction. Therefore, we rely on that assumption for the transformer as well.

The main goal of this repo is to [build a transformer from scratch](https://www.lesswrong.com/posts/98jCNefEaBBb7jwu6/building-a-transformer-from-scratch-ai-safety-up-skilling). As there are many levels of abstractions involved, a soft rule set defining from scratch has to be specified. In a first implementation, we will rely on [tinygrad](https://github.com/tinygrad/tinygrad/tree/master). In a second one maybe [numpy](https://numpy.org/) or pure C can be used, which will lead you to eventually implement a [grad](http://blog.ezyang.com/2019/05/pytorch-internals/) library in order to train the thing. 

I really like the goals defined in the article by [M. Hobbhahn](https://www.lesswrong.com/posts/98jCNefEaBBb7jwu6/building-a-transformer-from-scratch-ai-safety-up-skilling). So they are used here as well:

**Goals**
- Build the attention mechanism
- Build a single-head attention mechanism
- Build a multi-head attention mechanism
- Build an attention block 
- Build one or multiple of a text classification transformer, BERT or GPT. The quality of the final model doesn’t have to be great, just clearly better than random.
- Train the model on a small dataset. 
- Test that the model actually learned something

**Bonus goals**
- Visualize one attention head
- Visualize how multiple attention heads attend to the words of an arbitrary sentence
- Reproduce the grokking phenomenon (see e.g. [Neel’s and Tom’s piece](https://www.lesswrong.com/posts/N6WM6hs7RQMKDhYjB/a-mechanistic-interpretability-analysis-of-grokking)). 
- Answer some of the questions in [Jacob Hilton's post](https://github.com/jacobhilton/deep_learning_curriculum/blob/master/1-Transformers.md).

## Running 

```bash
python transformer.py -v
```

Run the **GPT-2** model by OpenAI for some text inferrence.
```bash
python gpt.py -p "How can entropy be reversed?"
```

## Transformer Overview 

(1) **Input Encoding**

The first step is to encode the input to the transformer into some hidden space. In case of NLP this is done using [BPE](https://en.wikipedia.org/wiki/Byte_pair_encoding). An interesting question could be if this hidden space could be learned as well rather than 'handcoded manually', similar to the [MuZero](https://www.deepmind.com/blog/muzero-mastering-go-chess-shogi-and-atari-without-rules) approach.

(2) **Positional Encodings**

In a second step these input encodings get summed with a tensor encoding the positions of the each word in its context (sentence). As the transformer architecture ditched the recurrence mechanism used in [RNNs]() in favor of multi-head self-attention to speed up training time massively (making use of the massive parallism of GPUs rather than the sequential manner of RNNs). The Transformers needs an alternative way to capture this information as well. Check [this](https://kazemnejad.com/blog/transformer_architecture_positional_encoding/) for further information on positional encodings.

(3) **Attention Mechanism**

The attention mechanism mimics the retrieval of a **value** $v_i$ for a **query** $q$ based on a *key* $k_i$ in some data(base).
    $$ attention(q,k,v) = \sum_i similarity(q, k_i) \times v_i $$
```
                  Data 
                (k1, v1)
                (k2, v2)
                (k2, v3)
    Query --->     .     ---> 
                   .
                   .
                (kn, vn)
```
For the similarity a distribution over all the keys for a certain query is computed and you sample from this distribution for an output.
```
        v1      v2      v3      v4      ..      vn
        |       |       |       |               |
        *   +   *   +   *   +   *        +      *   ---> attention value
        |       |       |       |               |
        a1      a2      a3      a4      ..      an
                       ...                          } softmax : a_i = softmax(s_i)
        s1      s2      s3      s4      ..      sn
      / ^     / ^     / ^     / ^             / ^
   q /__|____/__|____/__|____/__|        ____/  |   
        |       |       |       |               |
        k1      k2      k3      k4      ..      kn
```

For calculating the *similarity* $s_i$ various functions are possible.
$$
s_i = f(q, k_i) = \begin{cases}
    q^T k_i & \text{dot product} \\
    q^T k_i / \sqrt(d) & \text{scaled dot product} \\
    q^T Wk_i & \text{general dot product} \\
    w^T_q q + w^T_k k_i & \text{additive similarity}
\end{cases}
$$

(4) **Multi-Head Attention**

Starting with the input vector which contains of all the words, the Multi-Head Attention computes the attention between every position and very other position sort in an extra dimension (Nx) to produce an even better (higher-dimensional) embedding. The idea behind this is to open and widen the space of which words and and pairs of words refer to each other.

## Additional resources

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [The Transformer Family](https://lilianweng.github.io/posts/2020-04-07-the-transformer-family/)
- [OpenAI](https://openai.com/research/better-language-models): Better Language Models and their Implications
