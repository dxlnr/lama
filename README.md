# An Intuitive Introduction to the Transformer Architecture

As the transformer architecture is the State-of-the-Art archtitecture that is actually leading almost all the 
grounding breaking results in Deep Learning lately, this article tries to clarify what a transformer is, how it helped improve 
common architectures and how it can be intuitively understood.

## Getting Started

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

## Implementation

```bash
python transformer.py
# Add -v for more output.
python transformer.py -v
```

## Further Investigations


## Additional resources

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [The Transformer Family](https://lilianweng.github.io/posts/2020-04-07-the-transformer-family/)
