### [Attention Is All You Need](https://arxiv.org/pdf/1706.03762)

**Abstract**
- The dominant sequence transduction models are based on complex recurrent or convolutional neural networks that include an encoder and a decoder.
- The best performining models also connect the encoder and decoder through an attention mechanism.
- Transformer is based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.
- These models are superior in quality while being more parallelizable and requiring significantly less time to train.
---

**Introduction**
- Recurrent neural networks, long short-term memory and gated recurrent neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problmes such as language modeling and machine translation.
- There's a fundamental constraint of sequential computation in recurrent models. (hidden states)
- Attention mechanisms allow modeling of dependencies without regard to their distance in the input or output sequences, but such attention mechanisms are used in conjunction with a recurrent network.
- Transformer relies entirely on an attention mechanism to draw global dependencies between input and output.
- more parallelization -> relatively short training time
---

**Background**
- In Extended Neural GPU, ByteNet, and ConvS2S, the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions, and this makes it difficult to learn dependencies between distant positions.
- Transformer reduces this to a constant number of operations.
- Multi-Head Attention solves the problem of reduced effective resolution due to averaging attention-weighted positions.
- Self-attention (intra-attention) is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence.
- End-to-end memory networks are based on a recurrent attention mechanism instead of sequence-aligned recurrence.
- Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution.
---

**Model Architecture**
- Transformer uses stacked self-attention and point-wise, fully connected layers for both the encoder and decoder.

<img src="https://velog.velcdn.com/images/heayounchoi/post/90fe5c3d-7c0d-4daa-80c7-0eb583a5518b/image.png">

**_Encoder and Decoder Stacks_**

_Encoder:_
- The encoder is composed of a stack of N=6 identical layers.
- Each layer has two sub-layers.
- The first is a multi-head self-attention mechanism, and the second is a simple, position-wise fully connected feed-forward network.
- a residual connection around each of the two sub-layers, followed by layer normalization
- all sub-layers in the model, as well as the embedding layers, produce outputs of dimension 512

_Decoder:_
- + third sub-layer, which performs multi-head attention over the output of the encoder stack
- + modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions

**_Attention_**
- An attention function can be described as mapping a query and a set of key-value pairs to an output, where the query, keys, values, and output are all vectors.
- The output is computed as a weighted sum of the values, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

_Scaled Dot-Product Attention_

<img src="https://velog.velcdn.com/images/heayounchoi/post/6b45d029-311a-4f00-9455-6a8223213eeb/image.png">

<img src="https://velog.velcdn.com/images/heayounchoi/post/506d96b6-460b-48b3-b6fe-e506e73158d7/image.png">

- The two most commonly used attention functions are additive attention, and dot-product (multiplicative) attention.
- Dot-product attention is identical to scaled dot-product attention, except for the scaling factor.
- Additive attention computes the compatibility function using a feed-forward network with a single hidden layer.
- While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.
- While the small values of $$d_k$$ the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of $$d_k$$.
- for large values of $$d_k$$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremly small gradients.
- To counteract this effect, we scale the dot products.

_Multi-Head Attention_

<img src="https://velog.velcdn.com/images/heayounchoi/post/d6c2da39-4325-4b03-861c-16c58a42a45c/image.png">

- Instead of performing a single attention function with $$d_{model}$$-dimensional keys, values and queries, it is beneficial to linearly project the queries, keys and values h times with different, learned linear projections to $$d_q, d_k$$ and $$d_v$$ dimensions, respectively.
- On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding $$d_v$$-dimensional output values.
- These are concatenated and once again projected, resulting in the final values.
- Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions.
- In this work we employ h=8 parallel attention layers, or heads.
- For each of these we use $$d_k = d_v = d_{model}/h = 64$$.

_Applications of Attention in our Model_
- Transformer uses multi-head attention in three different ways:
- "encoder-decoder attention" layers. queries come from the previous decoder layer and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence.
- encoder self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder.
- decoder self-attention layers. allow each position in the decoder to attend to all positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attetnion by masking out all values in the input of the softmax which correspond to illegal connections.

**_Position-wise Feed-Forward Networks_**
- fully connected feed-forward network is applied to each position separately and identically

<img src="https://velog.velcdn.com/images/heayounchoi/post/6ae9e573-76e2-4338-9c6d-d6869935be1c/image.png">

- different parameters from layer to layer
- The dimensionality of input and output is $$d_{model} = 512$$, and the inner-layer has dimensionality $$d_{ff} = 2048$$.

**_Embeddings and Softmax_**
- share the same weight matrix between the two embedding layers and the pre-softmax linear transformation
- in the embedding layers, multiply those weights by $$\sqrt{d_{model}}$$

**_Positional Encoding_**
- Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence.
- To this end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks.
- In this work, we use sine and cosine functions of different frequencies:

<img src="https://velog.velcdn.com/images/heayounchoi/post/3a65109d-d941-4c3d-a812-c224f273dfd2/image.png">

where pos is the position and i is the dimension.
- This function was chosen because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset k, $$PE_{pos + k}$$ can be represented as a linear function of $$PE_{pos}$$.
---

**Why Self-Attention**
1. total computational complexity per layer
2. amount of computation that can be parallelized, as measured by the minimum number of sequential operations required
3. path length between long-range dependencies in the network

<img src="https://velog.velcdn.com/images/heayounchoi/post/4a70a9b6-6394-4fde-8903-0cbc56c3e9bc/image.png">

- In terms of computational complexity, self-attention layers are faster than recurrent layers when the sequence length n is smaller than the representation dimensionality d, which is most of the case with sentence representations used by state-of-the-art models in machine translations, such as word-piece and byte-pair representations.
- To improve computational performance for tasks involving very long sequences, self-attention could be restricted to considering only a neighborhood of size r in the input sequence centered around the respective output position.
- the complexity of a separable convolution is equal to the combination of a self-attention layer and a point-wise feed-forward layer, the approach we take in our model
- Not only do individual attention heads clearly learn to perform different tasks, many appear to exhibit behavior related to the syntactic and semantic structure of the sentences.
---

**Training**

**_Training Data and Batching_**

**_Hardward and Schedule_**

**_Optimizer_**
- Adam optimizer
- increase the learning rate linearly for the first warmup_steps training steps, and decrease it thereafter proportionally to the inverse square root of the step number

**_Regularization_**

_Residual Dropout_
- We apply dropout to the output of each sub-layer, before it is added to the sub-layer input and normalized.
- In addition, we apply dropout to the sums of the embeddings and the positional encodings in both the encoder and decoder stacks.

_Label Smoothing_
- hurts perplexity, as the model learns to be more unsure, but improves accuracy and BLEU score
---

**Result**

**_Machine Translation_**

**_Model Variations_**
- single-head attention is 0.9 BLEU worse than the best setting, quality also drops off with too many heads
- reducing the attention key size $$d_k$$ hurts model quality. This suggests that determining compatibility is not easy and that a more sophisticated compatibility function than dot product may be beneficial.
- bigger models are better
- dropout is very helpful in avoiding over-fitting
- replacing sinusoidal positional encoding with learned positional embeddings shows nearly identical results to the base model

**_English Constituency Parsing_**
- 문장의 구문 구조를 분석하는 기법으로, 영어 문장을 구성하는 개별 구성 요소(단어나 구)와 그 관계를 파악하여 문장의 구조를 나무(Tree) 형태로 표현
---
