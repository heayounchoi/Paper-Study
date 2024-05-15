### [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/pdf/2010.11929)

**Abstract**
- In vision, attention is either applied in conjunction with convolutional netoworks, or used to replace certain components of convolutional networks while keeping their overall structure in place.
- This paper shows that this reliance on CNNs is not necessary and a pure transformer applied directly to sequences of image patches can perform very well on image classification tasks.
- Vision Transformer (ViT) attains excellent results compared to state-of-the-art convolutional networks while requiring substantially fewer computational resources to train.
---

**Introduction**
- Thanks to Transformers' computational efficiency and scalability, it has become possible to train models of unprecedented size, with over 100B parameters.
- In computer vision, models that replaced the convolutions entirely have not yet been scaled effectively on modern hardware accelerators due to the use of specialized attention patterns.
- Therefore, in large-scale image recognition, classic ResNet-like architectures are still state of the art.
- In this paper, we split an image into patches and provide the sequence of linear embeddings of these patches as an input to a Transformer.
- Image patches are treated the same way as tokens (words) in an NLP application.
- When trained on mid-sized datasets such as ImageNet without strong regularization, these models yield modest accuracies of a few percentage points below ResNets of comparable size.
- Transformers lack some of the inductive biases inherent to CNNs, such as translation equivariance and locality, and therefore do not generalize well when trained on insufficient amounts of data.
- However, the picture changes if the models are trained on larger datasets (14M-300M images).
- large scale training trumps inductive bias
- ViT attains excellent results when pre-trained at sufficient scale and transferred to tasks with fewer datapoints.
---

**Related Work**
- Naive application of self-attention to images would require that each pixel attends to every other pixel.
- to apply Transforemrs in the context of image processing, several approximations have been tried in the past
1) self-attention applied only in local neighborhoods for each query pixel instead of globally. Such local multi-head dot-product self attention blocks can completely replace convolutions.
2) Sparse Transformers employ scalable approximations to global self-attention in order to be applicable to images.
3) apply attention in blocks of varying sizes, in the extreme case only along individual axes
- these require complex engineering to be implemented efficiently on hardware accelerators
- iGPT applies Transformers to image pixels after reducing image resolution and color space
---

**Method**

<img src="https://velog.velcdn.com/images/heayounchoi/post/2862f738-5734-49d2-a815-d37823215f32/image.png">

**_Vision Transformer (ViT)_**
- The Transformer uses constant latent vector size D through all of its layers, so we flatten the patches and map to D dimensions with a trainable linear projection. We refer to the output of this projection as the patch embeddings.
- Similar to BERT's \[class] token, we prepend a learnable embedding to the sequnce of embedded patches, whose state at the output of the Transformer encoder serves as the image representation y.
- Both during pre-training and fine-tuning, a classification head is attached to the output of the Transformer encoder.
- The classification head is implemented by a MLP with one hidden layer at pre-training time and by a single linear layer at fine-tuning time.
- Position embeddings are added to the patch embeddings to retain positional information.
- The MLP contains two layers with a GELU non-linearity.

_Inductive bias_
- Vision Transformer has much less image-specific inductive bias than CNNs.
- In CNNs, locality, two-dimensional neighborhood structure, and translation equivariance are baked into each layer throughout the whole model.
- In ViT, only MLP layer are local and translationally equivariant, while the self-attention layers are global.
- The two-dimensional neighborhood structure is used very sparingly: in the beginning of the model by cutting the image into patches and at fine-tuning time for adjusting the position embeddings for images of different resolution.
- Other than that, the position embeddings at initialization time carry no information about the 2D positions of the patches and all spatial relations between the patches have to be learned from scratch.

_Hybrid Architecture_
- As an alternative to raw image patches, the input sequence can be formed from feature maps of a CNN.
- In this hybrid model, the patch embedding projection E is applied to patches extracted from a CNN feature map.

**_Fine-tuning and Higher Resolution_**
- remove the pre-trained prediction head and attach a zero-initialized feedforward layer
- It is often beneficial to fine-tune at higher resolution than pre-training.
- since images of higher resolution has larger effective sequence length, we perform 2D interpolation of the pre-trained position embeddings, according to their location in the original image
- this resolution adjustment and patch extraction are the only points at which an inductive bias about the 2D structure of the images is manually injected into the Vision Transformer
---

**Experiements**
- ResNet vs ViT vs hybrid
- When considering the computational cost of pre-training the model, ViT performs very favourably, attaining state of the art on most recognition benchmarks at a lower pre-training cost.

**_Setup_**

_Datasets_

_Model Variants_

<img src="https://velog.velcdn.com/images/heayounchoi/post/caced419-90f9-4f7c-a5a0-71d0b30f086d/image.png">

- Transformer's sequence length is inversely proportional to the square of the patch size, thus models with smaller patch size are computationally more expensive

_Training & Fine-tuning_

_Metrics_
- few-shot or fine-tuning accuracy

**_Comparison to State of the Art_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/63b50ee7-5757-4bdb-9997-592d23e2cc4a/image.png">

<img src="https://velog.velcdn.com/images/heayounchoi/post/a27a702b-31f5-4fee-b8fe-161879b17c12/image.png">

**_Pre-training Data Requirements_**
- the convolutional inductive bias is useful for smaller datasets, but for larger ones, learning the relevant patterns directly from data is sufficient, even beneficial

**_Scaling Study_**
- ViT dominate ResNets on the performance/compute trade-off
- hybrids slightly outperform ViT at small computational budgets, but the difference vanishes for larger models
- ViT appear not to saturate within the range tried, motivating future scaling efforts

**_Inspecting Vision Transformer_**

**_Self-Supervision_**

---

**Conclusion**

---

**Appendix**
- scaling the depth of transformer results in the biggest improvements which are clearly visible up until 64 layers
- scaling the width of the network results in the smallest changes
- decreasing the patch size and thus increasing the effective sequence length shows robust improvements without introducing parameters
- these finding suggest that compute might be a better predictor of performance than the number of parameters, and that scaling should emphasize depth over width if any
- overall, scaling all dimensions proportionally results in robust improvements

<img src="https://velog.velcdn.com/images/heayounchoi/post/d42a51cb-be2c-4a72-b175-730fad7e9ab0/image.png">

- since ViT encoder operates on patch-level inputs, as opposed to pixel-level, the differences in how to encode spatial information is less important


