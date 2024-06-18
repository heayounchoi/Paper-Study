### [StyleCLIP: Text-Driven Manipulation of StyleGAN Imagery](https://arxiv.org/pdf/2103.17249)

**Abstract**
- Inspired by the ability of StyleGAN to generate highly realistic images in a variety of domains, much recent work has focused on understanding how to use the latent spaces of StyleGAN to manipulate generated and real images.
- However, discovering semantically meaningful latent manipulations typically involves painstaking human examination of the many degrees of freedom, or an annotated collection of images for each desired manipulation.
- In this work, we explore leveraging the power of CLIP models in order to develop a text-based interface for StyleGAN image manipulation that does not require such manual effort.
- We first introduce an optimization scheme that utilizes a CLIP-based loss to modify an input latent vector in response to a user-provided text prompt.
- Next, we describe a latent mapper that infers a text-guided latent manipulation step for a given input image, allowing faster and more stable text-based manipulation.
- Finally, we present a method for mapping text prompts to input-agnostic directions in StyleGAN's style space, enabling interactive text-driven image manipulation.

---

**Introduction**

<img src="https://velog.velcdn.com/images/heayounchoi/post/21dc9d3b-b23d-44e7-be61-73b8803dd489/image.png">

- existing controls enable image manipulations only along preset semantic directions, severely limiting the user's creativity and imagination
- Whenever an additional, unmapped, direction is desired, further manual effort and/or large quantities of annotated data are necessary.
- three contributions of this paper:
- (1) Text-guided latent optimization, where a CLIP model is used as a loss network. This is the most versatile approach, but it requires a few minutes of optimization to apply a manipulation to an image.
- (2) A latent residual mapper, trained for a specific text prompt. Given a starting point in latent space (the input image to be manipulated), the mapper yields a local step in latent space.
- (3) A method for mapping a text prompt into an input-agnostic (global) direction in StyleGAN's style space, providing control over the manipulation strength as well as the degree of disentanglement.

---

**Related Work**

**_Vision and Language_**

_Joint representations_

- CLIP

_Text-guided image generation and manipulation_

**_Latent Space Image Manipulation_**

- Our latent optimizer and mapper work in the W+ space, while the input-agnostic directions that we detect are in S.
- the manipulations are derived directly from text input, and our only source of supervision is a pretrained CLIP model

---

**StyleCLIP Text-Driven Manipulation**

- three ways for text-driven image manipulation
- (1) simple latent optimization scheme, where a given latent code of an image in StyleGAN's W+ space is optimized by minimizing a loss computed in CLIP space
- The optimization is performed for each (source image, text prompt) pair.
- Thus, despite it's versatility, several minutes are required to perform a single manipulation, and the method can be difficult to control.
- (2) mapping network is trained to infer a manipulation step in latent space, in a single forward pass
- The training takes a few hours, but it must only be done once per text prompt.
- The direction of the manipulation step may vary depending on the starting position in W+, which corresponds to the input image, and thus we refer to this mapper as local.
- Our experiments with the local mapper reveal that, for a wide variety of manipulations, the directions of the manipulation step are often similar to each other, despite different starting points.
- Also, since the manipulation step is performed in W+, it is difficult to achieve fine-grained visual effects in a disentangled manner.
- (3) transforms a given text prompt into an input agnostic (i.e., global in latent space) mapping direction
- The global direction is computed in StyleGAN's style space S, which is better suited for fine-grained and disentangled visual manipulation, compared to W+.

<img src="https://velog.velcdn.com/images/heayounchoi/post/bb3686a3-85ff-423c-8dc0-d8982091eb36/image.png">

---

**Latent Optimization**

<img src="https://velog.velcdn.com/images/heayounchoi/post/c89f5293-b926-4c86-a25f-842f62d0f4cb/image.png">

<img src="https://velog.velcdn.com/images/heayounchoi/post/e41a4a82-5086-4400-a49a-6417e1cb75dc/image.png">

---

**Latent Mapper**

- The latent optimization described above is versatile, as it performs a dedicated optimization for each (source image, text prompt) pair.
- On the downside, several minutes of optimization are required to edit a single image, and the method is somewhat sensitive to the values of its parameters.
- more efficient process:
- a mapping network is trained, for a specific text prompt t, to infer a manipulation step M_t(w) in the W+ space, for any given latent image embedding w

_Architecture_

<img src="https://velog.velcdn.com/images/heayounchoi/post/c14fa972-8699-46b6-8bb1-dc8811eaccf9/image.png">

- different StyleGAN layers are responsible for different levels of detail in the generated image
- it is common to split the layers into three groups (coarse, medium, and fine), and feed each group with a different part of the (extended) latent vector
- three fully-connected networks, one for each group/part
- one can choose to train only a subset of the three mappers
- There are cases where it is useful to preserve some attribute level and keep the style codes in the corresponding entries fixed.

<img src="https://velog.velcdn.com/images/heayounchoi/post/f9ff680d-be0f-4dd2-b421-e90bc0942517/image.png">

_Losses_

<img src="https://velog.velcdn.com/images/heayounchoi/post/3bbaa9d9-d25c-43c5-a082-fede94eab6c5/image.png">

<img src="https://velog.velcdn.com/images/heayounchoi/post/3e72d5d7-2f1a-4a1e-9735-0fde631a7f94/image.png">

- for edits that require identity preservation, we use the identity loss

<img src="https://velog.velcdn.com/images/heayounchoi/post/ffbdba40-5138-4b45-9182-f6e6f6ed2341/image.png">

<img src="https://velog.velcdn.com/images/heayounchoi/post/57e3355b-bb6c-4c50-ab12-e9788578b1a0/image.png">

<img src="https://velog.velcdn.com/images/heayounchoi/post/9f321a74-d9b7-4fde-8f66-8b3b1c51d74a/image.png">

- manipulation directions are not as different as one might expect

---

**Global Directions**

- While the latent mapper allows fast inference time, we find that it sometimes falls short when a fine-grained disentagled manipulation is desired.
- Furthermore, as we have seen, the directions of different manipulation steps for a given text prompt tend to be similar.
- Motivated by these observations, in this section we propose a method for mapping a text prompt into a single, global direction in StyleGAN's style space S, which has been shown to be more disentangled than other latent spaces.

_From natural language to delta t_

- in order to reduce text embedding noise, prompt engineering that feeds several sentences with the same meaning to the text encoder and averages their embeddings can be used
- our method is provided with text description of a target attribute and a corresponding neutral class

_Channelwise relevance_

<img src="https://velog.velcdn.com/images/heayounchoi/post/c0be91ce-d522-4c6a-b5b6-60177e4ceeb9/image.png">

<img src="https://velog.velcdn.com/images/heayounchoi/post/eacdb51d-c62e-4606-affb-40970b9d680b/image.png">

<img src="https://velog.velcdn.com/images/heayounchoi/post/bc1ab3a8-a759-413d-96a1-a884feb94664/image.png">

<img src="https://velog.velcdn.com/images/heayounchoi/post/531b72ff-a0ad-4c03-805e-8156e66998c0/image.png">

---

**Comparisons and Evaluation**

_Text-driven image manipulation methods_

<img src="https://velog.velcdn.com/images/heayounchoi/post/e25297a1-9f79-4068-8dd6-8d97d37e5680/image.png">

- for complex and specific attributes (especially those that involve identity), the mapper is able to produce better manipulations
- for simpler and/or more common attributes, a global direction suffices, while offering more disentangled manipulations

_Other StyleGAN manipulation methods_

<img src="https://velog.velcdn.com/images/heayounchoi/post/32340fbe-2236-4bdc-8616-da44a1071b85/image.png">

_Limitations_

- relies on pretrained StyleGAN generator and CLIP model for a joint language-vision embedding
- Thus, it cannot be expected to manipulate images to a point where they lie outside the domain of the pretrained generator (or remain inside the domain, but in regions that are less well covered by the generator).
- Similarly, text prompts which map into areas of CLIP space that are not well populated by images, cannot be expected to yield a visual manipulation that faithfully reflects the semantics of the prompt.
- drastic manipulations in visually diverse datasets are difficult to achieve

---

**Conclusions**

---

**Latent Mapper - Ablation Study**

**_Architecture_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/36b2d562-cddb-4545-9e44-bc0e11cf9c6a/image.png">

<img src="https://velog.velcdn.com/images/heayounchoi/post/92a8740b-bb90-4279-886e-b2b82d9aafc9/image.png">

**_Losses_**

_CLIP Loss_

<img src="https://velog.velcdn.com/images/heayounchoi/post/c9ca5490-5548-472b-9164-ba81f87fc7be/image.png">

_ID Loss_

<img src="https://velog.velcdn.com/images/heayounchoi/post/e622a344-1816-4e65-ae3b-4211ddd1a7fb/image.png">

---

**Additional Results**

---

**Video**
