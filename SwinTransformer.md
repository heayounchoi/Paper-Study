### [Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/pdf/2103.14030)

**Abstract**
- Challenges in adapting Transformer from language to vision arise from differences between the two domains, such as large variations in the scale of visual entities and the high resolution of pixels in images compared to words in text.
- The shifted windowing scheme brings greater efficiency by limiting self-attention computation to non-overlapping local windows while also allowing for cross-window connection.
- This hierarchical architecture has the flexibility to model at various scales and has linear computational complexity with respect to image size.
---

**Introduction**
- Transformer is notable for its use of attention to model long-range dependencies in the data.
- In existing Transformer-based models, tokens are all of a fixed scale, a property unsuitable for vision applications such as object detection. (scale variance)
- There exist many vision tasks such as semantic segmentation that require dense prediction at the pixel level, and this would be intractable for Transformer on high-resolution images, as the computational complexity of its self-attention is quadratic to image size.
- Swin Transformer constructs hierarchical feature maps and has linear computational complexity to image size.

<img src="https://velog.velcdn.com/images/heayounchoi/post/c514d0b4-9aa9-47b7-aef4-948ddcbf0522/image.png">

- With these hierarchical feature maps, the Swin Transformer model can conveniently leverage advanced techniques for dense prediction such as FPN or U-Net.
- These merits make Swin Transformer suitable as a general-purpose backbone for various vision tasks, in contrast to previous Transformer based architectures which produce feature maps of a single resolution and have quadratic complexity.

<img src="https://velog.velcdn.com/images/heayounchoi/post/8a64c092-1787-4a65-a644-285b7d7f1984/image.png">

- This strategy is also efficient in regards to real-world latency: all query patches within a window share the same key set, which facilitates memory access in hardware.
- In contrast, earlier sliding window based self-attention approaches suffer from low latency on general hardware due to different key sets for different query pixels.
- It is our belief that a unified architecture across computer vision and natural language processing could benefit both fields, since it would facilitate joint modeling of visual and textual signals and the modeling knowledge from both domains can be more deeply shared.
---

**Related Work**

_CNN and variants_

_Self-attention based backbone architectures_
- The self-attention is computed within a local window of each pixel to expedite optimization, and they achieve slightly better accuracy/FLOPs trade-offs than the couterpart ResNet architecture.
- However, their costly memory access causes their actual latency to be significantly larger than that of the convolutional networks.

_Self-attention/Transformers to complement CNNs_
- The self-attention layers can complement backbones or head networks by providing the capability to encode distant dependencies or heterogeneous interactions.
- More recently, the encoder-decoder design in Transformer has been applied for the object detection and instance segmentation tasks.

_Transformer based vision backbones_
- The pioneering work of ViT directly applies a Transformer architecture on non-overlapping medium-sized image patches for image classification.
- It achieves an impressive speed-accuracy trade-off on image classification compared to convolutional networks.
- While ViT requires large-scale training datasets to perform well, DeiT introduces several training strategies that allow ViT to also be effective using the smaller ImageNet-1K dataset.
- The results of ViT on image classification are encouraging, but its architecture is unsuitable for use as a general-purpose backbone network on dense vision tasks or when the input image resolution is high, due to its low-resolution feature maps and the quadratic increase in complexity with image size.
---

**Method**

**_Overall Architecture_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/71cf2179-84cc-4857-84d5-a84299745cac/image.png">

- To produce a hierarchical representation, the number of tokens is reduced by patch merging layers as the network gets deeper.
- The first patch merging layer concatenates the features of each group of 2x2 neighboring patches, and applies a linear layer on the 4C-dimensional concatenated features.
- This reduces the number of tokens by a multiple of 2x2=4 (2x downsampling of resolution), and the output dimension is set to 2C.

_Swin Transformer block_
- A Swin Transformer block consists of a shifted window based MSA module.

**_Shifted Window based Self-Attention_**

_Self-attention in non-overlapped windows_
- compute self-attention within local windows
- The windows are arranged to evenly partition the image in a non-overlapping manner.

_Shifted window partitioning in successive blocks_
- The window-based self-attention module lacks connections across windows, which limits its modeling power.
- The first module uses a regular window partitioning strategy which starts from the top-left pixel, and the 8x8 feature map is evenly partitioned into 2x2 windows of size 4x4 (M=4).
- Then, the next module adopts a windowing configuration that is shifted from that of the preceding layer, by displacing the windows by (M/2, M/2) pixels from the regularly partitioned windows.

_Efficient batch computation for shifted configuration_

<img src="https://velog.velcdn.com/images/heayounchoi/post/75723a5e-0177-4d62-8c89-5c913b91c064/image.png">

- An issue with shifted window partitioning is that it will result in more windows in the shifted configuration, and some of the windows will be smaller than MxM.
- We propose an efficient batch computation approach by cyclic-shifting toward the top-left direction.
- After this shift, a batched window may be composed of several sub-windows that are not adjacent in the feature map, so a masking mechanism is employed to limit self-attention computation to within each sub-window.
- With the cyclic-shift, the number of batched windows remains the same as that of regular window partitioning, and thus is also efficient.

_Relative position bias_
- relative position bias to each head in computing similarity

<img src="https://velog.velcdn.com/images/heayounchoi/post/8b6b9697-b12c-4588-88f0-4ce91a4bc380/image.png">

**_Architecture Variants_**

---

**Experiments**

**_Image Classification on ImageNet-1K_**

_Settings_

_Results with regular ImageNet-1K training_

<img src="https://velog.velcdn.com/images/heayounchoi/post/b8f28614-9a6a-44ed-9ed6-21fd5e76aa30/image.png">

- While RegNet and EfficientNet are obtained via a thorough architecture search, the proposed Swin Transformer is adapted from the standard Transformer and has strong potential for further improvement.

_Results with ImageNet-22K pre-training_

<img src="https://velog.velcdn.com/images/heayounchoi/post/b38d6d8d-d241-48fd-a8ac-a14c115e6d89/image.png">

**_Object Detection on COCO_**

_Settings_

_Comparison to ResNe(X)t_

<img src="https://velog.velcdn.com/images/heayounchoi/post/3b3eeb74-d46e-45d3-ac2e-d4c9f0d9a461/image.png">

<img src="https://velog.velcdn.com/images/heayounchoi/post/9c31bc74-1447-48e1-9a77-4f0552855ee6/image.png">

<img src="https://velog.velcdn.com/images/heayounchoi/post/ed2efde1-4a61-4212-ac75-df10ce2af5bf/image.png">

_Comparison to DeiT_
- The lower inference speed of DeiT is mainly due to its quadratic comlexity to input image size.

_Comparison to previous state-of-the-art_

**_Semantic Segmentation on ADE20K_**

_Settings_

_Results_

<img src="https://velog.velcdn.com/images/heayounchoi/post/501ef0e0-fa64-4c9f-86c5-27e68a3679ad/image.png">

**_Ablation Study_**

_Shifted windows_

_Relative position bias_
- While the inclusion of absolute position embedding improves image classification accuracy, it harms object detection and semantic segmentation.
- While the recent ViT/DeiT models abandon translation invariance in image classification even though it has long been shown to be crucial for visual modeling, we find that inductive bias that encourages certain translation invariance is still preferable for general-purpose visual modeling, particularly for the dense prediction tasks of object detection and semantic segmentation.

_Different self-attention methods_

---

**Conclusion**

---

**Detailed Architectures**

<img src="https://velog.velcdn.com/images/heayounchoi/post/883c1a84-08c4-4ecd-a828-5d6393ca276c/image.png">

---

**Detailed Experimental Settings**

**_Image classification on ImageNet-1K_**
- GAP on the output feature map of the last stage, followed by a linear classifier
- accurate as using an additional class token as in ViT and DeiT

_Regular ImageNet-1K training_

_ImageNet-22K pre-training_

**_Object detection on COCO_**

**_Semantic segmentation on ADE20K_**

---

**More Experiments**

**_Image classification with different input size_**

**_Different Optimizers for ResNe(X)t on COCO_**
- improved accuracy with AdamW optimizer for smaller backbones

**_Swin MLP-Mixer_**
