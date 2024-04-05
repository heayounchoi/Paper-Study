### [Feature Pyramid Networks for Object Detection](https://arxiv.org/pdf/1612.03144.pdf)

**Abstract**
- Feature pyramids are a basic component in recognition systems for detecting objects at different scales.
- But recent deep learning object detectors have avoided pyramid representations, in part because they are compute and memory intensive.
- This paper exploits the inherent multi-scale, pyramidal hierarchy of deep convolutional networks to construct feature pyramids with marginal extra cost.
- A top-down architecture with lateral connections is developed for building high-level semantic feature maps at all scales.
- Using FPN in a basic Faster R-CNN system, this method achieves SOTA single-model results on the COCO detection benchmark.
- This method can run at 6 FPS on a GPU.
---

**Introduction**

<img src="https://velog.velcdn.com/images/heayounchoi/post/664c6494-3ef2-4989-92d3-5a16462d8288/image.png" width="50%">

- Feature image pyramids (feature pyramids built upon image pyramids) are scale-invariant in the sense that an object's scale change is offset by shifting its level in the pyramid.
- ConvNetsn are capable of representing higher-level semantics, and are also more robust to variance in scale, thus facilitating recognition from features computed on a single input scale.
- To get more accurate results, pyramids are still needed. 
- The principle advantage of featurizing each level of an image pyramid is that it produces a multi-scale feature representation in which all levels are semantically strong, including the high-resolution levels.
- But featurizing each level of an image pyramid increases inference time considerably, and training deep networks end-to-end on an image pyramid is also infeasible in terms of memory. 
- For these reasons, Fast and Faster R-CNN opt to not use featurized image pyramids under default settings.
- A deep ConvNet's in-network feature hierarchy produces feature maps of different spatial resolutions, but introduces large semantic gaps caused by different depths. The high-resolution maps have low-level features that harm their representational capacity for object recognition.
- The Single Shot Detector (SSD) is one of the first attempts at using a ConvNet's pyramidal feature hierarchy as if it were a featurized image pyramid.
- The SSD-style pyramid reuses the multi-scale feature maps from different layers computed in the forward pass and thus come free of cost.
- But to avoid using low-level features SSD foregoes reusing already computed layers and instead builds the pyramid starting from high up in the network and then by adding several new layers.
- Thus it misses the opportunity to reuse the higher-resolution maps of the feature hierarchy, which are important for detecting small objects.
- The goal of this paper is to naturally leverage the pyramidal shape of a ConvNet's feature hierarchy while creating a feature pyramid that has strong semantics at all scales.
- The architecture combines low-resolution, semantically strong features with high-resolution, semantically weak features via a top-down pathway and lateral connections.
- The result is a feature pyramid that has rich semantics at all levels and is built quickly from a single input image scale.
---

**Related Work**
**_Hand-engineered features and early neural networks_**
- SIFT, HOG: computed densely over entire image pyramids
- shallow ConvNets over image pyramids

**_Deep ConvNet object detectors_**
- OverFeat adopted a strategy similar to early neural network face detectors by applying a ConvNet as a sliding window detector on an image pyramid.
- R-CNN adopted a region proposal-based strategy in which each proposal was scale-normalized before classifying with a ConvNet. (warping)
- SPPnet demonstrated that such region-based detectors could be applied much more efficiently on feature maps extracted on a single image scale.
- Fast/Faster R-CNN advocate using features from single scale form accuracy and speed but multi-scale detection performs better for small objects.

**_Methods using multiple layers_**
- segmentation models concatenate features of multiple layers before computing predictions, which is equivalent to summing transformed features
- SSD and MS-CNN predict objects at multiple layers of the feature hierarchy without combining features or scores.

<img src="https://velog.velcdn.com/images/heayounchoi/post/93863b5a-4c9a-438b-836d-56947a73efd7/image.png" width="50%">

- There are methods exploiting lateral/skip connections that associate low-level feature maps across resolutions and semantic levels, but predictions are made only once on the finest level. Image pyramids are still needed to recognize objects across multiple scales.
---

**Feature Pyramid Networks**
- The method takes a single-scale image of an arbitrary size as input, and outputs proportionally sized feature maps at multiple levels, in a fully convolutional fashion.
- This process is independent of the backbone convolutional architectures.
- The construction of the pyramid involves a bottom-up pathway, a top-down pathway, and lateral connections.

**_Bottom-up pathway_**
- The bottom-up pathway is the feed-forward computation of the backbone ConvNet, which computes a feature hierarchy consisting of feature maps at several scales with a scaling step of 2. (ex. 4, 8, 16, 32 ...)

**_Top-down pathway and lateral connections_**
- The top-down pathway hallucinates higher resolution features by upsampling spatially coarser, but semantically stronger, feature maps from higher pyramid levels.
- These features are then enhanced with features from the bottom-up pathway via lateral connections. 
- Each lateral connection merges feature maps of the same spatial size from the bottom-up pathway and the top-down pathway.
- The bottom-up feature map is of lower-level semantics, but its activations are more accurately localized as it was subsampled fewer times.

<img src="https://velog.velcdn.com/images/heayounchoi/post/5ae4783e-d359-4480-a12c-96ba58667bea/image.png" width="50%">

- With a coarser-resolution feature map, the spatial resolution is upsampled by a factor of 2 (using nearest neighbor upsampling for simplicity).
- The upsampled map is then merged with the corresponding bottom-up map by element-wise addition.
- 3x3 convolution is appended on each merged map to generate the final feature map to reduce the aliasing effect of upsampling.
- All extra convolutional layers have 256-channel outputs and there are no non-linearities in these extra layers.
---

**Applications**
**_Feature Pyramid Networks for RPN_**
- single-scale feature map of RPN is replaced by FPN.
- a head of the same design (3x3 conv and two sibling 1x1 convs) is attached to each level on the feature pyramid.
- because the head slides densely over all locations in all pyramid levels, it is not necessary to have multi-scale anchors on a specific level.
- Instead, a single scale is assigned to each level.
- anchors have areas of (32^2, 64^2, 128^2, 256^2, 512^2 pixels with multiple aspect ratios. (15 anchors over the pyramid)
- the parameters of the heads are shared across all feature pyramid levels

**_Feature Pyramid Networks for Fast R-CNN_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/9d39c71d-6b1b-46ac-8ce4-2cfc5a3d2f32/image.png" width="50%">

- Fast R-CNN predictor heads are attached to all RoIs of all levels.
- heads all share parameters regardless of their levels
---

**Experiments on Object Detection**
**_Region Proposal with RPN_**

_Implementation details_
- Input image resized such that its shorter side has 800 pixels
- mini-batch size: 2
- 256 anchors per image
- weight decay: 0.0001
- momentum: 0.9
- learning rate: 0.02 -> 0.002
- anchor boxes outside the image not ignored for training

_Ablation Experiments_

<img src="https://velog.velcdn.com/images/heayounchoi/post/e3e4be95-ab0e-4dac-aab6-791fd40211c6/image.png" width="50%">

- performance on small objects boosted by a large margin
- (f) leads to more anchors caused by its large spatial resolution

**_Object Detection with Fast/Faster R-CNN_**

_Implementation details_
- mini-batch: 2
- 512 RoIs per image
