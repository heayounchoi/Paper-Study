### [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/pdf/1505.04597)

**Abstract**
- In this paper, we present a network and training strategy that relies on the strong use of data augmentation to use the available annotated samples more efficiently.
- The architecture consists of a contracting path to capture context and a symmetric expanding path that enables precise localization.
- fast network; segmentation of a 512x512 image takes less than a second on a recent GPU
---

**Introduction**
- in biomedical image processing, the desired output should include localization
- thousands of training images are usually beyond reach in biomedical tasks
- recent work: network in a sliding-window setup to predict the class label of each pixel by providing a local region (patch) around that pixel as input
- this network can localize and the training data in terms of patches is much larger than the number of training images
- two drawbacks:
- 1) slow because the network must be run separately for each patch, and a lot of redundancy due to overlapping patches
- 2) trade-off between localization accuracy and the use of context
- more recent approaches proposed a classifier output that takes into account the features from multiple layers
- U-Net modifies and extends FCN such that it works with very few training images and yields more precise segmentations

<img src="https://velog.velcdn.com/images/heayounchoi/post/23ff23e9-49b5-4064-a253-cdf87564932e/image.png">

- The main idea in FCN is to supplement a usual contracting network by successive layers, where pooling operators are replaced by upsampling operators.
- In order to localize, high resolution features from contracting path are combined with the upsampled output.
- A successive convolution layer can then learn to assemble a more precise output based on this information. (FCN 다시 읽기)
- One important modification in our architecture is that in the upsampling part we have also a large number of feature channels, which allow the network to propagate context information to higher resolution layers.
- As a consequence, the expansive path is more or less symmetric to the contracting path, and yields a u-shaped architecture.
- This strategy allows the seamless segmentation of arbitrarily large images by an overlap-tile strategy.

<img src="https://velog.velcdn.com/images/heayounchoi/post/2db754e5-63c5-4a9a-8a69-7d9ccf430f66/image.png">

- To predict the pixels in the border region of the image, the missing context is extrapolated by mirroring the input image.
- This tiling strategy is important to apply the network to large images, since otherwise the resolution would be limited by the GPU memory.
- excessive data augmentation by applying elastic deformations to training images
- important in biomedical segmentation since deformation used to be the most common variation in tissue
- another challenge: separation of touching objects of the same class
- large weight on the separating background labels between touching cells

<img src="https://velog.velcdn.com/images/heayounchoi/post/994af678-edb8-40b1-aea5-49df61385e1d/image.png">

---

**Network Architecture**
- contracting path: repeated application of two 3x3 convs (unpadded), ReLLU, 2x2 max pooling with stride 2 for downsampling
- double the number of feature channesl at each downsampling step
- expansive path: upsampling of the feature map on every step, 2x2 conv ("up-convolution") that halves the number of feature channels, a concatenation with the correspondingly cropped feature map from the contracting path, two 3x3 convs, ReLU
- cropping necessary due to the loss of border pixels in every convolution
- final layer: 1x1 conv to map each 64-component feature vector to the desired number of classes
- total 23 conv layers
- to allow a seamless tiling of the output segmentation map, it is important to select the input tile size such that all 2x2 max-pooling operations are applied to a layer with an even x- and y-size
---

**Training**
- SGD optimize
- due to the unpadded convs, the output image is smaller than the input by a constant border width
- to minimize overhead and make maximum use of the GPU memory, large input tiles > large batch size -> single batch
- high momentum (0.99) such that a large number of the previously seen training samples determine the update in the current optimization step
- pre-compute the weight map for each ground truth segmentation to compensate the different frequency of pixels from a certain class in the training dataset, and to force the network to learn the small separation borders between touching cells

<img src="https://velog.velcdn.com/images/heayounchoi/post/55bf7db4-0fb2-4521-852a-28ea20a98917/image.png">

- initial weights from Gaussian distribution with a standard deviation of $$\sqrt{2/N}$$, where N denotes the number of incoming nodes of one neuron

**_Data Augmentation_**
- deformations (displacement), dropout
---

**Experiments**
- 30 training images / 35 training images / 20 training images

<img src="https://velog.velcdn.com/images/heayounchoi/post/fbf29053-25a5-40a0-ab80-6a82146f1b80/image.png">

