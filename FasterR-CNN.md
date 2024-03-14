### Faster R-CNN: Towards Real-Time Object Detection with Region Proposal Networks

**Abstract**
- SOTA object detection networks depend on region proposal algorithms to hypothesize object locations
- in this work, a Region Proposal Netwok (RPN) that shares full-image conv features with the detection network is introduced, thus enabling nearly cost-free region proposals
- an RPN is a fully conv network that simultaneously predicts object bounds and objectness scores at each position
- for the very deep VGG-16 model, Faster R-CNN has a frame rate of 5fps on a GPU, while achieving SOTA object detection accuracy with only 300 proposals per image
---

**Related Work**

_Object Proposals_
- grouping super-pixels: Selective Search, CPMC, MCG
- sliding windows: objectness in windows, EdgeBoxes
- object proposal methods were adopted as external modules independent of the detectors

_Deep Networks for Object Detection_
- R-CNN mainly plays as a classifier, and it does not predict object bounds (except for refining by bounding box regression)
- its accuracy depends on the performance of the region proposal module
- in the OverFeat method, a fc layer is trained to predict the box coordinates for the localization task that assumes a single object
- the fc layer is then turned into a conv layer for detecting multiple class-specific objects
- the MultiBox methods, generate region proposals from a network whose last fc layer simultaneously predicts multiple class-agnostic boxes, generalizing the "single-box" fashion of OverFeat
- these class-agnostic boxes are used as proposals for R-CNN
- the MultiBox proposal network is applied on a single image crop or multiple large image crops
- MultiBox does not share features between the proposal and detection networks
- shared computation of conv has been attracting increasing attention for efficient, yet accurate, visual recognition
- OverFeat computes conv features from an image pyramid for classification, localization, and detection
- adaptively-sized pooling (SPP) on shared conv feature maps is developed for efficient region-based object detection and semantic segmentation
- Fast R-CNN enables end-to-end detector training on shared conv features and shows compelling accuracy and speed
---

**Faster R-CNN**
- Faster R-CNN is composed of two modules:
> 1) a deep fully conv network that proposes regions
> 2) Fast R-CNN detector that uses the proposed regions

<img src="https://velog.velcdn.com/images/heayounchoi/post/eb4ecc28-3ea8-4083-b8bc-d51c57d600ba/image.png" width="50%">

**_Region Proposal Networks_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/27dc2d70-14ca-47bb-b28b-d2b6fed76e5e/image.png" width="50%">

- a RPN takes an image (of any size) as input and outputs a set of rectangular object proposals, each with an objectness score
- to generate region proposals, Faster R-CNN slides a small network over the conv feature map output by the last shared conv layer
- this small network takes as input n x n spatial window of the input conv feature map
- each sliding window is mapped to a lower-dimensional feature
- this feature is fed into two sibling fc layers--a box-regression layer (reg) and a box-classification layer (cls)

_Anchors_
- at each sliding-window location, we simultaneously predict multiple region proposals, where the number of maximum possible proposals for each location is denoted as k
- the reg layer has 4k outputs encoding the coordinates of k boxes, and the cls layer outputs 2k scores that estimate probability of object or not object for each proposal
- the k proposals are parameterized relative to k reference boxes, which we call anchors
- an anchor is centered at the sliding window in question, and is associated with a scale and aspect ratio
- by default, 3 scales and 3 aspect ratios are used, yielding k=9 anchors at each sliding position
- for a conv feature map of a size W x H, there are WHk anchors in total

**_Translation-Invariant Anchors_**
- an important property of this paper's approach is that it is translation invariant, both in terms of the anchors and the functions that compute proposals relative to the anchors
- the MultiBox method uses k-means to generate 800 anchors, which are not translation invariant
- MultiBox does not gaurantee that the same proposal is generated if an object is translated
- the translation-invariant property also reduces the model size
- MultiBox has a (4+1)*800-dimensional fc output layer, whereas this paper's method has a (4+2)*9-dimensional conv output layer in the case of k=9 anchors
- as a result, this paper's output layer has two orders of magnitude fewer parameters than MultiBox's ouput layer

**_Multi-Scale Anchors as Regression References_**
- ways for multi-scale predictions:
> 1) image/feature pyramids
> 2) pyramid of filters
> 3) pyramid of anchors

_Loss Function_
- for training RPNs, a binary class label (of being an object or not) is assigned to each anchor
- positive label:
> 1) anchor(s) with the highest IoU overlap with a ground-truth box
> 2) anchor that has an IoU overlap higher than 0.7 with any ground-truth box
- negative label: IoU ratio lower than 0.3 for all ground-truth boxes
- anchors that are neither positive nor negative do not contribute to the training objective
<img src="https://velog.velcdn.com/images/heayounchoi/post/2b2f606a-b975-4944-972f-ab2069751f8f/image.png" width="50%">

- in other methods, bounding-box regression is performed on features pooled from arbitrarily sized RoIs, and the regression weights are shared by all region sizes
- in faster r-cnn, the features used for regression are of the same spatial size (3x3) on the feature maps
- to account for varying sizes, a set of k bounding-box regressors are learned
- each regressor is responsible for one scale and one aspect ratio, and the k regressors do not share weights

_Training RPNs_
- RPN can be trained end-to-end by backpropagation and SGD
- each mini-batch arises from a single image that contains many positive and negative example anchors
- 256 anchors are randomly sampled in an image to compute the loss function of a mini-batch, where the sampled positive and negative anchors have a ratio of up to 1:1
- weight initialization for new layers: zero-mean Gaussian distribution with standard deviation 0.01
- all other layers (ex. shared conv layers) initialized by pre-trained model for ImageNet classification
- learning rate of 0.001 for 60k mini-batches, and 0.0001 for the next 20k mini-batches on PASCAL VOC dataset
- momentum: 0.9
- weight decay: 0.0005

**_Sharing Features for RPN and Fast R-CNN_**
- three ways for training networks with features shared:
> 1) alternating training: first train RPN and use the proposals to train Fast R-CNN. the network tuned by fast r-cnn is then used to initialize RPN, and this process is iterated.
> 2) approximate joint training: 두 모델을 합쳐서 학습시킴. 하지만 proposal boxes' coordinates에 대한 미분은 무시함. 학습 시간이 줄어드는 장점이 있음
> 3) non-approximate joint training: needs an RoI pooling layer that is differentiable w.r.t. the box coordinates. solution can be given by an "RoI warping" layer

**_Implementation Details_**
- single-scale training with shorter side s = 600 pixels
- on the re-scaled images, the total stride for both ZF and VGG nets on the last conv layer is 16 pixels and thus is ~10 pixels on a typical PASCAL image before resizing (~500x375)
- accuracy may be further improved with a smaller stride
- for anchors, 3 scales with box areas of 128^2, 256^2, and 512^2 pixels and 3 aspect ratios of 1:1, 1:2, and 2:1
<img src="https://velog.velcdn.com/images/heayounchoi/post/ef2dd3f1-854d-4133-a467-d84eeef56f80/image.png" width="50%">

- during training, all cross-boundary anchors are ignored so they do not contribute to the loss
- to reduce redundancy, NMS is adopted on the proposal regions based on their cls scores (0.7)
- after NMS, top-N ranked proposal regions are used for detection

---

**Experiments**
**_Experiments on PASCAL VOC_**
- for the ImageNet pre-trained network, the "fast" version of ZF net that has 5 conv layers and 3 fc layers, and the public VGG-16 model that has 13 conv layers and 3 fc layers is used
- using RPN yields a much faster detection system than using either SS or EB because of shared conv computations; the fewer proposals also reduce the region-wise fc layers' cost
<img src="https://velog.velcdn.com/images/heayounchoi/post/d4be52c3-b30e-4fcd-8797-bc0dc1e4a161/image.png" width="50%">

**_Experiments on MS COCO_**
- mAP@\[.5, .95] is evaluated
- trained on 8-GPU implementation
- mini-batch 8 for RPN and 16 for Fast R-CNN
- both trained for 240k iteration with learning rate 0.003, and for 80k iterations with 0.0003
- for the anchors, 3 aspect ratios and 4 scales (adding 64^2)
- in Fast R-CNN step, the negative samples are defined as those with a maximum IoU with ground truth in the interval of \[0, 0.5)
- in the SPPnet system, the negative samples in \[0.1, 0.5) are used for network fine-tuning, but the negative samples in \[0, 0.5) are still visited in the SVM step with hard-negative mining
- but the Fast R-CNN system abandons the SVM step, so the negative samples in \[0, 0.1) are never visited
- including these \[0, 0.1) samples improves mAP@0.5 on the COCO dataset (복잡한 배경) for both Fast R-CNN and Faster R-CNN systems (but the impact is negligible on PASCAL VOC)
