**Abstract**
- employs several innovations to improve training and testing speed while also increasing detection accuracy
- trains the very deep VGG16 network 9x faster than R-CNN
- 213x faster at test-time
- achieves a higher mAP on PASCAL VOC 2012
---

**Introduction**
**_R-CNN and SPPnet_**
- drawbacks of R-CNN:
> 1) Training is a multi-stage pipeline: first fine-tunes a ConvNet on object proposals, then fits SVMs to ConvNet features to replace the softmax classifiers, and lastly learns bounding-box regressors
> 2) Training is expensive in space and time: for SVM and bounding-box regressor training, features are extracted from each object proposal in each image and written to disk
> 3) Object detection is slow: at test-time, features are extracted from each object proposal in each test image, taking 47s/image on detection with VGG16 (on a GPU)

- R-CNN is slow b/c it performs a ConvNet forward pass for each object proposal, without sharing computation
- SPPnets (spatial pyramid pooling networks) speeds up R-CNN by sharing computation
- SPPnet computes a conv feature map for the entire input image and then classifies each object proposal using a feature vector extracted from the shared feature map
- SPPnet accelerates R-CNN by 10 to 100x at test time and 3x at trainig time
- however, fine-tuning algorithm cannot update the conv layers that precede the spatial pyramid pooling and this limitation (fixed conv layers) limits the accuracy of very deep networks

**_Contributions_**
- advantages of Fast R-CNN:
> 1) higher detection quality (mAP) than R-CNN, SPPnet
> 2) trainig is single-stage, using a multi-task loss
> 3) training can update all network layers
> 4) no disk storage is required for feature caching
---

**Fast R-CNN architecture and training**

<img src="https://velog.velcdn.com/images/heayounchoi/post/a33eca5e-932b-43eb-aa45-a53a7c855e39/image.png" width="50%">

- takes as input an entire image and a set of object proposals
- first processes the whole image with several conv and max pooling layers to produce a conv feature map
- then, for each object proposal a region of interest (RoI) pooling lyaer extracts a fixed-length feature vector from the feature map
- each feature vector is fed into a sequence of fc layers that finally branch into two sibling output layers: one that produces softmax probability estimates over K object classes plus a catch-call "background" class and another layer that outputs four real-valued numbers for each of the K object classes (bounding-box positions)

**_The RoI pooling layer_**
- uses max pooling to convert the features inside any valid region of interest into a small feature map with a fixed spatial extent of H x W, where H and W are layer hyper-parameters that are independent of any particular RoI
- each RoI is defined by a four-tuple (r, c, h, w) that specifies its top-left corner (r, c) and its height and width
- works by dividing the h x w RoI window into an H x W grid of sub-windows of approximate size h/H x w/W and then max-pooling the values in each sub-window into the corresponding output grid cell

**_Initializing from pre-trained networks_**
- pre-trained ImageNet networks config: five max pooling layers and between five and thirteen conv layers
- three transformations when initializing a Fast R-CNN net with a pre-trained net:
> 1) replace last max pooling layer by a RoI pooling layer that is configured by setting H and W to be compatiblbe with the net's first fc layer (ex. H = W = 7 for VGG16)
> 2) replace last fc layer and softmax with the two sibling layers described earlier
> 3) modify network to take two data inputs

**_Fine-tuning for detection_**
- training all network weights with back-propagation is an important capability of Fast R-CNN
- why SPPnet is unable to update weights below the spatial pyramid pooling layer?
> - back-propagation through the SPP layer is highly inefficient when each training sample (ex. RoI) comes from a different image, which is exactly how R-CNN and SPPnet networks are trained
- in Fast R-CNN training, SGD mini batches are sampled hierarchically, first by sampling N images and then by sampling R/N RoIs from each image
- RoIs from the same image share computation and memory in the forward and backward passes

_Multi-task loss_

<img src="https://velog.velcdn.com/images/heayounchoi/post/1d9b2fc0-675c-4d4d-aac4-6b5a6273037d/image.png" width="50%">
<img src="https://velog.velcdn.com/images/heayounchoi/post/462170ad-c75e-4c8f-8925-49d10afe07cf/image.png" width="50%">

_Mini-batch sampling_
- N = 2 images
- R = 128, sampling 64 RoIs from each image
- 25% of the RoIs that have IoU overlap of at least 0.5 labeled as a foreground object class
- remaining RoIs that have a maximum IoU with ground truth betweeen 0.1 and 0.5 are background exmaples
- during training, horiz flip with probability of 0.5

_Back-propagation through RoI pooling layers_

<img width="50%" alt="image" src="https://github.com/heayounchoi/Paper-Study/assets/118031423/bf7b7731-43d1-43be-a5d1-0b0518fc841b">
<img width="50%" alt="image" src="https://github.com/heayounchoi/Paper-Study/assets/118031423/56a3310b-289f-4686-9825-20f6d35231eb">

_SGD hyper-parameters_
- fc layers are initialized from zero-mean Gaussian distributions with standard deviations 0.01 (softmax) and 0.001 (regression) and biases are initialized to 0
- all layers use a per-layer learning rate of 1 for weights and 2 for biases and a global learning rate of 0.001
- on VOC07 or VOC12 trainval, SGD for 30k mini-batch iterations and learning rate lowered to 0.0001 and trained for another 10k iterations
- when trained on larger datasets, SGD is ran for more iterations
- momentum 0.9
- parameter decay 0.0005

**_Scale invariance_**

(1) "brute force" learning
- each image is processed at a pre-defined pixel size during both training and testing
- network must directly learn scale-invariant object detection from the training data

(2) image pyramids
- provides approximate scale-invariance to the network through an image pyramid
- at test time, image pyramid is used to approximately scale-normalize each object proposal
- during multi-scale training, a pyramid scale is randomly sampled each time an image is samples, as a form of data augmentation
---

**Fast R-CNN detection**
- at test-time, R is typically around 2000
- when using an image pyramid, each RoI is assigned to the scale such that the scaled RoI is closest to 224^2 pixels in area

**_Truncated SVD for faster detection_**
- for whole-image classification, the time spent computing the fc layers is small compared to the conv layers
- on the contrary, for detection the number of RoIs to process is large and nearly half of the forward pass time is spent computing the fc layers
- large fc layers are easily accelerated by compressing them with truncated SVD

<img src="https://velog.velcdn.com/images/heayounchoi/post/66fec899-24bc-4b45-a5b9-1e568e9bc4bd/image.png" width="50%">
<img src="https://velog.velcdn.com/images/heayounchoi/post/42bcc978-561a-4cf0-90f2-d64e6e610c6f/image.png" width="50%">

---

**Main results**
**_Experimental setup_**
- three pre-trained ImageNet models are used
> - S: CaffeNet from R-CNN
> - M: VGG_CNN_M_1024, same depth as S but wider
> - L: VGG16
- all experiments use single-scale training and testing (s=600)

**_Training and test time_**
- for VGG16, Fast R-CNN processes images 146x faster than R-CNN without truncated SVD and 213x faster with it
- training time is reduced by 9x, from 84 hours to 9.5
- also eliminates hundreds of gigabytes of disk storage, because it does not cache features

_Truncated SVD_
- can reduce detection time by more than 30% with only a small drop in mAP and without needing to perform additional fine-tuning after model compression
- further speed-ups are possible with smaller drops in mAP if one fine-tunes again after compression

**_Which layers to fine-tune?_**
- training through the RoI pooling layer is important for very deep nets
- conv1 is generic and task independent, so not all conv layers need to be fine-tuned
---

**Design evaluation**

**_Does multi-task training help?_**
- multi-task training improves pure classification accuracy relative to training for classification alone

**_Scale invariance: to brute force or finesse?_**
- deep ConvNets are adept at directly learning scale invariance
- single-scale processing offers the best tradeoff between speed and accuracy

**_Do SVMs outperform softmax?_**
- softmax, unlike one-vs-rest SVMs, introduces competition between classes when scoring a RoI
- therefore, "one-shot" fine-tuning is sufficient compared to previous multi-stage training approaches

**_Are more proposals always better?_**
- classifying sparse proposals is a type of cascade in which the proposal mechanism first rejects a vast number of candidates leaving the classifier with a small set to evaluate
- this cascade improves detection accuracy when applied to DPM detections




