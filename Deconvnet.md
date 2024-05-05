### [Learning Deconvolution Network for Semantic Segmentation](https://arxiv.org/pdf/1505.04366)

**Abstract**
- The deconvolution network is composed of deconvolution and unpooling layers, which identify pixel-wise class labels and predict segmentation masks.
- apply the trained network to each proposal in an input image, and construct the final semantic segmentation map by combining the results from all proposals in a simple manner
- The proposed algorithm mitigates the limitations of the existing methods based on fully convolutional networks by integrating deep convolution network and proposal-wise prediction; this segmentation method typically identifies detailed structures and handles objects in multiple scales naturally.
---

**Introduction**
- Semantic segmentation based on FCNs have a couple of critical limitations.
- First, the network can handle only a single scale semantics within image due to the fixed-size receptive field.
- Therefore, the object that is substantially larger or smaller than the receptive field may be fragmented or mislabeled.
- In other words, label prediction is done with only local information for large objects and the pixels that belong to the same object may have inconsistent labels.

<img src="https://velog.velcdn.com/images/heayounchoi/post/412a97b9-63f5-4ef4-b883-28eaee0a21d9/image.png">

- Also, small objects are often ignored and classified as background.
- Although FCN attempts to sidestep this limitation using skip architecture, this is not a fundamental solution and performance gain is not significant.
- Second, the detailed structures of an object are often lost or smoothed because the label map, input to the deconvolution layer, is too coarse and deconvolution procedure is overly simple.
- Note that, in the original FCN, the label map is only 16x16 in size and is deconvolved to generate segmentation result in the original input size through bilinear interpolation.
- main contributions of this paper:
- multi-layer deconvolution network, which is composed of deconvolution, unpooling, and ReLU layers
- instance-wise segmentations using object proposals
---

**Related Work**
- multi-scale superpixels classification
- region proposal approach
- FCN
- FCN + CRF
- weakly supervised setting techniques
- deconvolution is also used to visualize activated features to understand the behavior of a trained CNN model
---

**System Architecture**

**_Architecture_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/8abbacb9-8320-45c4-bf09-d7bcfa24e103/image.png">

- The final output of the network is a probability map in the same size to input image, indicating probability of each pixel that belongs to one of the predefined classes.
- VGG 16-layer net for convolutional part with its last classification layer removed
- conv network: 13 conv layers altogether, rectification and pooling operations are sometimes performed between convolutions, and 2 fully connected layers are augmented at the end to impose class-specific projection
- deconv network: mirrored version of conv network

**_Deconvolution Network for Segmentation_**

_Unpooling_
- Pooling in convolution network is designed to filter noisy activations in a lower layer by abstracting activations in a receptive field with a single representative value.
- Although it helps classification by retaining only robust activations in upper layers, spatial information within a receptive field is lost during pooling, which may be critical for precise localization that is required for semantic segmentation.

<img src="https://velog.velcdn.com/images/heayounchoi/post/4b5f4c19-2800-4ebc-84fd-2e8779970639/image.png">

- To resolve such issue, unpooling layers are employed in deconvolution network, which perform the reverse operation of pooling and reconstruct the original size of activations.
- It records the locations of maximum activations selected during pooling operation in switch variables, which are employed to place each activation back to its original pooled location.

_Deconvolution_
- The output of an unpooling layer is an enlarged, yet sparse activation map. 
- The deconvolution layers densify the sparse activations obtained by unpooling through convolution-like operations with multiple learned filters.
- However, contrary to convolution layers, which connect multiple input activations within a filter window to a single activation, deconvolutional layers associate a single input activation with multiple outputs.

**_Analysis of Deconvolution Network_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/b007f817-41ec-4a4a-9539-a87a80bcbf04/image.png">

- We can observe that coarse-to-fine object structures are reconstructed through the propagation in the deconvolutional layers; lower layers tend to capture overall coarse configuration of an object (e.g. location, shape and region), while more complex patterns are discovered in higher layers.
- Unpooling captures example-specific structures by tracing the original locations with strong activations back to image space. As a result, it effectively reconstructs the detailed structure of an object in finer resolutions.
- Learned filters in deconvolution layers tend to capture class-specific shapes. Through deconvolutions, the activations closely related to the target classes are amplified while noisy activations from other regions are suppressed effectively.

<img src="https://velog.velcdn.com/images/heayounchoi/post/66b1d6c9-ca07-481c-886b-bd2c09abe051/image.png">

**_System Overview_**
- Algorithm of the model poses semantic segmentation as instance-wise segmentation problem.
- That is, the network takes a sub-image potentially containing objects-which we refer to as instance(s) afterwards-as an input and produces pixel-wise class prediction as an output.
- Semantic segmentation on a whole image is obtained by applying the network to each candidate proposals extracted from the image and aggregating outputs of all proposals to the original image space.
- Instance-wise segmentation has a few advantages over image-level prediction.
- It handles objects in various scales effectively and identifies fine details of objects while the approaches with fixed-size receptive fields have troubles with these issues.
- Also, it alleviates training complexity by reducing search space for prediction and reduces memory requirement for training.
---

**Training**
- very deep network + limited number of examples

**_Batch Normalization_**
- a batch normalization layer added to the output of every conv and deconv layer

**_Two-stage Training_**
- train the network with easy examples first and fine-tune the trained network with more challenging examples later
- To construct training examples for the first stage training, we crop object instances using ground-truth annotations so that an object is centered at the cropped bounding box.
- By limiting the variations in object location and size, we reduce search space for semantic segmentation significantly and train the network with much less training examples successfully.
- In the second stage, we utilize object proposals to construct more challenging examples.
- Specifically, candidate proposals sufficiently overlapped with ground-truth segmentations are selected for training.
- Using the proposals to construct training data makes the network more robust to the misalignment of proposals in testing, but makes training more challenging since the location and scale of an object may be significantly different across training examples.
---

**Inference**
- The proposed network is trained to perform semantic segmentation for individual instances.
- Given an input image, we first generate a sufficient number of candidate proposals, and apply the trained network to obtain semantic segmentation maps of individual proposals.
- Then we aggregate the outputs of all proposals to produce semantic segmentation on a whole image.

**_Aggregating Instance-wise Segmentation Maps_**
- Since some proposals may result in incorrect predictions due to misalignment to object or cluttered background, we should suppress such noises during aggregation.
- The pixel-wise maximum or average of the score maps corresponding all classes turns out to be sufficiently effective to obtain robust results.

<img src="https://velog.velcdn.com/images/heayounchoi/post/75db5d56-6aef-46d3-88ef-81913cb55e80/image.png">

- Class conditional probability maps in the original image space are obtained by applying softmax function to the aggregated maps obtained by equation 1 or 2.
- Finally, we apply the fully-connected CRF to the output maps for the final pixel-wise labeling, where unary potential are obtained from the pixel-wise class conditional probability maps.

**_Ensemble with FCN_**
- This model is appropriate to capture the fine-details of an object.
- FCN is typically good at extracting the overall shape of an object.
- Instance-wise prediction is useful for handling objects with various scales, while fully convolutional network with a coarse scale may be advantageous to capture context within image.
- Given two sets of class conditional probability maps of an input image computed independently by the proposed method and FCN, compute the mean of both output maps and apply the CRF to obtain the final semantic segmentation.
---

**Experiments**

**_Implementation Details_**

_Network Configuration_

<img src="https://velog.velcdn.com/images/heayounchoi/post/5997db7b-41f1-447c-921d-03bc4d39d2a1/image.png">

- network contains approximately 252M parameters in total

_Dataset_
- PASCAL VOC 2012 segmentation dataset

_Training Data Construction_
- two-stage training strategy
- first stage:
- draw a tight bounding box corresponding to each annotated object in training images, and extend the box 1.2 times larger to include local context around the object
- then crop the window using the extended bounding box to obtain a training example
- the class label for each cropped region is provided based only on the object located at the center while all other pixels are labeled as background
- second stage:
- each training example is extracted from object proposal(Edgeboxes), where all relevant class labels are used for annotation
- same post-processing as the one used in the first stage to include context
- maintain the balance for the number of examples across classes by adding redundant examples for the classes with limited number of examples
- 250x250 image resize
- random crop 224x224 with optional horiz flipping
- number of training examples: 0.2M first stage, 2.7M second stage (sufficiently large to train deconvnet from scratch)

_Optimization_
- SGD
- initial learning rate: 0.01
- momentum: 0.9
- weight decay: 0.0005
- pretrained VGG16
- deconv net weights initialized with zero-mean Gaussians
- dropout X
- batch normalization
- reduce learning rate when valid acc does not improve
- mini batch 64

_Inference_
- egde-box to generate object proposals
- for each testing image, generate approximately 2000 object proposals, and select top 50 proposals based on their objectness scores
- equation 1 for aggregate proposal-wise prediction

**_Evaluation on Pascal VOC_**
- CRF as post-processing enhances accuracy by approximately 1% point

<img src="https://velog.velcdn.com/images/heayounchoi/post/02ed4794-d289-4a8c-9680-830ecf87b4f9/image.png">

- DeconvNet produces fine segmentations compared to FCN, and handles multi-scale objects effectively through instance-wise prediction
- FCN tends to fail in labeling too large or small objects due to its fixed-size receptive field (다시 보기)
- Deconvnet sometimes returns noisy predictions, when the proposals are misaligned or located at background regions
- inaccurate predictions from both FCN and DeconvNet are sometimes corrected by ensemble
