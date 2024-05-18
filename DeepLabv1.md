### [Semantic Image Segmentation with Deep Convolutional Nets and Fully Connected CRFs](https://arxiv.org/pdf/1412.7062)

**Abstract**
- responses at the final layer of DCNNs are not sufficiently localized for accurate object segmentation
- This is due to the very invariance properties that make DCNNs good for high level tasks like image classification and object detection.
- This poor localization property of deep networks can be overcome by combining the responses at the final DCNN layer with a fully connected Conditional Random Field (CRF).
- 8 fps on a modern GPU
---

**Introduction**
- DCNNs trained in an end-to-end manner deliver strikingly better results than systems relying on carefully engineered representations, such as SIFT or HOG features.
- This success can be partially attributed to the built-in invariance of DCNNs to local image transformations, which underpins their ability to learn hierarchical abstractions of data.
- While this invariance is clearly desirable for high-level vision tasks, it can hamper low-level tasks, such as pose estimation and semantic segmentation - where we want precise localization, rather than abstraction of spatial details.
- There are two technical hurdles in the application of DCNNs to image labeling tasks: signal downsampling, and spatial 'insensitivity' (invariance).
- The first problem relates to the reduction of signal resolution incurred by the repeated combination of max-pooling and downsampling performed at every layer of standard DCNNs.
- Instead, we employ the 'atrous' (with holes) algorithm originally developed for efficiently computing the undecimated discrete wavelet transform.
- This allows efficient dense computation of DCNN responses.
- The second problem relates to the fact that obtaining object-centric decisions from a classifier requires invariance to spatial transformations, inherently limiting the spatial accuracy of the DCNN model.
- We boost our model's ability to capture fine details by employing a fully-connected Conditional Random Field (CRF).
- Conditional Random Fields have been broadly used in semantic segmentation to combine class scores computed by multi-way classifiers with the low-level information captured by the local interactions of pixels and edges or superpixels.
**- Even though works of increased sophistication have been proposed to model the hierarchical dependency and/or high-order dependencies of segments, we use the fully connected pairwise CRF for its efficient computation, and ability to capture fine edge details while also catering for long range dependencies.**
- The three main advantages of our "DeepLab" system are (1) speed: by virtue of the 'atrous' algorithm, our dense DCNN operates at 8 fps, while Mean Field Inference for the fully-connected CRF requires 0.5 second, (2) accuracy, (3) simplicity: our system is composed of a cascade of two fairly well-established modules, DCNNs and CRFs.
---

**Related Work**
- DeepLab works directly on the pixel representation.
- This is in contrast to the two-stage approaches that are now most common in semantic segmentaion with DCNNs: such techniques make the system commit to potential errors of the front-end segmentation system.
- The main difference between DeepLab and other state-of-the-art models is the combination of pixel-level CRFs and DCNN-based 'unary terms'.
- This approach treats every pixel as a CRF node, exploits long-range dependencies, and uses CRF inference to directly optimize a DCNN-driven cost function.
---

**Convolutional Neural Networks for Dense Image Labeling**
- VGG-16 based

**_Efficient Dense Sliding Window Feature Extraction with the Hole Algorithm_**
- Dense spatial score evaluation is instrumental in the success of our dense CNN feature extractor.
- convert the fully-connected layers of VGG-16 into convolutional ones and run the network in a convolutional fashion on the image at its original resolution
- skip subsampling after the las two max-pooling layers in the network and modify the convolutional filters in the layers that follow them by introducing zeros to increase their length (2x in the last three convolutional layers and 4x in the first fully connected layer)
- We can implemtent this more efficiently by keeping the filters intact and instead sparsely sample the feature maps on which they are applied on using an input stride of 2 or 4 pixels, respectively.

<img src="https://velog.velcdn.com/images/heayounchoi/post/89cc3653-09aa-456c-af2d-f492a8b35db5/image.png">

- Loss function is the sum of cross-entropy terms for each spatial position in the CNN output map (subsampled by 8 compared to the original image).
- All positions and labels are equally weighted in the overall loss function.
- During testing, we need class score maps at the original image resolution.
- The class score maps (corresponding to log-probabilities) are quite smooth, which allows us to use simple bilinear interpolation to increase their resolution by a factor of 8 at a negligible computational cost.
- FCN produces very coarse scores (subsampled by a factor of 32) at the CNN output
- This forced them to use learned upsampling layers, significantly increasing the complexity and training time of their system.

<img src="https://velog.velcdn.com/images/heayounchoi/post/2fa139b8-be4f-4b93-8a7e-0b55d0bd3d84/image.png">

**_Controlling the Receptive Field Size and Accelerating Dense Computation with Convolutional Nets_**
- spatial subsampling으로 first FC layer spatial size 7x7 -> 4x4 (or 3x3)
- this has reduced the receptive field of the network down to 128x128 (with zero-padding) or 308x308 (in convolutional mode) and has reduced computation time for the first FC layer by 2-3 times (originally 224x224 with zero-padding and 404x404 pixels in convolutional mode)
- given a 306x306 input image, it produces 39x39 dense raw feature scores at the top of the network at a rate of about 8 fps during testing
- fc layer channels: 4096 -> 1024
---

**Detailed Boundary Recovery: Fully-Connected Conditional Random Fields and Multi-Scale Prediction**

**_Deep Convolutional Networks and the Localization Challenge_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/70a030a6-562a-41a3-af2d-d0db814c602e/image.png">

- DCNN score maps can reliably predict the presence and rough position of objects in an image but are less well suited for pin-pointing their exact outline.
- There is a natural trade-off between classification accuracy and localization accuracy with convolutional networks: Deeper models with multiple max-pooling layers have proven most successful in classification tasks, however their increased invariance and large receptive fields make the problem of inferring position from the scores at their top output levels more challenging.
- Recent work has pursued two directions to address this localization challenge.
- The first approach is to harness information from multiple layers in the conv network in order to better estimate the object boundaries.
- The second approach is to employ a super-pixel representation, essentially delegating the localization task to a low-level segmentation method.
- We pursue a novel alternative direction based on coupling the recognition capacity of DCNNs and the fine-grained localization accuracy of fully connected CRFs.
- Traditionally, CRFs have been employed to smooth noisy segmentation maps.
- using short-range CRFs can be detrimental, as our goal should be to recover detailed local structure rather than further smooth it
- using contrast-sensitive potentials in conjunction to local-range CRFs can potentially improve localization but still miss thin-structures and typically requires solving an expensive discrete optimization problem

<img src="https://velog.velcdn.com/images/heayounchoi/post/ddfd0d53-fa22-4f27-a3da-52c316738a40/image.png">

**_Multi-Scale Prediction_**
- to increase the boundary localization accuracy
- attach to the input image and the output of each of the first four max pooling layers a two-layer MLP (first layer: 128 3x3 conv filters, second layer: 128 1x1 conv filters) whose feature map is concatenated to the main network's last layer feature map
- The aggregate feature map fed into the softmax layer is thus enhanced by 5*128 = 640 channels.
- introducing these extra direct connections from fine-resolution layers improves localization performance, yet the effect is not as dramatic as the one obtained with the fully-connected CRF
---

**Experimental Evaluation**

_Dataset_
- PASCAL VOC 2012 segmentation benchmark

_Training_
- piecewise training, decoupling the DCNN and CRF training stages, assuming the unary terms provided by the DCNN are fixed during CRF training
- cross-validate parameters for fully connected CRF model

_Evaluation on Validation set_

<img src="https://velog.velcdn.com/images/heayounchoi/post/2b696917-b402-4f55-b618-9495c58b888f/image.png">

_Multi-Scale features_

<img src="https://velog.velcdn.com/images/heayounchoi/post/9629ef08-edd9-47f9-9653-bf7db3ad7b69/image.png">

_Field of View_

<img src="https://velog.velcdn.com/images/heayounchoi/post/d70987fb-8bd0-4f2b-aeab-ab523dd8dce7/image.png">

_Mean Pixel IOU along Object Boundaries_

<img src="https://velog.velcdn.com/images/heayounchoi/post/f37f93ef-6d9b-4187-8ebc-704ac2c19a63/image.png">

_Comparison with State-of-art_

<img src="https://velog.velcdn.com/images/heayounchoi/post/a52212d0-cb70-4093-be16-3b6543d86b4d/image.png">

_Reproducibility_

_Test set results_

---

**Discussion**
