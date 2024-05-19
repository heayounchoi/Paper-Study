### [DeepLab: Semantic Image Segmentation with Deep Convolutional Nets, Atrous Convolution, and Fully Connected CRFs](https://arxiv.org/pdf/1606.00915)

**Abstract**
- three main contributions:
1) convolution with upsampled filters, or 'atrous convolution', a powerful tool in dense prediction tasks. Atrous convolution allows us to explicitly control the resolution at which feature responses are computed within Deep Convolutional Neural Networks. It also allows us to effectively enlarge the field of view of filters to incorporate larger context without increasing the number of parameters or the amount of computation.
2) atrous spatial pyramid pooling (ASPP) to robustly segment objects at multiple scales. ASPP probes an incoming convolutional feature layer with filters at multiple sampling rates and effective fields-of-views, thus capturing objects as well as image context at multiple scales.
3) improve the localization of object boundaries by combining methods from DCNNs and probabilistic graphical models. The commonly deployed combination of max-pooling and downsampling in DCNNs achieves invariance but has a toll on localization accuracy. We overcome this by combining the responses at the final DCNN layer with a fully connected CRF, which is shown both qualitatively and quantitatively to improve localization performance.
---

**Introduction**
- In particular we consider three challenges in the application of DCNNs to semantic image segmentation: (1) reduced feature resolutiom, (2) existence of objects at multiple scales, and (3) reduced localization accuaracy due to DCNN invariance.
- to overcome first challenge and efficiently produce denser feature maps, we remove the downsampling operator from the last few max pooling layers of DCNNs and instead upsample the filters in subsequent convolutional layers, resulting in feature maps computed at a higher sampling rate
- Filter upsampling amounts to inserting holes ('trous' in French) between nonzero filter taps.
- We use the term atrous convolution as a shorthand for convolution with upsampled filters.
- We recover full resolution feature maps by a combination of atrous convolution, which computes feature maps more densely, followed by simple bilinear interpolation of the feature responses to the original image size.
- This scheme offers a simple yet powerful alternative to using deconvolutional layers in dense prediction tasks.
- Compared to regular convolution with larger filters, atrous convolution allows us to effectively enlarge the field of view of filters without increasing the number of parameters or the amount of computation.
- second challenge: motivated by spatial pyramid pooling, we propose a computationally efficient scheme of resampling a given feature layer at multiple rates prior to convolution
- Rather than actually resampling features, we efficiently implement this mapping using multiple parallel atrous convolutional layers with different sampling rates; we call the proposed technique "atrous spatial pyramid pooling" (ASPP).
- third challenge: CRF

<img src="https://velog.velcdn.com/images/heayounchoi/post/0a8e62b3-9dbd-4c90-838a-6c4d88a4a3ee/image.png">

---

**Related Work**
1) segmentation -> classification
2) classification / segmentation
3) classification + segmentation
- Our approach treats each pixel as a CRF node receiving unary potentials by the DCNN.
- Crucially, the Gaussian CRF potentials in the fully connected CRF model can capture long-range dependencies and at the same time the model is amenable to fast mean field inference.
---

**Methods**

**_Atrous Convolution for Dense Feature Extraction and Field-of-View Enlargement_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/f5b3488f-8a2f-440d-9ea6-5970acf10244/image.png">

<img src="https://velog.velcdn.com/images/heayounchoi/post/45f475c2-0a26-46cc-a78f-4df408bbb92c/image.png">

<img src="https://velog.velcdn.com/images/heayounchoi/post/d5aeb4a7-ab0d-4a14-b008-db7d2034155f/image.png">

- kernel: vertical Gaussian derivative
- if one implatns the resulting feature map in the original image coordinates, we realize that we have obtained responses at only 1/4 of the image positions
- Instead, we can compute responses at all image positions if we convolve the full resolution image with a filter 'with holes', in which we upsample the original filter by a factor of 2, and introduces zeros in between filter values.
- Although the effective filter size increases, we only need to take into account the non-zero filter values, hence both the number of filter parameters and the number of operations per position stay constant.

<img src="https://velog.velcdn.com/images/heayounchoi/post/3e6f653c-657f-42d6-9b43-1204592f04de/image.png">

- We have adopted a hybrid approach that strikes a good efficiency/accuracy trade-off, using atrous convolution to increase by a factor of 4 the density of computed feature maps, followed by fast bilinear interpolation by an additional factor of 8 to recover feature maps at the original image resolution.
- Bilinear interpolation is sufficient in this setting because the class score maps (corresponding to log-probabilities) are quite smooth.
- Unlike the deconvolutional approach, the proposed approach converts image classification networks into dense feature extractors without requiring learning any extra parameters, leading to faster DCNN training in practice.
- Atrous convolution also allows us to arbitrarily enlarge the field-of-view of filters at any DCNN layer.
- It offers an efficient mechanism to control the field-of-view and finds the best trade-off between accurate localization (small field-of-view) and context assimilation (large field-of-view).
- two ways to perform atrous convolution:
1) implicitly upsample the filters by inserting holes (zeros), or equivalently sparsely sample the input feature maps
2) subsample the input feature map by a factor equal to the atrous convolution rate r, deinterlacing it to produce r^2 reduced resolution maps, one for each of the rxr possible shifts. This is foolowed by applying standard convolution to these intermediate feature maps and reinterlacing them to the original image resolution.

**_Multiscale Image Representations using Atrous Spatial Pyramid Pooling_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/86109bb0-8d52-4c33-a57d-870935f9cf03/image.png">

**_Structured Prediction with Fully-Connected Conditional Random Fields for Accurate Boundary Recovery_**
- Traditionally, CRFs have been employed to smooth noisy segmentation maps.
- Typically these models couple neighboring nodes, favoring same-label assignments to spatially proximal pixels.
- Qualitatively, the primary function of these short-range CRFs is to clean up the spurious predictions of weak classifiers built on top of local hand-engineered features.
- Compared to these weak classifiers, modern DCNN architectures such as the one we use in this work produce score maps and semantic label predictions which are qualitatively different.
- The score maps are typically quite smooth and produce homogeneous classification results.
- In this regime, using short-range CFTs can be detrimental, as our goal should be to recover detailed local structure rather than further smooth it.
- To overcome these limitations of short-range CRFs, we integrate into our system the fully connected CRF model.
- The first kernel of pairwise potential forces pixels with similar color and position to have similar labels, while the second kernel only considers spatial proximity when enforcing smoothness.
---

**Experimental Results**
- CNN output map is subsampled by 8 compared to the original image

**_PASCAL VOC 2012_**

_Dataset_

_Results from our conference version_

_Field of View and CRF_

<img src="https://velog.velcdn.com/images/heayounchoi/post/39fd2712-3245-4b59-ae16-adf4f41d0c9a/image.png">

_Test set evaluation_

_Improvements after conference version of this work_
(1) different learning policy during training
(2) atrous spatial pyramid pooling
(3) employment of deepter networks and multi-scale processing

_Learning rate policy_

<img src="https://velog.velcdn.com/images/heayounchoi/post/8b2cd1fc-7672-4543-9b46-33b344ba3e87/image.png">

<img src="https://velog.velcdn.com/images/heayounchoi/post/07879b3f-6137-4d2b-99f7-19d2f7644d5f/image.png">

_Atrous Spatial Pyramid Pooling_

<img src="https://velog.velcdn.com/images/heayounchoi/post/16940fe0-97f2-4999-8948-e447ad5a5e65/image.png">

<img src="https://velog.velcdn.com/images/heayounchoi/post/9ec3fe7d-72c7-4c6b-9054-e54f1a5dfae5/image.png">

_Deeper Networks and Multiscale Processing_

<img src="https://velog.velcdn.com/images/heayounchoi/post/ec4bf89e-fbc0-435a-8bb3-1bd38ac9676d/image.png">

_Qualitative results_

<img src="https://velog.velcdn.com/images/heayounchoi/post/f9379b93-57d4-47ae-be4a-068fb4eb9e01/image.png">

- removes false positives and refines object boundaries

_Test set results_

<img src="https://velog.velcdn.com/images/heayounchoi/post/e548c901-8070-442a-b457-55c73059787a/image.png">

_VGG-16 vs ResNet-101_

<img src="https://velog.velcdn.com/images/heayounchoi/post/1278530e-21e5-4c25-a858-9d500195b0b2/image.png">

**_PASCAL-Context_**

_Dataset_

_Evaluation_

_Qualitative results_

<img src="https://velog.velcdn.com/images/heayounchoi/post/2a19b615-9351-4897-8d69-36126a7b9688/image.png">

**_PASCAL-Person-Part_**

_Dataset_

_Evaluation_
- no improvement when adopting either LargeFOV or ASPP on this dataset

_Qualitative results_

<img src="https://velog.velcdn.com/images/heayounchoi/post/58ab35bc-7659-44e5-82d1-79c7db582058/image.png">

**_Cityscapes_**

_Dataset_

_Test set results of pre-release_

_Val set results_

_Current test result_

_Qualitative results_

<img src="https://velog.velcdn.com/images/heayounchoi/post/02734e52-a3fc-4caf-bfe5-0f510246e51b/image.png">

**_Failure Modes_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/7875a48b-48e3-4b7c-9013-9165c3240e85/image.png">

- model fails to capture the delicate boundaries of objects, such as bicycle and chair
- The details could not even be recovered by the CRF post processing since the unary term is not confident enough.
- We hypothesize the encoder-decoder structure may alleviate the problem by exploiting the high resolution feature maps in the decoder path.
---

**Conclusion**
