### [Fully Convolutional Networks for Semantic Segmentation](https://arxiv.org/pdf/1411.4038)

**Introduction**

<img src="https://velog.velcdn.com/images/heayounchoi/post/85a4941e-33a5-4a48-88fb-dfa272bbf3be/image.png">

- first work to train FCNs end-to-end (1) for pixelwise prediction and (2) from supervised pre-training
- Fully convolutional versions of existing networks predict dense outputs from arbitrary-sized inputs.
- In-network upsampling layers enable pixelwise prediction and learning in nets with subsampled pooling.
- Patchwise training is common, but lacks the efficiency of fully convolutional training.
> patchwise training: 전체 이미지 대신 이미지의 작은 부분(패치)을 사용하여 모델을 학습시키는 방식. 전체 이미지의 맥락을 고려하지 않고 개별 패치만을 사용하여 학습하기 때문에, 패치 간의 관계나 전체 이미지의 구조적 특성을 놓칠 수 있고, 추론 단계에서 전체 이미지를 처리하기 위해 여러 패치로 나누어 예측을 수행한 후, 이를 다시 합쳐야하는 경우가 많음.
- does not make use of pre- and post-processing complications, including superpixels, proposals, or post-hoc refinement by random fields or local classifiers
> superpixels: 이미지를 보다 의미 있는 덩어리로 그룹화하여 처리하는 기술
> post-hoc refinement: 이미 생성된 결과를 개선하기 위해 추가 처리를 적용하는 것
> random fields: 통계적 또는 확률적 모델로, 이미지, 시간 또는 공간 데이터와 같이 여러 변수 간의 상호작용을 모델링하는 데 사용됨.
> CRF(Conditional Random Fields): 주로 시퀀스 데이터를 처리하는 데 사용되며, 이미지 처리에서는 픽셀 또는 슈퍼픽셀 간의 관계를 모델링하는 데 유용함
> local classifiers: 입력 데이터의 일부 지역적 특성에 기반하여 분류 결정을 내리는 모델
- this model transfers recent success in classification to dense prediction by reinterpreting classification nets as fully convolutional and fine-tuning from their learned representations
- Semantic segmentation faces an inherent tension between semantics and location: global information resolves what while local information resolves where.
- Deep feature hierarchies jointly encode location and semantics in a local-to-global pyramid.
- defines a novel "skip" architecture to combine deep, coarse, semantic information and shallow, fine, appearance information
---

**Fully convolutional networks**

**_Adapting classifiers for dense prediction_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/1bf490c2-53b8-42d7-a820-8149d6806166/image.png">

**_Shift-and-stitch is filter rarefaction_**
- Input shifting and output interlacing is a trick that yields dense predictions from coarse outputs without interpolation, introduced by OverFeat.
- If the outputs are downsampled by a factor of f, the input is shifted (by left and top padding) x pixels to the right and y pixels down, once for every value of (x, y).
- These f^2 inputs are each run through the convnet, and the outputs are interlaced so that the predictions correspond to the pixels at the centers of their receptive fields.
- Changing only the filters and layer strides of a convnet can produce the same output as this shift-and-stitch trick.

**_Upsampling is backwards strided convolution_**
- Another way to connect coarse outputs to dense pixels is interpolation.
- In a sense, upsampling with factor f is convolution with a fractional input stride of 1/f.
- So long as f is intergral, a natural way to upsample is therefore backwards convolution (sometimes called deconvolution) with an output stride of f.
- Such an operation is trivial to implement, since it simply reverses the forward and backward passes of convolution.
- in-network upsampling is fast and effective for learning dense prediction

**_Patchwise training is loss sampling_**
- Sampling in patchwise training can correct class imbalance and mitigate the spatial correlation of dense patches.
- In fully convolutional training, class balance can also be achieved by weighting the loss, and loss sampling can be used to address spatial correlation.
- Whole image training is effective and efficient.
---

**Segmnetation Architecture**
- PASCAL VOC 2011 segmentation challenge
- trained with a per-pixel multinomial logistic loss and validated with the standard metric of mean pixel intersection over union, with the mean taken over all classes, including background.
- The training ignores pixels that are masked out (as ambiguous or difficult) in the ground truth.

**_From classifier to dense FCN_**
- begin by convolutionalizing proven classification architectures
- append a 1x1 convolution with channel dimension 21 to predict scores for each of the PASCAL classes (including background) at each of the coarse output locations, followed by a deconvolution layer to bilinearly upsample the coarse outputs to pixel-dense outputs

<img src="https://velog.velcdn.com/images/heayounchoi/post/8bb210a6-d0cb-46c8-a273-e4a468444bac/image.png">

- Despite similar classification accuracy, GoogLeNet version did not match the segmentation result.

**_Combining what and where_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/6d7e3c60-6c8e-40d4-b659-c25a09d57efd/image.png">

<img src="https://velog.velcdn.com/images/heayounchoi/post/5e0659c4-08b0-4d29-b674-715385100a0b/image.png">

- learning the skip net improves performance on the validiation set by 3.0 mean IU to 62.4
- IU metric emphasizes large-scale correctness

_Refinement by other means_
- Decreasing the stride of pooling layers is the most straightforward way to obtain finer predictions.
- However, doing so is prolematic for VGG16-based net.
- Setting the pool5 lyaer to have stride 1 requires the convolutionalized fc6 to have a kernel size of 14x14 in order to maintain its receptive field size.
- In addition to their computational cost, also had difficulty learning such large filters.
- Another way to obtain finer predictions is to use the shift-and-stitch trick.

**_Experimental framework_**

_Optimization_

_Fine-tuning_
- fine-tune all layers by backpropagation through the whole net
- Fine-tuning the output classifier alone yields only 70% of the full fine-tuning performance.
- Fine-tuning takes three days on a single GPU for the coarse FCN-32s version, and about one day each to upgrade to the FCN-16s and FCN-8s versions.

_Patch Sampling_
- sampling does not have a significant effect on convergence rate compared to whole image training, but takes significantly more time due to the larger number of images that need to be considered per batch

_Class Balancing_
X

_Dense Prediction_
- Final layer deconvolutional filters are fixed to bilinear interpolation, while intermediate upsampling layers are initialized to bilinear upsampling, and then learned.

_Augmentation_
- no improvement

_More Training Data_
- improves validatoin score by 3.4 points
---

**Results**

_Metrics_

<img src="https://velog.velcdn.com/images/heayounchoi/post/61ac347c-831b-4ef6-98e8-019c36c7257f/image.png">

_PASCAL VOC_

<img src="https://velog.velcdn.com/images/heayounchoi/post/f7d530e0-a3a5-49f9-a85a-c9cb70e14da6/image.png">

_SIFT Flow_
- is a dataset of 2,688 images with pixel labels for 33 semantic categories ("bridge", "mountain", "sun"), as well as three geometric categories ("horizontal", "vertical", and "sky")
- learn a two-headed version of FCN-16s with semantic and geometric prediction layers and losses
- learning and inference as fast as each independent model by itself
---

**A. Upper Bounds on IU**
- Pixel-perfect prediction is not necessary to achieve mean IU well above SOTA, and, conversely, mean IU is not a good measure of the fine-scale accuracy.
