# Rich feature hierarchies for accurate object detection and semantic segmentation

**Abstract**
- two key insights of R-CNN:
    - (1) one can apply high-capacity convolutional neural networks to bottom-up region proposals in order to localize and segment objects
    - (2) when labeled training data is scarce, supervised pre-training for an auxiliary task, followed by domain-specific fine-tuning, yields a significant performance boost
- R-CNN means Regions with CNN features

---
**Introduction**
- the last decade of progress on various visual recognition tasks has been based considerably on the use of SIFT and HOG
- SIFT and HOG are blockwise orientation histograms, a representation that can be associated roughly with complex cells in V1, the first cortical area in the primate visual pathway
- but recognition occurs several stages downstream, which suggests that there might be hierarchical, multi-stage processes for computing features that are even more informative for visual recognition
- neocognitron: a biologically-inspired hierarchical and shift-invariant model for pattern recognition
- 그 다음에 CNN
- 근데 SVM 생기고 SVM이 대세였음
- ImageNet 나오고 다시 CNN이 대세(기존 CNN이랑 달라진건 ReLU랑 dropout regularization 정도)
- this paper is the first to show that a CNN can lead to dramatically higher object detection performance on PASCAL VOC as compared to systems based on simpler HOG-like features
- unlike image classification, detection requires localizing objects within an image
- one approach frames localization as a regression problem
- an alternative is to build a sliding-window detector
- but these two approaches give poor results
- in this paper, CNN localization problem is solved by operating within the "recognition using regions" paradigm, which has been successful for both object detection and semantic segmentation
- at thes time, this method generates around 2000 category-independent region proposals for the input image, extracts a fixed-length feature vector from each proposal using a CNN, and then classifies each region with category-specific linear SVMs
<img src="https://velog.velcdn.com/images/heayounchoi/post/90dbfed7-9c20-4761-b10a-da7a43a6cf28/image.png" width="50%">

- on ILSVRC2013 detection, R-CNN outperforms OverFeat, with a mAP of 31.4% versus 24.3%
- the second principle contribution of this paper is to show that supervised pre-training on a large auxiliary dataset(ILSVRC), followed by domain-specific fine-tuning on a small dataset (PASCAL), is an effective paradigm for learning high-capacity CNNs when data is scarce
- fine-tuning for detection improved mAP performance by 8 percentage points
- R-CNN is also quite efficient
- features are shared across all categories and are also two orders of magnitude lower dimensional than previously used region features
- because R-CNN operates on regions it is natural to extend it to the task of semantic segmentation
- with minor modifications, competitive results can be achieved on segmentation tasks

---
**Object detection with R-CNN**
- three moduls of R-CNN:
> 1) generates category-independent region proposals
> 2) large convolutional neural network that extracts a fixed-length feature vector from each region
> 3) set of class-specific linear SVMs

**_Module design_**
_Region proposals_
- while R-CNN is agnostic to the particular region proposal method, selective search is used to enable a controlled comparison with prior detection work

_Feature extraction_
- 4096-dimensional feature vector is extracted from each region proposal
- features are computed by forward propagating a mean-subtracted 227x227 RGB image through five conv layers and two fc layers
- regardless of the size or aspect ratio of the candidate region, all pixels are warped in a tight bounding box around it to the required size

**_Test-time detection_**
1) selective search on the test image to extract around 200 region proposals (fast mode)
2) warp proposals
3) forward propagate through the CNN to compute features
4) for each class, score each extracted feature vector using SVM trained for that class
5) apply greedy non-maximum suppression (for each class independently) that rejects a region if it has an intersection-over-union (IoU) overlap with a higher scoring selected region larger than a learned threshold

**_Training_**
_Supervised pre-training_
- pre-trained on ILSVRC2012

_Domain-specific fine-tuning_
- continue SGD training of the CNN parameters using only warped region proposals
- (N+1)-way classification layer (N: number of object classes, plus 1 for background)
- all region proposals with >= 0.5 IoU overlap are treated with a ground-truth box as positives for that box's class and the rest as negatives
- base learning rate 0.001 (1/10th of the initial pre-training rate)
- in each SGD iteration, 32 positive windows and 96 background windows are uniformly sampled to construct a mini-batch of size 128

_Object category classifiers_
- IoU overlap threshold 0.3
- one linear SVM is optimized per class
- since training data is too large to fit in memory, hard negative mining method is used

---
**Visualization, ablation, and modes of error**
**_Visualizing learned features_**
- 기존에는 deconvolution 방법을 사용했음
- deconvolution 방법은 CNN의 중간 계층에서의 활성화를 역추적하여, 네트워크가 특정 출력을 생성하는데 중요한 입력 이미지의 부분을 시각화함
- 이 논문에서는 새로운 방법을 제안함. 네트워크 내 특정 단위를 객체 탐지기처럼 사용해, 활성화 값이 최대인 영역을 시각화함으로써 해당 단위에 의해 "대변되는" 시각적 정보를 나타냄
<img src="https://velog.velcdn.com/images/heayounchoi/post/37501e5e-4505-4711-8af3-47b214582f20/image.png" width="50%">

**_Ablation studies_**
_Performance layer-by-layer, without fine-tuning_
- much of the CNN's representational power comes from its conv layers

_Performance layer-by-layer, with fine-tuning_
- improvements can be gained from learning domain-specific non-linear classifiers on top of general features

_Comparison to recent feature learning methods_
- feature learning을 사용하는 DPM ST(sketch token)와 DPM HSC(histogram of sparse codes), 그리고 standard HOG-based DPM과 비교했을때도 R-CNN variants의 성능이 훨씬 좋았음

**_Network architectures_**
- OxfordNet이 TorontoNet보다 성능은 좋지만 forward passrk 7배 더 오래 걸림

**_Detection error analysis_**
- compared with DPM, significantly more of R-CNN errors result from poor localization(IoU 낮은거), rather than confusion with background or other object classes, indicating that the CNN features are much more discriminative than HOG
- loose localization likely results from use of bottom-up region proposals and the positional invariance learned from pre-training the CNN for whole-image classification
- bounding-box regression method fixes many localization errors
- fine-tuning improves robustness for all characteristics including occlusion, trunctation, viewpoint, and part visibility

---
**The ILSVRC2013 detection dataset**
**_Dataset overview_**
- training set은 annotation도 완전하지 않고 image distribution도 val/test set과 달라서 val set을 나눠서 negative examples를 뽑음(hard negative mining)

**_Region proposals_**
- selective search fast mode
- since selective search is not scale invariant and therefore the number of regions produced depends on the image resolution, ILSVRC images are resized to a fixed with (500 pixels) before running selective search

**_Training data_**
- val1+trainN
- training data is required for three procedures in R-CNN
> 1) CNN fine-tuning
> 2) detector SVM training
> 3) bounding-box regressor training
- CNN fine-tuning: 50k SGD iterations (13 hours on NVIDIA Tesla K20)
- SVM training: all ground-truth boxes were used as positive examples
- hard negative mining was performed on a randomly selected subset of 5000 images from val1
- bounding-box regressors: trained on val1

**_Relationship to OverFeat_**
- OverFeat이 9배 빠름

---
**Appendix**

**_A. Object proposal transformations_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/121062ec-07e2-467e-bd93-b0b4b0ca3781/image.png" width="50%">

**_B. Positive vs. negative examples and softmax_**
- for fine-tuning, each object proposal is mapped to the ground-truth instance with which is has maximum IoU overlap (if any) and label it as a positive for the matched ground-truth class if the IoU is at least 0.5
- all other proposals are labeled "background" (negative)
- for training SVMs, only the ground-truth boxes are positive for their respective classes and label proposals with less than 0.3 IoU overlap with all instances of a class is negative for that class
- propossals that fall into the grey zone are ignored
- using jittered examples (set for fine-tuning) expands the number of positive examples by approximately 30x
- these positive examples does not emphasize precise localization, and the softmax classifier was trained on randomly sampled negative examples (not "hard negatives")
- therefore, even after fine-tuning, SVMs are necessary

**_C. Boudning-box regression_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/1cde8e4c-22c7-4b22-873f-088739de7d7c/image.png" width="50%">
<img src="https://velog.velcdn.com/images/heayounchoi/post/e3aa6b36-5381-40de-b1b0-abc1f2a89adb/image.png" width="50%">
<img src="https://velog.velcdn.com/images/heayounchoi/post/6e008910-62e9-4ab9-abf7-e4cebf14b425/image.png" width="50%">
