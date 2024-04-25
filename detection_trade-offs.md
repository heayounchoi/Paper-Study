### [Speed/accuracy trade-offs for modern convolutional object detectors](https://arxiv.org/pdf/1611.10012.pdf)

Abstract
- speed/memory/accuracy balance
- meta architectures: Faster R-CNN / R-FCN / SSD
- COCO detection task

1. Introduction
- fewer proposals for Faster R-CNN can speed up significantly without a big loss in accuracy, making it competitive with SSD and RFCN
- SSDs performance is less sensitive to the quality of the feature extractor than Faster R-CNN and R-FCN
- gains in accuracy are only possible by sacrificing speed

2. Meta-architectures
- R-CNN method took the straightforward approach of cropping externally computed box proposals out of an input image and running a neural net classifier on these crops.
- This approach can be expensive because many crops are necessary, leading to significant duplicated computation from overlapping crops.
- Fast R-CNN alleviated this problem by pushing the entire image once through a feature extractor then cropping from an intermediate layer so that crops share the computation load of feature extraction.
- recent works have shown that it is possible to generate box proposals using nerual networks as well (anchors methodology)

2.1 Meta-architectures
- SSD (Single Shot Multibox Detector) / Faster R-CNN / R-FCN (Region-based Fully Convolutional Networks)

2.1.1 Single Shot Detector (SSD)
- this paper uses the term SSD to refer broadly to architectures that use a single feed-forward convolutional network to directly predict classes and anchor offsets without requiring a second stage per-proposal classification operation

2.1.2 Faster R-CNN
- detection happens in two stages
- In the first stage, called the region proposal network (RPN), images are processed by a feature extractor, and features at some selected intermediate level are used to predict class-agnostic box proposals.
- In the second-stage, these box proposals are used to crop featuers from the same intermediate feature map which are subsequently fed to the remainder of the feature extractor in order to predict a class and class-specific box refinement for each proposal.
- there is part of the computation that must be run once per region, and thus the running time depends on the number of regions proposed by the RPN

2.2 R-FCN
- like Faster R-CNN, but instead of cropping features from the same layer where region proposals are predicted, crops are taken from the last layer of features prior to prediction
- This approach of pushing cropping to the last layer minimizes the amount of per-region computation that must be done.

3. Experimental setup

3.1 Architectural configuration
3.1.1 Feature extractors
- The choice of feature extractor is crucial as the number of parameters and types of layers directly affect memory, speed, and performance of the detector.
- VGG-16, Resnet-101, Inception v2, Inception v3, Inception Resnet (v2), MobileNet

3.1.2 Number of proposals
- 10 ~ 300 proposals comparison

3.1.3 Output stride settings for Resnet and Inception Resnet.
- using stride 8 instead of 16 improves the mAP by a factor of 5%, but increases running time by a factor of 63%

3.2 Loss function configuration
3.2.1 Matching
- Determining classification and regression targets for each anchor requires matching anchors to groundtruth instances.
- Common approaches include greedy bipartite matching (e.g., based on Jaccard overlap) or many-to-one matching strategies. (referred as Bipartite or Argmax)
- After matching, there is typically a sampling procedure designed to bring the number of positive anchors and negative anchors to some desired ratio.

3.2.2 Box encoding
3.2.3 Location loss
- Smooth L1

3.3 Input size configuration
- In Faster R-CNN and R-FCN, models are trained on images scaled to M pixels on the shorter edge whereas in SSD, images are always resized to a fixed shape M x M.
- explore evaluating each model on downscaled images as a way to trade accuracy for speed

3.4 Training and hyperparameter tuning
3.5 Benchmarking procedure
3.6 Model Details
3.6.1 Faster R-CNN
3.6.2 R-FCN
3.6.3 SSD

4. Results
4.1 Analyses
4.1.1 Accuracy vs time
- Generally, R-FCN and SSD models are faster on average while Faster R-CNN tends to lead to slower but more accurate models.
- Faster R-CNN models can be just as fast if the number of regions proposed are limited.
- better accuracy can be attained within the family of detectors by sacrificing speed

4.1.2 Critical points on the optimality frontier
- SSD models with Inception v2 and Mobilenet feature extractors are most accurate of the fastest models
- if we ignore postprocessing costs, Mobilenet seems to be roughly twice as fast as Inception v2 while being slightly worse in accuracy
- R-FCN models using Residual Network feature extractors strikes the best balance between speed and accuracy among the model configs
- Faster R-CNN w/Resnet models can attain similar speeds if we limit the number of proposals to 50
- Faster R-CNN with dense output Inception Resnet models attain the best possible accuracy

4.1.3 The effect of the feature extractor
- correlation between classification and detection performance is only significant for Faster R-CNN and R-FCN while the performance of SSD appears to be less reliant on its feature extractor's classification accuracy

4.1.4 The effect of object size
- even though SSD models typically have very poor performance on small objects, they are competitive with Faster RCNN and R-FCN on large objects

4.1.5 The effect of image size
- decreasing resolution by a factor of two in both dimensions consistently lowers accuracy (by 15.88% on average) but also reduces inference time by a relative factor of 27.4% on average
- high resolution inputs allow for small objects to be resolved

4.1.6 The effect of the number of proposals
- for Faster R-CNN with Inception Resnet, we can obtain 96% of the accuracy of using 300 proposals while reducing running time by a factor of 3
- R-FCN은 box classifier가 Faster R-CNN 처럼 expensive 하지 않기 때문에 큰 차이 없음

4.1.7 FLOPs analysis
- For denser block models such as Resnet 101, FLOPs/GPU time is typically greater than 1, perhaps due to efficiency in caching.
- For Inception and Mobilenet models, this ratio is typically less than 1 -- conjectured that this could be that factorization reduces FLOPs, but adds more overhead in memory I/O or potentially that current GPU instructions (cuDNN) are more optimized for dense convolution.

4.1.8 Memory analysis
- high correlation with running time with larger and more powerful feature extractors requiring much more memory

4.1.9 Good localization at .75 IOU means good localization at all IOU thresholds
- detectors that have poor performance at the higher IOU thresholds always also show poor performance at the lower IOU thresholds

4.2 State-of-the-art detection on COCO
- encouraging for diversity helped against a hand selected ensemble
- ensembling and multicrop were responsible for almost 7 points of improvement over a single model

4.3 Example detections

5. Conclusion
