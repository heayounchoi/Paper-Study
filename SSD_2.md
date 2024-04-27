### [SSD: Single Shot MultiBox Detector](https://arxiv.org/pdf/1512.02325)

**Abstract**
- SSD discretizes the output space of bounding boxes into a set of default boxes over different aspect ratios and scales per feature map location.
- At prediction time, the network generates scores for the presence of each object category in each default box and produces adjustments to the box to better match the object shape.
- Additionally, the network combines predictions from multiple feature maps with different resolutions to naturally handle objects of various sizes.
- SSD is simple relative to methods that require object proposals because it completely eliminates proposal generation and subsequent pixel or feature resampling stages and encapsulates all computation in a single network.
- on PASCAL VOC2007 test, SSD achieves 74.3% mAP for 300x input and 76.9% mAP for 512x input.
---

**Introduction**
- while Faster R-CNN operates at 7 fps, SSD oprates 59 fps with higher accuracy
- The fundamental improvement in speed comes from eliminating bounding box proposals and the subsequent pixel or feature resampling stage.
- Our improvements include using a small convolutional filter to predict object categories and offsets in bounding box locations, using separate predictors (filters) for different aspect ratio detections, and applying these filters to multiple feature maps from the later stages of a network in order to perform detection at multiple scales.
- using multiple layers for prediction at different scales led to 74.3% mAP, while YOLO achieved 63.4% mAP
- The core of SSD is predicting category scores and box offsets for a fixed set of default bounding boxes using smal convolutional filters applied to feature maps.
- To achieve high detection accuracy, produces predictions of different scales from feature maps of different scales, and explicitly separate predictions by aspect ratio.
---

**The Single Shot Detector (SSD)**

<img src="https://velog.velcdn.com/images/heayounchoi/post/925ea48d-d604-4cb9-9119-4f267c752fc2/image.png" width="50%">

**_Model_**
- The SSD approach is based on feed-forward convolutional network that produces a fixed-size collection of bounding boxes and scores for the presence of object class instances in those boxes, followed by a non-maximum suppression step to produce the final detections.

_Multi-scale feature maps for detection_
- convolutional feature layers are added to the end of the truncated base network
- these layers decrease in size progressively and allow predictions of detections at multiple scales
- the convolutional model for predicting detections is different for each feature layer

_Convolutional predictors for detection_

<img src="https://velog.velcdn.com/images/heayounchoi/post/74b16b04-0dba-4944-b0f0-2a8d6325ef58/image.png" width="50%">

_Default boxes and aspect ratios_

**_Training_**
- The key difference between training SSD and training a typical detector that uses region proposals, is that ground truth information needs to be assigned to specific outputs in the fixed set of detector outputs.

_Matching strategy_
- network predicts high scores for multiple overlapping default boxes rather than picking only the one with maximum overlap

_Training objective_

<img src="https://velog.velcdn.com/images/heayounchoi/post/1ed8381d-8fb9-4e23-8866-b920175ee679/image.png" width="50%">

_Choosing scales and aspect ratios for default boxes_
- To handle different object scales, some methods suggest processing the image at different sizes and combining the results afterwards.
- By utilizing feature maps from several different layers in a single network for prediction we can mimic the same effect, while also sharing parameters across all object scales.
- We both use the lower and upper feature maps for detection. (lower layers capture more fine details of the input objects and upper feature maps captures global context)
- We design the tiling of default boxes so that specific feature maps learn to be responsive to particular scales of the objects.
  
<img src="https://velog.velcdn.com/images/heayounchoi/post/75c44032-1bf7-4776-8bb8-bff105d669b7/image.png" width="50%">

- By combining predictions for all default boxes with different scales and aspect ratios from all locations of many feature maps, we have a diverse set of predictions, covering various input object sizes and shapes.
- For example, in Fig. 1, the dog is matched to a default box in the 4x feature map, but not to any default boxes in the 8x feature map. This is because those boxes have different scales and do not match the dog box, and therefore are considered as negatives during training.

_Hard negative mining_
- After the matching step, most of the default boxes are negatives, especially when the number of possible default boxes is large.
- This introduces a significant imbalance between the positive and negative training examples.
- Instead of using all the negative examples, we sort them using the highest confidence loss for each default box and pick the top ones so that the ratio between the negatives and positives is at most 3:1.

_Data augmentation_
- Sample a patch so that the minimum jaccard overlap with the objects is 0.1, 0.3, 0.5, 0.7, or 0.9.
- keep the overlapped part of the ground truth box if the center of it is in the sampled patch
- each sampled patch is resized to fixed size and is horizontally flipped with probability of 0.5 + photo-metric distortions
---

**Experimental Results**

_Base network_
- based on VGG16
- convert fc6 and fc7 to conv layers, subsample parameters from fc6 and fc7, change pool5 from 2x s2 to 3x s1, and use the a trous algorithm to fill the holes

**_PASCAL VOC2007_**
- SSD can detect various object categories with high quality
- The majority of its confident detections are correct.
- The recall is around 85-90%, and is much higher with "weak" (0.1 jaccard overlap) criteria.
- Compared to R-CNN, SSD has less localization error, indicating that SSD can localize objects better because it directly learns to regress the object shape and classify object categories instead of using two decoupled steps.
- However, SSD has more confusions with similar object categories, partly becuase it shares locations for multiple categories.
- SSD has much worse performance on smaller objects than bigger objects.
- those small objects may not even have any information at the very top layers
- increasing the input size can help improve detecting small objects

**_Model analysis_**

_Data augmentation is crucial_
- +8.8% mAP with the sampling strategy
- Fast and Faster R-CNN are likely to benefit less from this strategy because they use a feature pooling step during classification that is relatively robust to object translation by design

_More default box shapes is better_
- -0.6% if boxes with 1/3 and 3 aspect ratios removed
- -2.1% if boxes with 1/2 and 2 aspect ratios removed

_Atrous is faster_
- if we use the full VGG16, speed is about 20% slower

_Multiple output layers at different resolutions is better_
- A major contribution of SSD is using default boxes of different scales on different ouput layers.
- accuarcy decreases with fewer layers
- it hurts performance by a large margin if we use very coarse feature maps
- the reason might be that we do not have enough large boxes to cover large objects after the pruning
- when we use primarily finer resolution maps, the performance starts increasing again because even after pruning a sufficient number of parge boxes remains

**_PASCAL VOC2012_**
- Compared to YOLO, SSD is significantly more accurate, likely due to the use of convolutional default boxes from multiple feature maps and the matching strategy during training.

**_COCO_**

**_Preliminary ILSVRC results_**

**_Data Augmentation for Small Object Accuracy_**
- random crops generated by the data augmentation strategy can be thought of as a "zoom in" operation
- to implement a "zoom out" operation that creats more small training examples, we first randomly place an image on a canvas of 16x of the original image size filled with mean values before we do any random crop operation (+2-3% mAP)
- this data augmentation trick helps detecting small objects significantly

**_Inference time_**
- considering the large number of boxes generated from the method, it is essential to perform non-maximum suppression efficiently during inference
- by using a confidence threshold of 0.01, we can filter out most boxes
- we then apply nms with jaccard overlap of 0.45 per class and keep the top 200 detections per image
- about 80% of the forward time is spent on the base network
- using a faster base network could even further improve the speed
---

**Related Work**
- original R-CNN approach vs RPN approach vs YOLO-stype approach
---

**Conclusions**
- a promising future direction is to explore its use as part of a system using recurrent neural networks to detect and track objects in video simultaneously
