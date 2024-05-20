### [Panoptic-DeepLab: A Simple, Strong, and Fast Baseline for Bottom-Up Panoptic Segmentation](https://arxiv.org/pdf/1911.10194)

**Abstract**
- Panoptic-DeepLab adopts the dual-ASPP and dual-decoder structures specific to semantic, and instance segmentation, respectively.
- equipped with MobileNetV3, Panoptic-DeepLab runs nearly in real-time with a single 1025x2049 image (15.8 frames per second)
---

**Introduction**
- The goal of panoptic segmentation is to assign a unique value, encoding both semantic label and instance id, to every pixel in an image.
- It requires identifying the class and extent of each individual 'thing' in the image, and labelling all pixels that belong to each 'stuff' class.

<img src="https://velog.velcdn.com/images/heayounchoi/post/1fd67112-8238-41d1-b9cc-b1e2ba9e64fa/image.png">

- For panoptic segmentation, the top-down methods, attaching another semantic segmentation branch to Mask R-CNN, generate overlapping instance masks as well as duplicate pixel-wise semantic predictions.
- To settle the conflict, the commonly employed heuristic resolves overlapping instance masks by their predicted confidence scores, or even by the pairwise relationship between categories (e.g., ties should be always in front of person).
- Additionally, the discrepancy between semantic and instance segmentation results are sorted out by favoring the instance predictions.
- Though effective, it may be hard to implement the hand-crafted heuristics in a fast and parallel fashion.
- Another effective way is to develop advanced modules to fuse semantic and instance segmentation results.
- However, these top-down methods are usually slow in speed, resulted from the multiple sequential processes in the pipeline.
- On the other hand, bottom-up methods naturally resolve the conflict by predicting non-overlapping segments.
- bottom-up methods are simple and fast, but demonstrate inferior performance compared to top-down ones prevailing in public benchmarks
- Panoptic-DeepLab requires only three loss functions during training, and introduces extra marginal parameters as well as additional slight computation overhead when building on top of a modern semantic segmentation model.
- the instance segmentation branch involves a simple instance center regression, where the model learns to predict instance centers as well as the offset from each pixel to its corresponding center, resulting in an extremely simple grouping operation by assigning pixels to their closest predicted center.
---

**Related Works**

_Top-down_
- Most state-of-the-art methods tackle panoptic segmentation from the top-down or proposal-based perspective.
- These methods are often referred to as two-stage methods because they require an additional stage to generate proposals.

_Bottom-up_
- These works typically get the semantic segmentation prediction before detecting instances by grouping 'thing' pixels into clusters.

_Keypoint representation_
- Different from keypoint-based detection, our Panoptic-DeepLab only requires class-agnostic object center prediction.
---

**Panoptic-DeepLab**

<img src="https://velog.velcdn.com/images/heayounchoi/post/9d0a7202-7a00-4a03-aac4-c02b8aad902b/image.png">

**_Architecture_**
- Panoptic-DeepLab consists of four components: (1) an encoder backbone shared for both semantic segmentation and instance segmentation, (2) decoupled ASPP modules and (3) decoupled decoder modules specific to each task, and (4) task-specific prediction heads.

_Basic architecture_
- The encoder backbone is adapted from an ImageNet-pretrained neural network paired with atrous convolution for extracting denser feature maps in its last block.
- Our light-weight decoder module follows DeepLabV3+ with two modifications: (1) we introduce an additional low-level feature with output stride 8 to the decoder, thus the spatial resolution is gradually recovered by a factor of 2, and (2) in each upsampling stage we apply a single 5x5 depthwise-separable convolution.

_Semantic segmentation head_
- We employ the weighted bootstrapped cross entropy loss for semantic segmentation, predicting both 'thing' and 'stuff' classes.
- The loss improves over bootstrapped cross entropy loss by weighting each pixel differently.

_Class-agnostic instance segmentation head_
- Motivated by Hough Voting, we represent each object instance by its center of mass.
- For every foreground pixel (i.e., pixel whose class is a 'thing'), we further predict the offset to its corresponding mass center.
- During training, groundtruth instance centers are encoded by a 2-D Gaussian with standard deviation of 8 pixels.
- In particular, we adopt the MSE loss to minimize the distance between predicted heatmaps and 2D Gaussian-encoded groundtruth heatmaps.
- We use L1 loss for the offset prediction, which is only activated at pixels belonging to object instances.
- During inference, predicted foreground pixels (obtained by filtering out background 'stuff' regions from semantic segmentation prediction) are grouped to their closest predicted mass center, forming our class-agnostic instance segmentation results.

**_Panoptic Segmentation_**
- During inference, we use an extremely simple grouping operation to obtain instance masks, and a highly efficient majority voting algorithm to merge semantic and instance segmentation into final panoptic segmentation.

_Simple instance representation_
- To obtain the center point prediction, we first perform a keypoint-based NMS on the instance center heatmap prediction, essentially equivalent to applying max pooling on the heatmap prediction and keeping locations whose values do not change before and after max pooling.
- only locations with top-k highest confidence scores are kept.
- max-pooling with kernel size 7, threshold 0.1, and k = 200

_Simple instance grouping_

<img src="https://velog.velcdn.com/images/heayounchoi/post/847bbeb5-9f93-4389-8da0-1c2884dbbf77/image.png">

_Efficient merging_
- the semantic label of a predicted instance mask is inferred by the majority vote of the corresponding predicted semantic labels

**_Instance Segmentation_**
- Panoptic-DeepLab can also generate instance segmentation predictions as a by-product.
- We compute the class-specific confidence score for each instance mask as Score(Objectness) x Score(Class) where Score(Objectness) is unnormalized objectness score obtained from the class-agnostic center point heatmap, and Score(Class) is obtained from the average of semantic segmentation predictions within the predicted mask region.
---

**Experiments**

_Cityscapes_

_Mapillary Vistas_

_COCO_

_Experimental Setup_

<img src="https://velog.velcdn.com/images/heayounchoi/post/66d35b2d-0702-461e-9320-1f0d4beb01ac/image.png">

**_Ablation Studies_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/6c46e639-2b0f-4112-ac39-0e8a1f97032f/image.png">

_Multi-task learning_
- multi-task learning does not bring extra gain to mIoU

**_Cityscapes_**

_Val set_

_Test set_

**_Mapillary Vistas_**

_Val set_

_Test set_

**_COCO_**

_Val set_
- top-down method가 성능 아직 더 좋음

_Test-dev set_
- comparable to most top-down methods without using heavier backbone or deformable convolution

**_Runtime_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/b34c5b52-e9dc-4134-aef7-48caa6026b16/image.png">

**_Discussion_**

_Scale variation_

<img src="https://velog.velcdn.com/images/heayounchoi/post/2ec0307f-2e1b-4af1-87c2-e9a7914428d2/image.png">

- top-down methods handle scale variation to some extent by the ROIPooling or ROIAligh operations which normalize regional features to a canonical scale
- incorporating scale-aware information to feature pyramid or image pyramid may improve the performance of bottom-up methods

_PQ Thing vs PQ Stuff_
- Panoptic-DeepLab has higher PQ Stuff but lower PQ Thing when compared with other top-down approaches which better handle instances of large scale variation

_Panoptic vs instance annotations_
- COCO has two types of annotations: panoptic and instance
- former do not allow overlapping masks, while the latter allows overlaps

_End-to-end training_
- still requires come post-processing steps
---

**Conclusion**

---

**Instance and Panoptic Annotation**

<img src="https://velog.velcdn.com/images/heayounchoi/post/ff5b665e-70d7-4e68-82ec-a938f2da89ae/image.png">

- all top-down methods based on Mask R-CNN use the instance annotation when trained on COCO, while bottom-up methods including Panoptic-DeepLab use the panoptic annotation on all datasets
