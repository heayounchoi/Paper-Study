### [Panoptic Feature Pyramid Networks](https://arxiv.org/pdf/1901.02446)

**Abstract**
- In this work, we aim to unify these methods at the architectural level, designing a single network for both tasks.
- Our approach is to endow Mask R-CNN with a semantic segmentation branch using a shared FPN backbone.
---

**Introduction**
- We show a simple, flexible, and effective architecture that can match accuracy for both tasks using a single network that simultaneously generates region-based outputs (for instance segmentation) and dense-pixel outputs (for semantic segmentation).

<img src="https://velog.velcdn.com/images/heayounchoi/post/025d399d-e6f0-4fe0-9f26-976a8d3fec93/image.png">

- Using a single FPN for solving both tasks simultaneously yields accuracy equivalent to training two separate FPNs, with roughly half the compute.
- With the same compute, a joing network for the two tasks outperforms two independent networks by a healthy margin.

<img src="https://velog.velcdn.com/images/heayounchoi/post/69c6b127-1828-4bb5-a52c-c52bde35edc1/image.png">

---

**Related Work**

_Panoptic segmenation_
- Every competitive entry in the panoptic challenges used separate networks for instance and semantic segmentation, with no shared computation.

_Instance segmentation_
- region-based approaches
- start with pixel-wise semantic segmentation and then perform grouping to extract instances (use separate networks to predict the instance level information)
- position-sensitive pixel labeling to encode instance information fully convolutionally
- region-based approaches remain dominant on detection leaderboards

_Semantic segmentation_
- FCNs
- atrous convolution (dilated convolution)
- such an approach can substantially increase compute and memory, limiting the type of backbone network that can be used
- As an alternative to dilation, an encoder-decoder or 'U-Net' architecture can be used to increase feature resolution.
- Encoder-decoders progressively upsample and combine high-level features from a feedforward network with features from lower-levels, ultimately generating semantically meaningful, high-resolution features.
- In our work we adopt an encoder-decoder framework, namely FPN.
- In contrast to 'symmetric' decoders, FPN uses a lightweight decoder.
- FPN was designed for instance segmentation, and it serves as the default backbone for Mask R-CNN.
- We show that without changes, FPN can also be highly effective for semantic segmentation.

_Multi-task learning_
- In general, using a single network to solve multiple diverse tasks degrades performance, but various strategies can mitigate this.
- Our work studies the benefits of multi-task training for stuff and thing segmentation.
---

**Panoptic Feature Pyramid Network**

**_Model Architecture_**

_Feature Pyramid Network_

_Instance segmentation branch_
- To output instance segmentations, we use Mask R-CNN, which extends Faster R-CNN by adding an FCN branch to predict a binary segmentation mask for each candidate region.

_Panoptic FPN_
- To achieve accurate predictions, the features used for this task should: (1) be of suitably high resolution to capture fine structures, (2) encode sufficiently rich semantics to accurately predict class labels, and (3) capture multi-scale information to predict stuff regions at multiple resolutions.
- Although FPN was designed for object detection, these requiremetns - high-resolution, rich, multi-scale features - identify exactly the characteristics of FPN.

_Semantic segmentation branch_

<img src="https://velog.velcdn.com/images/heayounchoi/post/e99b6e72-0a33-4152-8837-42636500c7ad/image.png">

- merge the information from all levels of the FPN pyramid into a single output
- each upsampling stage consists of 3x3 convolution, group norm, ReLU, and 2x bilinear upsampling
- resulting feature maps at the same scale are then element-wise summed
- A final 1x1 convolution, 4x bilinear upsampling, and softmax are used to generate the per-pixel class labels at the original image resolution.
- In addition to stuff classes, this branch also outputs a special 'other' class for all pixels belonging to objects (to avoid predicting stuff classes for such pixels).

_Implementation details_

---

**_Inference and Training_**

_Panoptic inference_
- As the instance and semantic segmentation outputs from Panoptic FPN may overlap; we apply the simple post-processing to resolve all overlaps.
- This post-processing is similar in spirit to nms and operates by: (1) resolving overlaps between different instances based on their confidence scores, (2) resolving overlaps between instance and semantic segmentation outputs in favor of instances, and (3) removing any stuff regions labeled 'other' or under a given area threshold.

_Joint training_
- During training the instance segmentation branch has three losses: classification loss, bounding-box loss, and mask loss.
- The total instance segmentation loss is the sum of these losses, where classification loss and bbox loss are normalized by the number of sampled RoIs and mask loss is normalized by the number of foreground RoIs.
- The semantic segmentation loss is computed as a per-pixel cross entropy loss between the predicted and the ground-truth labels, normalized by the number of labeled image pixels.
- re-weight between the total instance segmentation loss and the semantic segmentation loss

<img src="https://velog.velcdn.com/images/heayounchoi/post/6b216c2d-c142-4f62-8c21-d37182f93835/image.png">

---

**_Analysis_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/6bf69145-76f5-4703-b107-fba26388c02d/image.png">

<img src="https://velog.velcdn.com/images/heayounchoi/post/d4bf69dd-48d3-403a-82f4-235595376279/image.png">

---

**Experiments**

<img src="https://velog.velcdn.com/images/heayounchoi/post/ad8f8036-ec99-4376-b0dd-ffd90be14917/image.png">

**_Experimental Setup_**

_COCO_
- The COCO dataset was developed with a focus on instance segmentation, but more recently stuff annotations were added.

_Cityscapes_

_Single-task metrics_

_Panoptic segmentation metrics_

_COCO training_

_Cityscapes training_

**_FPN for Semantic Segmentation_**

_Cityscapes_
- adding dilation into FPN could potentially yield further improvement

<img src="https://velog.velcdn.com/images/heayounchoi/post/05d01f28-bb1e-4ba0-a00d-5d78db1ac388/image.png">

_COCO_

<img src="https://velog.velcdn.com/images/heayounchoi/post/40f44a28-a879-4f86-b3b4-48ca69d58929/image.png">

_Ablations_

<img src="https://velog.velcdn.com/images/heayounchoi/post/abbc822d-ba71-4f1e-b51a-3ba4f34226a6/image.png">

**_Multi-Task Training_**

---

**_Panoptic FPN_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/af445854-55d3-490e-924f-7c4fc536bf18/image.png">

_Main results_

_Ablations_

_Comparisons_

---

**Conclusion**
