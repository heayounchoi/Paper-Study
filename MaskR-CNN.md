### [Mask R-CNN](https://arxiv.org/pdf/1703.06870)

**Abstract**
- Mask R-CNN extends Faster R-CNN by adding a branch for predicting an object mask in parallel with the existing branch for bounding box recognition
- adds only a small overhead to Faster R-CNN, 5 fps
- easy to generalize to other tasks (ex. human poses)
---

**Introduction**
- instance segmentation is challenging because it requires the correct detection of all objects in an image while also precisely segmenting each instance
- combines object detection and semantic segmentation, where the goal is to classify each pixel into a fixed set of categories without differentiating object instances
- Mask R-CNN extends Faster R-CNN by adding a branch for predicting segmentation masks on each RoI, in parallel with the existing branch for classification and bounding box regression

<img src="https://velog.velcdn.com/images/heayounchoi/post/1c99bf98-6918-4a0b-af64-133fc328df10/image.png">

- The mask branch is a small FCN applied to each RoI, predicting a segmentation mask in a pixel-to-pixel manner.
- RoIPool performs coarse spatial quantization for feature extraction
- To fix the misalignment, a simple, quantization-free layer, called RoIAlign, that preserves exact spatial locations, is proposed.
- RoIAlign improves mask accuracy by relative 10% to 50%
- RoIAlign is essential to decouple mask and class prediction: we predict a binary mask for each class independently, without competition among classes, and rely on the network's RoI classification branch to predict the category.
- In constrast, FCNs usually perform per-pixel multi-class categorization, which couples segmentation and classification, and based on our experiments works poorly for instance segmentation.
- this model can run at about 200ms epr frame on a GPU, and training on COCO takes one to two days on a single 8-GPU machine
---

**Related Work**

**_R-CNN_**
- R-CNN -> Fast R-CNN -> Faster R-CNN

**_Instance Segmentation_**
- Driven by the effectiveness of R-CNN, many approaches to instance segmentation are based on segment proposals.
- earlier methods: bottom-up segments
- DeepMask: learn to propose segment candidates, which are then classified by Fast R-CNN. segmentation precedes recognition, which is slow and less accurate
- other works: predict segment proposals from bounding-box proposals, followed by classification
- Mask R-CNN: parallel prediction of masks and class labels, which is simpler and more flexible
- FCIS: fully convolutional instance segmentation. segment proposal system + object detection system. output channels simultaneously address object classes, boxes, and masks, making the system fast
- but FCIS exhibits systematic errors on overlapping instances and creates spurious edges, showing that it is challenged by the fundamental difficulties of segmenting instances
- other instance segmentation methods driven by the success of semantic segmentation: starting from per-pixel classification results (e.g., FCN outputs), these methods attempt to cut the pixels of the same category into different instances (segmentation-first strategy)
- Mask R-CNN is based on an instance-first strategy
---

**Mask R-CNN**
- mask output requires extraction of much finer spatial layout of an object

**_Faster R-CNN_**

**_Mask R-CNN_**
- in contrast to most recent systems, where classification depends on mask predictions
- multi-task loss on each sampled RoI: $$L = L_{cls} + L_{box} + L_{mask}$$
- The mask branch has a $$Km^2$$-dimensional output for each RoI, which encodes K binary masks of resolution m x m, one for each of the K classes.
- To this we apply a per-pixel sigmoid, and define $$L_{mask}$$ as the averagage binary cross-entropy loss.
- For an RoI associated with ground-truth class k, $$L_{mask}$$ is only defined on the k-th mask (other mask outputs do not contribute to the loss)
- we rely on the dedicated classification branch to predict the class label used to select the output mask, and this decouples mask and class prediction

**_Mask Representation_**
- predict an m x m mask from each RoI using an FCN

**_RoIAlign_**
- Quantizations of RoIPool introduce misalignments between the RoI and the extracted features.
- While this may not impact classification, which is robust to small translations, it has a large negative effect on predicting pixel-accurate masks.
- To address this, we propose an RoIAlign layer that removes the harsh quantization of RoIPool, properly aligning the extracted features with the input.
- avoid any quantization of the RoI boundaries or bins
- use bilinear interpolation to compute the exact values of the input features at four regularly sampled locations in each RoI bin, and aggregate the result (using max or average)

<img src="https://velog.velcdn.com/images/heayounchoi/post/b3d74737-884b-441b-b391-0c19f47114a0/image.png">

**_Network Architecture_**
- backbone: ResNet and ResNeXt, FPN
- network head: ResNet, FPN

<img src="https://velog.velcdn.com/images/heayounchoi/post/a1516348-b2a6-4a37-9d60-3e5460e7b49c/image.png">

**_Implementation Details_**

_Training_
- The mask target is the intersection between an RoI and its associated ground-truth mask.

_Inference_
- proposal -> nms -> mask branch
- The mask branch can predict K masks per RoI, but we only use the k-th mask, where ks is the predicted class by the classification branch.
--- 

**Experiments: Instance Segmentation**
- COCO

**_Main Results_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/cd10db22-dfad-4ec8-b112-bebe3c1e78b2/image.png">

<img src="https://velog.velcdn.com/images/heayounchoi/post/deb952e6-90fb-4c67-954e-b2ee4ccb0575/image.png">

**_Ablation Experiments_**

_Architecture_
- Better backbones bring expected gains: deeper networks do better, FPN outperforms C4 features, and ResNeXt improves on ResNet.
- not all frameworks automatically benefit from deeper or advanced networks

_Multinomial vs. Independent Masks_
- once the instance has been classified as a whole (by the box branch), it is sufficient to predict a binary mask without concern for the categories, which makes the model easier to train

_Class-Specific vs. Class-Agnostic Masks_
- class-agnostic masks are nearly as effective
- This further highlights the division of labor in our approach which largely decouples classification and segmentation.

_RoIAlign_
- RoIAlign improves AP by about 3 points over RoIPool, with much of the gain coming at high IoU.
- RoIAlign is insensitive to max/average pool.
- also compare RoIWarp (bilinear sampling)
- RoIWarp still quantizes the RoI
- RoIWarp performs on par with RoIPool and much worse than RoIAlign
- This highlights that propoer alignment is key.
- using large stride features is more accurate with RoIAlign

_Mask Branch_
- Fully convolution > fully connected

**_Bounding Box Detection Results_**
- RoIAlign, multi-task training increases accuracy for object detection task

**_Timing_**
- fast prototyping can be completed in less than one day when training on the train set
---

**Mask R-CNN for Human Pose Estimation**
- Our framework can easily be extended to human pose estimation.
- We model a keypoint's location as a one-hot mask, and adopt Mask R-CNN to predict K masks, one for each of K keypoint types (e.g., left shoulder, right elbow).
- This task helps demonstrate the flexibility of Mask R-CNN.

_Implementation Details_
- relatively high resolution output (compared to masks) is required for keypoint-level localization accuracy

_Main Results and Ablations_
- adding the mask branch to the box-only (Faster R-CNN) or keypoint-only versions consistently improves these tasks
- however, adding the keypoint branch reduces the box/mask AP slightly, suggesting that while keypoint detection benefits from multitask training, it does not in turn help the other tasks
- learning all three tasks jointly enables a unified system to efficiently predict all outputs simultaneously
---

**Appendix A: Experiments on Cityscapes**
- val/test 데이터 간의 domain shift가 있는 경우가 있다고 함
---

**Appendix B: Enhanced Results on COCO**
