### [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/pdf/1506.02640.pdf)

**Abstract**
- frames object detection as a regression problem to spatially separated bounding boxes and associated class probabilities
- base YOLO model processes images in real-time at 45 frames per second
- Compared to SOTA detection systems, YOLO makes more localization errors but is less likely to predict false positives on background.
- YOLO learns very general representations of objects
---

**Introduction**

<img src="https://velog.velcdn.com/images/heayounchoi/post/5403f009-edfa-44dd-b3f4-77cae9ae174c/image.png" width="70%">

- YOLO is extremely fast
- fast version runs at more than 150 fps, which means we can process streaming video in real-time with less than 25 milliseconds of latency
- YOLO reasons globally about the image when making predictions
- unlike sliding window and region proposal-based techniques, YOLO sees the entire image during training and test time so it implicitly encodes contextual information about classes as well as their appearance
- Fast R-CNN, mistakes background patches in an image for objects because it can't see the larger context
- YOLO makes less than half the number of background errors compared to Fast R-CNN
- Since YOLO is highly generalizable it is less likely to break down when applied to new domains or unexpected inputs
---

**Unified Detection**
- YOLO divides the input image into an SxS grid
- if the center of an object falls into a grid cell, that grid cell is responsible for detecting that object
- each grid cell predicts B bounding boxes and confidence scores for those boxes
- these confidence scores reflect how confident the model is that the box contains an object and also how accurate it thinks the box is that it predicts
<img src="https://velog.velcdn.com/images/heayounchoi/post/34e011bf-d91e-402a-9356-28e10b774d02/image.png" width="30%">
<img src="https://velog.velcdn.com/images/heayounchoi/post/e552285e-fa1f-412a-88e3-4d1f87946ae9/image.png" width="70%">

- each grid cell also predicts C conditional class probabilities
<img src="https://velog.velcdn.com/images/heayounchoi/post/2e54c037-8ca0-4f3c-a79f-0e5a92a6f268/image.png" width="30%">

- one set of class probabilities is predicted per grid cell, regardless of the number of boxes B
<img src="https://velog.velcdn.com/images/heayounchoi/post/79a5d8b0-1d3c-4ecf-aa87-dde867994502/image.png" width="70%">
<img src="https://velog.velcdn.com/images/heayounchoi/post/43193083-477f-450b-875b-47c8bcc0453e/image.png" width="70%">

**_Network Design_**
<img src="https://velog.velcdn.com/images/heayounchoi/post/bc1b7f21-3096-45ba-8daf-ef42fafecb2e/image.png">
- Fast YOLO uses a neural network with fewer conv layers (9 instead of 24) and fewer filters in those layers
- other than the size of the network, all training and testing parameters are the same between YOLO and Fast YOLO

**_Training_**
- first 20 conv layers used to pretrain with ImageNet dataset
- four conv layers and two fc layers with randonly initialized weights added to perform detection
- input resolution increased from 224x224 to 448x448
- leaky rectified linear activation used for the final layer and all other layers
<img src="https://velog.velcdn.com/images/heayounchoi/post/e23bbb02-398a-4792-972f-283e1f81b37e/image.png" width="50%">

- optimize for sum-squared error in the output of the model
- this is used because it is easy to optimize, but it weights localization error equally with classification error which may not be ideal
- because many grid cells do not contain any object in every image, this pushes the confidence scores of those cells towards zero, often overpowering the gradients from cells that do contain objects (모델이 학습 과정에서 실제 객체가 존재하지 않는 셀의 신뢰도 점수를 0에 가깝게 만들려고 하고, 이러한 셀이 대다수를 차지하기 때문에 모델이 실제로 객체가 존재하는 셀에 대한 예측 정확도를 높이는 것보다 신뢰도 점수를 낮추는 방향으로 과도하게 최적화될 수 있음)
<img src="https://velog.velcdn.com/images/heayounchoi/post/faf89270-4aa8-47d4-a9b4-722c185b0986/image.png" width="70%">

- sum-squared error also equally weights errors in large boxes and small boxes
- error metric should reflect that small deviations in large boxes matter less than in small boxes
- to partially address this, the square root of the bounding box width and height is predicted instead of the width and height directly
<img src="https://velog.velcdn.com/images/heayounchoi/post/facb5db4-328b-488f-9f75-70261e68fb9d/image.png" width="70%">

- "responsible" means it has the highest IOU of any predictor in that grid cell
<img src="https://velog.velcdn.com/images/heayounchoi/post/bbe59921-b17f-4986-9d95-2782253b61aa/image.png" width="70%">
<img src="https://velog.velcdn.com/images/heayounchoi/post/71a6ac50-591a-44e3-a5e7-aa8a64bdb868/image.png" width="70%">
<img src="https://velog.velcdn.com/images/heayounchoi/post/20aae70d-da42-4970-b234-e4a1755b5dfd/image.png" width="70%">

**_Inference_**
- on PASCAL VOC the network predicts 98 bounding boxes per image and class probabilities for each box

**_Limitations of YOLO_**
- YOLO imposes strong spatial constraints on boudning box predictions since each grid cell only predicts two boxes and can only have one class
- this spatial constraint limits the number of nearby objects that the model can predict
- model struggles with small objects that appear in groups, such as flocks of birds
- 그리고 downsampling layers가 많아서 bounding box prediction을 위해 coarse features를 학습함
---

**Comparison to Other Detection Systems**

_Deformable parts models_
- DPM use a sliding window approach to object detection
- DPM uses a disjoint pipeline to extract static features, classify regions, predict bounding boxes for high scoring regions, etc.

_R-CNN_
- R-CNN and its variants use region proposals instead of sliding windows to find objects in images
- very slow, takes more than 40 seconds per image at test time

_Deep MultiBox_
- uses conv neural network to predict regions of interest instead of using Selective Search
- MultiBox can perform single object detection by replacing the confidence prediction with a single class prediction
- however, MultiBox cannt perform general object detection

_OverFeat_
- OverFeat uses conv neural network to perform localization and adapt that localizer to perform detection
- it optimizes for localization, not detection performance
- like DPM, the localizer only sees local information when making a prediction

_MultiGrasp_
- YOLO's grid approach to bounding box prediction is based on the MultiGrasp system for regression to grasps
- MultiGrasp only needs to predict a single graspable region for an image containing one object
- it doens't have to estimate the size, location, or boundaries of the object or predict it's class, only find a region suitable for grasping
---

**Experiments**
**_Comparison to Other Real-Time Systems_**
- YOLO using VGG-16 is more accurate but significantly slower than YOLO
- DPM은 빠르지만 neural network approaches에 비해 성능이 별로임
- VGG-16 version of Faster R-CNN is 10 mAP higher but also 6 times slower than YOLO
<img src="https://velog.velcdn.com/images/heayounchoi/post/412a0b58-c5af-46e9-8aaa-52e378ca345b/image.png" width="70%">

**_VOC 2007 Error Analysis_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/6dd610e9-2144-4991-9eef-68777ab64165/image.png" width="70%">

**_Combining Fast R-CNN and YOLO_**
- for every bounding box that R-CNN predicts, if YOLO predicts a similar box, that prediction gets a boost based on the probability predicted by YOLO and the overlap between the two boxes
- when best Fast R-CNN model is combined with YOLO, its mAP increases by 3.2%

**_Generalizability: Person Detection in Artwork_**
- R-CNN drops off considerably when applied to artwork because it uses Selective Search for bounding box proposals which is tuned for natural images
- DPM performs well because it has strong spatial models of the shape and layout of objects
- YOLO performs well because it models the size and shape of objects, as well as relationships between objects and where objects commonly appear
