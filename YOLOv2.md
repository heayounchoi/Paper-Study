### [YOLO9000: Better, Faster, Stronger](https://arxiv.org/pdf/1612.08242.pdf)

**Abstract**
- At 67 FPS, YOLOv2 gets 76.8 mAP on VOC 2007, and 78.6 mAP at 40FPS.
- This paper proposes a method to jointly train on object detection and classification.
- This join training allows YOLO9000 to predict detections for object classes that don't have labelled detection data.
---

**Introduction**
- First this paper improves upon the base YOLO detection system to produce YOLOv2.
- Then it uses its dataset combination method and joint training algorithm to train a model on more than 9000 classes from ImageNet as well as detection data from COCO.
---

**Better**
- Error analysis of YOLO compared to Fast R-CNN shows that YOLO makes a significant number of localization errors.
- Furthermore, YOLO has relatively low recall compared to region proposal-based methods.
- Instead of scaling up YOLO, YOLOv2 simplifyies the network and then make the representation easier to learn.

**_Batch Normalization_**
- Batch normalization leads to significant improvements in convergence while eliminating the need for other forms of regularization.
- By adding batch normalization on all of the conv layers in YOLO, YOLOv2 gets more than 2% improvement in mAP.
- With batch normalization, dropout can be removed from the model without overfitting.

**_High Resolution Classifier_**
- Detection methods use classifier pre-trained on ImageNet, which operates on input images smaller than 256 x 256.
- YOLOv1 trains the classifier network at 224 x 224 and increases the resolution to 448 for detection. 
- This means the network has to simultaneously switch to learning object detection and adjust to the new input resolution.
- YOLOv2 first fine tunes the classification network at the full 448 x 448 resolution for 10 epochs on ImageNet.
- This gives the network time to adjust its filters to work better on higher resolution input.
- The model then fine tunes the resulting network on detection.
- This high resolution classification network gives an increase of almost 4% mAP.

**_Convolution With Anchor Boxes_**
- YOLO predicts the coordinates of bounding boxes directly using fully connected layers on top of the convolutional feature extractor.
- Instead of predicting coordinates directly, Faster R-CNN predicts offsets and confidences for anchor boxes.
- Predicting offsets instead of coordinates simplifies the problem and makes it easier for the network to learn.
- YOLOv2 removes the fully connected layers from YOLO and use anchor boxes to predict bounding boxes.
- first, eliminate one pooling layer to make the ouput of the network's conv layers higher resolution
- shrink the network to operate on 416 input images instead of 448 x 448
- this is done to get an odd number of locations in the feature map so there is a single center cell
- objects, especially large objects, then to occupy the center of the image so it's good to have a single location right at the center to predict these objects instead of four locations that are all nearby
- YOLO's conv layers downsample the image by a factor of 32 so by using an input image of 416, the output feature map is 13 x 13
- instead of using class prediction mechanism from the spatial location, YOLOv2 predicts class and objectness for every anchor box
- Following YOLO, the objectness prediction still predicts the IOU of the ground truth and the proposed box and the class predictions predict the conditional probability of that class given that there is an object.
- YOLO only predicts 98 boxes per image but with anchor boxes, YOLOv2 predicts more than a thousand.
- Without anchor boxes, the intermediate model gets 69.5 mAP with a recall of 81%. With anchor boxes, YOLOv2 gets 69.2 mAP with a recall of 88%.
- Even though the mAP decreases, the increas in recall means that the model has more room to improve.

**_Dimension Clusters_**
- There are two issues with anchor boxes when using them with YOLO.
- The first is that the box dimensions are hand picked.
- The network can learn to adjust the boxes appropriately but if better priors are pickeed for the network to start with, it is easier for the network to learn to predict good detections.
- Instead of choosing prios by hand, k-means clustering is ran on the training set bounding boxes to automatically find good priors.
- If we use standard k-means with Euclidean distance larger boxes generate more error than smaller boxes.
- However, what we really want are priors that lead to good IOU scores, which is independent of the size of the box.
  
<img src="https://velog.velcdn.com/images/heayounchoi/post/82c0d915-c3e1-4b49-85f9-ca6f5b00f249/image.png" width="50%">

- run k-means for various values of k and plot the average IOU with closest centroid
  
<img src="https://velog.velcdn.com/images/heayounchoi/post/be4468af-c141-4654-b96e-fc6ad9c775b6/image.png" width="50%">

- The cluster centroids are significantly different than hand-picked anchor boxes.
- There are fewer short, wide boxes and more tall, thin boxes.

**_Direct location prediction_**
- When using anchor boxes with YOLO, a second issue is encountered: model instability, especially during the early iterations.
- Most of the instability comes from predicting the (x, y) locations for the box.
- The formulation predicting locations in RPNs is unconstrained so any anchor box can end up at any point in the image, regardless of what location predicted the box.
- With random initialization the model takes a long time to stabilize to predicting sensible offsets.
- Instead of predicting offsets, YOLOv2 follows the approach of YOLO and predict location coordinates relative to the location of the grid cell.
- This bounds the ground truth to fall between 0 and 1.
- The network predicts 5 bouding boxes at each cell in the ouput feature map.
- The network predicts 5 coordinates for each bounding box, tx, ty, tw, th, and to.
- If the cell is offset from the top left corner of the image by (cx, cy) and the bounding box prior has width and height pw, ph, then the predictions correspond to:
  
<img src="https://velog.velcdn.com/images/heayounchoi/post/96f84271-d3b0-4ad1-bc2a-d12c9d326e4c/image.png" width="50%">

- Since the location prediction is constrained, the parametrization is easier to learn, making the network more stable.
- Using dimension clusters along with directly predicting the bouding box center location improves YOLO by almost 5% over the version with anchor boxes.

**_Fine-Grained Features_**
- YOLOv2 predicts detections on a 13 x 13 feature map.
- While this is sufficient for large objects, it may benefit from finer grained features for localizing smaller objects.
- YOLOv2 adds a passthrough layer that brings features from an earlier layer at 26 x 26 resolution.
- The passthrough layer concatenates the higher resolution features with the low resolution features by stacking adjacent features into different channels instead of spatial locations, similar to the identity mappings in ResNet.
- This turns the 26 x 26 x 512 feature map into a 13 x 13 x 2048 feature map, which can be concatenated with the original features.
- This gives a modest 1% performance increase.

**_Multi-Scale Training_**
- Since YOLOv2 only uses conv and pooling layers, it can be resized on the fly.
- Instead of fixing the input image size, the network is changed every few iterations.
- Every 10 batches, the network randomly chooses a new image dimension size.
- Since the model downsamples by a factor of 32, it's pulled from the following multiples of 32: {320, 352, ..., 608}.
- This regime forces the network to learn to predict well across a variety of input dimensions.
- At low resolutions YOLOv2 operates as a cheap, fairly accurate detector.
- At 228 x 228 it runs at more than 90 FPS with mAP almost as good as Fast R-CNN.
- At high resolution YOLOv2 is a SOTA detector with 78.6 mAP on VOC 2007 while still operating above real-time speeds.
---

**Faster**
- Most detection frameworks rely on VGG-16 as the base feature extractor.
- The YOLO framework uses a custom network based on the Googlenet architecture.

**_Darknet-19_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/913769b3-55e2-4f0e-a920-3a5751394932/image.png" width="50%">

- has 19 conv layers and 5 maxpooling layers

**_Training for detection_**
- network modified for detection by removing the last conv layer and instead adding on three 3 x 3 conv layers with 1024 filters each followed by a final 1 x 1 conv layer with the number of outputs needed for detection
- For VOC, predict 5 boxes with 5 coordinates each and 20 classes per box so 125 filters.
- also add a passthrough layer from the final 3 x 3 x 512 layer to the second to the last conv layer so that the model can use fine grain features
- train the network for 160 epochs with a starting learning rate of 10^-3, dividing it by 10 at 60 and 90 epochs
- weight decay 0.0005
- momentum 0.9
---

**Stronger**
- mechanism for jointly training on classification and detection data
- during training, mix images from both detection and classification datasets
- when the network sees an image labelled for detection, backpropagate based on the full YOLOv2 loss function
- when it sees a classification image, only backpropagate loss from the classification-specific parts of the architecture
- YOLO 9000 구현 관련 내용.. ~
