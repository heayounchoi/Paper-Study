### [End-to-End Object Detection with Transformers](https://arxiv.org/pdf/2005.12872)

**Abstract**
- The main ingredients of the new framework, called DEtection TRansformer or DETR, are a set-based global loss that forces unique predictions via bipartite matching, and a transformer encoder-decoder architecture.
- Given a fixed small set of learned object queries, DETR reasons about the relations of the objects and the global image context to directly output the final set of predictions in parallel.
- DETR demonstrates accuracy and run-time performance on par with the well-established and highly-optimized Faster R-CNN baseline on the challenging COCO object detection dataset.
- DETR can be easily generalized to produce panoptic segmentation in a unified manner.
---

**Introduction**

<img src="https://velog.velcdn.com/images/heayounchoi/post/b101d495-41a7-425d-9796-526b1dfceca2/image.png">

- The self-attention mechanisms of transformers, which explicitly model all pairwise interactions between elements in a sequence, make these architectures particularly suitable for specific constraints of set prediction such as removing duplicate predictions.
- DETR simplifies the detection pipeline by dropping multiple hand-designed components that encode prior knowledge, like spatial anchors or non-maximal suppression.
- Compared to most previous work on direct set prediction, the main features of DETR are the conjunction of the bipartite matching loss and transformers with (non-autoregressive) parallel decoding.
- In constrast, previous work focused on autoregressive decoding with RNNs.
**- The matching loss function uniquely assigns a prediction to a ground truth object, and is invariant to a permutation of predicted objects, so we can emit them in parallel.**
- DETR achieves comparable performances to Faster R-CNN.
**- More precisely, DETR demonstrates significantly better performance on large objects, a result likely enabled by the non-local computations of the transformer.**
**- It obtains, however, lower performances on small objects.**
- DETR requires extra-long training schedule and benefits from auxiliary decoding losses in the transformer.
- A simple segmentation head trained on top of a pre-trained DETR outperforms competitive baselines on Panoptic Segmentation.
---

**Related Work**

**_Set Prediction_**
- The basic set prediction task is multilabel classification for which the baseline approach, one-vs-rest, does not apply to problems such as detection where there is an underlying structure between elements (i.e., near-identical boxes).
- Most current detectors use postprocessings such as nms to avoid near-duplicates, but direct set prediction are postprocessing-free.
- They need global inference schemes that model interactions between all predicted elements to avoid redundancy.
- For constant-size set prediction, dense fully connected networks are sufficient but costly.
- A general approach is to use auto-regressive sequence models such as rnn.
- In all cases, the loss function should be invariant by a permutation of the predictions.
- The usual solution is to design a loss based on the Hungarian algorithm, to find a bipartite matching between ground-truth and prediction.
- This enforces permutation-invariance, and guarantees that each target element has a unique match.

**_Transformers and Parallel Decoding_**
- Transformers introduced self-attention layers, which similarly to Non-Local Neural Networks, scan through each element of a sequence and update it by aggregating information from the whole sequence.
- Transformers were first used in auto-regressive models, following early sequence-to-sequence models, generating output tokens one by one.
- However, the prohibitive inference cost (proportional to output length, and hard to batch) lead to the development of parallel sequence generation.

**_Object detection_**
- Most modern object detection methods make predictions relative to some initial guesses.
- Recent work demonstrate that the final performance of these systems heavily depends on the exact way these initial guesses are set.
- In DETR we are able to remove this hand-crafted process and streamline the detection process by directly predicting the set of detections with absolute box prediction w.r.t. the input image rather than an anchor.

_Set-based loss_
- Learnable NMS methods and relation networks explicitly model relations between different predictions with attention.
- Using direct set losses, they do not require any post-processing steps.
- However, these methods employ additional hand-crafted context features like proposal box coordinates to model relations between detections efficiently, while we look for solutions that reduce the prior knowledge encoded in the model.

_Recurrent detectors_
- CNN, RNN based encoder-decoder architectures that use bipartite-matching
---

**The DETR model**
- Two ingredients are essential for direct set predictions in detection: (1) a set prediction loss that forces unique matching between predicted and ground truth boxes; (2) an architecture that predicts (in a single pass) a set of objects and models their relation.

<img src="https://velog.velcdn.com/images/heayounchoi/post/1a82865e-f192-4428-b848-62e644cf8699/image.png">

**_Object detection set prediction loss_**
- DETR infers a fixed-size set of N predictions, in a single pass through the decoder, where N is set to be significantly larger than the typical number of objects in an image.
- Our loss produces an optimal bipartite matching between predicted and ground truth objects, and then optimize object-specific (bounding box) losses.
- consider the ground truth set of objects also as a set of size N padded with no object
- The matching cost takes into account both the class prediction and the similarity of predicted and ground truth boxes.
- Each element i of the ground truth set can be seen as a set of target class label and a vector that defines ground truth box center coordinates and its height and width relative to the image size.
- This procedure of finding matching plays the same role as the heuristic assignment rules used to match proposal or anchors to ground truth objects in modern detectors.
- The main difference is that we need to find one-to-one matching for direct set prediction without duplicates.

_Boudning box loss_
- Unlike many detectors that do box predictions as a delta w.r.t. some initial guesses, we make box predictions directly.
- While such approach simplify the implementation it poses an issue with relative scaling of the loss.
- The most commonly-used L1 loss will have different scales for small and large boxes even if their relative errors are similar.
- To mitigate this issue we use a linear combination of the L1 loss and the generalized IoU loss that is scale-invariant.

<img src="https://velog.velcdn.com/images/heayounchoi/post/88f41e1b-fd21-4a3e-aefd-ecb0f3e7a4cb/image.png">

**_DETR architecture_**

_Backbone_
- Starting from the initial image, a conventional CNN backbone generates a lower-resolution activation map.
- The input images are batched together, applying 0-padding adequately to ensure they all have the same dimensions as the largest image of the batch.

_Transformer encoder_
- First, a 1x1 conv reduces the channel dimension of the high-level activation map from C (2048) to a smaller dimension, creating a new feature map.
- The encoder expects a sequence as input, hence we collapse the spatial dimensions of the new feature map into one dimension.
- multi-head self-attention module -> FFN
- positional encodings

_Transformer decoder_
- The difference with the original transformer is that our model decodes the N objects in parallel at each decoder layer, while the original one uses an autoregressive model that predicts the output sequence one element at a time.
- Since the decoder is also permutation-invariant, the N input embeddings must be different to produce different results.
**- These input embeddings are learnt positional encodings that we refer to as object queries, and similarly to the encoder, we add them to the input of each attention layer.**
**- The N object queries are transformed into an output embedding by the decoder.**
- They are then independently decoded into box coordinates and class labels by a FFN, resulting N final predictions.

_Prediction feed-forward networks (FFNs)_
- The final prediction is computed by a 3-layer perceptron with ReLU activation function and hidden dimension, and a linear projection layer.
- The FFN predicts the normalized center coordinates, height and width of the box w.r.t. the input image, and the linear layer predictes the class label using a softmax function.

_Auxiliary decoding losses_
- using auxiliary losses in decoder during training is helpful, especially to help the model output the correct number of objects of each class
- add prediction FFNs and Hungarian loss after each decoder layer
- all predictions FFNs share their parameters
- use an additional shared layer-norm to normalize the input to the prediction FFNs from different decoder layers

---

**Experiments**

_Dataset_
- COCO 2017 detection and panoptic segmentation datasets
- 7 instances per image on average, up to 63 instances in a single image in training set, ranging from small to large on the same images

_Technical details_
- backbone: ImageNet-pretrained ResNet model from torchvision with frozen batchnorm layers (50/101)
- also increase the feature resolution by adding a dilation to the last stage of the backbone and removing a stride from the first convolution of this tstage
- This modification increases the resolution by a factor of two, thus improving performance for small objects, at the cost of a 16x higher cost in the self-attentions of the encoder, leading to an overall 2x increase in computational cost.
- scale augmentation (shortest side 480~800, longest ~1333)
- random crop augmentations

**_Comparison with Faster R-CNN_**
- DETR achives higher $$AP_L$$ but lower $$AP_S$$

**_Ablations_**

_Number of encoder layers_
- The encoder seems to separate instances already, which likely simplifies object extraction and localization for the decoder.

_Number of decoder layers_
- NMS improves performance for the predictions from the first decoder.
- This can be explained by the fact that a single decoding layer of the transformer is not able to compute any cross-correlations between the output elements, and thus it is prone to making multiple predictions for the same object.
- In the second and subsequent layers, the self-attention mechanism over the activations allows the model to inhibit duplicate predictions.
- At the last layers, we observe a small loss in AP as NMS incorrectly removes true positive predictions.
- decoder attention is fairly local, meaning that it mostly attends to object extremeties such as heads or legs
- after the encoder has separated instances via global attention, the decoder only needs to attend to the extremities to extract the class and object boundaries

_Importance of FFN_
- FFN inside transformers can be seen as 1x1 conv layers, making encoder similar to attention augmented conv networks.

_Importance of positional encodings_
- There are two kinds of positional encodings in our model: spatial positional encodings and output positional encodings (object queries).

_Loss ablations_
- There are three components to the loss: classification loss, L1 bounding box distance loss, and GIoU loss.
- GIoU loss on its own accounts for most of the model performance.

**_Analysis_**

_Decoder output slot analysis_

_Generalization to unseen numbers of instances_

**_DETR for panoptic segmentation_**
- Similarly to the extension of Faster R-CNN to Mask R-CNN, DETR can be naturally extended by adding a mask head on top of the decoder outputs.
- Predicting boxes is required for the training to be possible, since the Hungarian matching is computed using distances between boxes.
- We also add a mask head which predicts a binary mask for each of the predicted boxes.
- It takes as input the output of transformer decoder for each object and computes multi-head attention scores of this embedding over the output of the encoder, generating M attention heatmaps per object in a small resolution.
- To make the final prediction and increase the resolution, an FPN-like architecture is used.
- The final resolution of masks has stride 4 and each mask is supervised independently using the DICE/F-1 loss and Focal loss.

<img src="https://velog.velcdn.com/images/heayounchoi/post/c7e70708-345b-458b-a150-122933ef91b2/image.png">

- To predict the final panoptic segmentation we simply use an argmax over the mask scores at each pixel, and assign the corresponding categories to the resulting masks.

<img src="https://velog.velcdn.com/images/heayounchoi/post/73945dd1-74b0-42a8-b2b7-36084c70157e/image.png">

<img src="https://velog.velcdn.com/images/heayounchoi/post/c641ec0c-4673-4379-8855-c6745e3b1b1c/image.png">

<img src="https://velog.velcdn.com/images/heayounchoi/post/62c57cca-71d2-4cc0-b95f-070a88cf0dd4/image.png">

- DICE 계수와 F-1 점수는 수학적으로 동일한 개념을 나타냄. DICE 계수는 주로 segmentation에서 사용되고, F-1 점수는 classification에서 사용됨. 

_Training details_
- During inference we first filter out the detection with a confidence below 85%, then compute the per-pixel argmax to determine in which mask each pixel belongs.
- We then collapse different mask predictions of the same stuff category in one, and filter the empty ones (less than 4 pixels).

_Main results_
- The result break-down shows that DETR is especially dominant on stuff classes, and we hypothesize that the global reasoning allowed by the encoder attention is the key element to this result.
- For things class, despite a severe deficit of up to 8 mAP compared to the baselines on the mask AP computation, DETR obtaines competitive things PQ.
- Mask AP가 낮고 PQ가 높은 경우, 모델이 개별 객체의 마스크 예측에서는 약점을 보이지만, 전체 이미지의 팬옵틱 분학에서는 강점을 보인다는 것을 의미함.
---

**Conclusion**
- This new design for detectors comes with new challenges, in particular regarding training, optimization and performances on small objects.
---

**Appendix**

**_Preliminaries: Multi-head attention layers_**

_Multi-head_

<img src="https://velog.velcdn.com/images/heayounchoi/post/dbf88589-b11c-47d6-9f66-c42c9b31aab0/image.png">

_Single head_

<img src="https://velog.velcdn.com/images/heayounchoi/post/d08a314b-2515-4084-b6f8-fcb9ac32f4d0/image.png">

_Feed-forward network (FFN) layers_
- two-layers of 1x1 convs with ReLU activations
- residual connection/dropout/layernorm after the two layers

**_Losses_**
- All losses are normalized by the number of objects inside the batch.
- since each GPU receives a sub-batch, it is not sufficient to normalize by the number of objects in the local batch, since in general the sub-batches are not balanced across GPUs
- Instead, it is important to normalize by the total number of objects in all sub-batches.

_Box loss_

<img src="https://velog.velcdn.com/images/heayounchoi/post/5bb0faa5-fd4f-4b72-b9e5-e8ef1c25fbac/image.png">

- 첫 번째 요소는 전통적인 IoU로, 예측된 박스와 실제 박스 간의 겹침 정도를 평가
- 두 번째 요소는 보정 항목으로, 최소 바운딩 박스 내에서 예측된 박스와 실제 박스가 차지하지 않는 영역의 비율을 계산
- 값이 0에 가까울수록 예측된 박스와 실제 박스가 정확하게 일치함을 의미함

_DICE/F-1 loss_

<img src="https://velog.velcdn.com/images/heayounchoi/post/49947142-786d-4fee-aa9c-9e742a749a4a/image.png">

**_Detailed architecture_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/ef93c341-cd2f-40f4-a380-cbd9b05a911e/image.png">

- Image features from the CNN backbone are passed through the transformer encoder, together with spatial positional encoding that are added to queries and keys at every multi-head self-attention layer.
- Then, the decoder receives queries (initially set to zero), output positional encoding (object queries), and encoder memory, and produces the final set of predicted class labels and bounding boxes through multiple multi-head self-attention and decoder-encoder attention.
- The first self-attention layer in the first decoder layer can be skipped.

_Computational complexity_

_FLOPs computation_

**_Training hyperparameters_**

_Backbone_
- ImageNet pretrained torchvision ResNet-50, discarding the last classification layer
- having the backbone learning rate roughly an order of magnitude smaller than the rest of the network is important to stabilize training

_Transformer_

_Losses_

_Baseline_

_Spatial positional encoding_
- fixed absolute encoding to represent spatial positions

**_Additional results_**

_Increasing the number of instances_
- By design, DETR cannot predict more objects than it has query slots, i.e. 100 in our experiments.
- model detects all instances when up to 50 objects are visible in an image, and starts saturating and misses more and more instances
- when the image contains all 100 instances, the model only detects 30 on average, which is less than if the image contains only 50 instances that are all detected
- The counter-intuitive behavior of the model is likely because the images and the detections are far from the training distribution.
- this experiment suggests that the model does not overfit the label and position distribution of the dataset since it yields near-perfect detections up to 50 objects
