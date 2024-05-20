### [Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1706.05587)

**Abstract**
- we propose to augment our previously proposed Atrous Spatial Pyramid Pooling module, which probes convolutional features at multiple scales, with image-level features encoding global context and further boost performance
- The proposed 'DeepLabv3' system significantly improves over our previous DeepLab versions without DenseCRF post-processing.
---

**Introduction**
- two challenges in applying DCNNs for the task of semantic segmentation:
1) reduced feature resolution caused by consecutive pooling operations or convolution striding, which allows DCNNs to learn increasingly abstract feature representations
- Atrous convolution, also known as dilated convolution, allows us to repurpose ImageNet pretrained networks to extract denser feature maps by removing the downsampling operations from the last few layers and upsampling the corresponding filter kernels, equivalent to inserting holes ('trous' in French) between filter weights.
- With atrous convolution, one is able to control the resolution at which feature responses are computed within DCNNs without requiring learning extra parameters.

<img src="https://velog.velcdn.com/images/heayounchoi/post/504561dc-d20c-45f3-bfdd-2524e0b9e7ed/image.png">

2) existence of objects at multiple scales
- Thrid, extra modules are cascaded on top of the original network for capturing long range information.
- Fourth, spatial pyramid pooling probes an incoming feature map with filters or pooling operations at multiple rates and multiple effective field-of-views, thus capturing objects at multiple scales.
- We experiment with laying out the modules in cascade or in parallel (specifically, ASPP method).
- We discuss an important practical issue when applying a 3x3 atrous convolution with an extremely large rate, which fails to capture long range information due to image boundary effects, effectively simply degenerating to 1x1 convolution, and propose to incorporate image-level features into the ASPP module.
---

**Related Work**
- four types of FCNs that exploit context information for semantic segmentation

_Image pyramid_
- The same model, typically with shared weights, is applied to multi-scale inputs.
- Feature responses from the small scale inputs encode the long-range context, while the large scale inputs preserve the small object details.
- The main drawback of this type of models is that it does not scale well for larger/deeper DCNNs due to limited GPU memory and thus it is usually applied during the inference stage.

_Encoder-decoder_
- This model consists of two parts: (a) the encoder where the spatial dimension of feature maps is gradually reduced and thus longer range information is more easily captured in the deepter encoder output, and (b) the decoder where object details and spatial dimension are gradually recovered.
- This type of model is also explored in the context of object detection.

_Context module_
- This model contains extra modules laid out in cascade to encode long-range context.
- CRF, bilateral convolution, ...

_Spatial pyramid pooling_
- This model employs spatial pyramid pooling to capture context at several ranges.
- ASPP, ...
- Recently, Pyramid Scene Parsing Net (PSP) performs spatial pooling at several grid scales and demonstrates outstanding performance on several semantic segmentation benchmarks.

- In this work, we mainly explore atrous convolution as a context module and tool for spatial pyramid pooling.
- To be concrete, we duplicate several copies of the original last block in ResNet and arrange them in cascade, and also revisit the ASPP module which contains several atrous convolutions in parallel.
- cascaded modules are applied directly on the feature maps instead of belief maps
- To further capture global context, we propose to augment ASPP with image-level features.

_Atous convolution_

---

**Methods**

**_Atrous Convolution for Dense Feature Extraction_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/7ff25e90-76d9-415c-8efc-383a2b29a3d4/image.png">

- For the DCNNs deployed for the task of image classification, the final feature responses (before fully connected layers or global pooling) is 32 times smaller than the input image dimension, and thus output_stride = 32.
- If one would like to double the spatial density of computed feature responses in the DCNNs (i.e., output_stride = 16), the stride of last pooling or convolutional layer that decreases resolution is set to 1 to avoid signal decimation.
- Then, all subsequent convolutional layers are replaced with atrous convolutional layers having rate r = 2.
- This allows us to extract denser feature responses without requiring learning any extra parameters.

**_Going Deeper with Atrous Convolution_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/d92c38af-2060-49b6-8eda-5018d3161baf/image.png">

- The motivation behind this model is that the stride 2 of the last convolution makes it easy to capture long range information in the deeper blocks.

**_Multi-grid Method_**
- Motivated by multi-grid methods which employ a hierarchy of grids of different sizes, we adopt different atrous rates within block4 to block7 in the proposed model.

**_Atrous Spatial Pyramid Pooling_**
- ASPP is inspired by the success of spatial pyramid pooling which showed that it is effective to resample features at different scales for accurately and efficiently classifying regions of an arbitrary scale.
- We include batch normalization within ASPP.

<img src="https://velog.velcdn.com/images/heayounchoi/post/fb89eea5-0c4a-4997-bca2-25e7e4c695b3/image.png">

- ASPP with different atrous rates effectively captures multi-scale information.
- However, we discover that as the sampling rate becomes larger, the number of valid filter weights (i.e., the weights that are applied to the valid feature region, instead of padded zeros) becomes smaller.
- In the extreme case where the rate value is close to the feature map size, the 3x3 filter, instead of capturing the whole image context, degenerates to simple 1x1 filter since only the center filter weight is effective.
- To overcome this problem and incorporate global context information to the model, we adopt image-level features.
- Specifically, we apply global average pooling on the last feature map of the model, feed the resulting image-level features to a 1x1 convolution with 256 filters (and batch normalization), and then bilinearly upsample the feature to the desired spatial dimension.

<img src="https://velog.velcdn.com/images/heayounchoi/post/727a9469-92ab-45d6-994e-1f98b4e0e9d0/image.png">

- The resulting features from all the branches are then concatenated and pass through another 1x1 convolution (also with 256 filters and batch normalization) before the final 1x1 convolution which generates the final logits.
---

**Experimental Evaluation**

**_Training Protocol_**

_Learning rate policy_
- "poly" learning rate policy

_Crop size_
- For atrous convolution with large rates to be effective, large crop size is required; otherwise, the filter weights with large atrous rate are mostly applied to the padded zero region.

_Batch normalization_
- Our added modules on top of ResNet all include batch normalization parameters, which we found important to be trained as well.
- large batch size is required to train batch normalization parameters

_Upsampling logits_
- In our previous works, the target groundtruths are downsampled by 8 during training when output_stride = 8.
- We find it important to keep the groundtruths intact and instead upsample the final logits, since downsampling the groundtruths removes the fine annotations resulting in no back-propagation of details.

_Data augmentation_

**_Going Deeper with Atrous Convolution_**

_ResNet-50_

<img src="https://velog.velcdn.com/images/heayounchoi/post/5ab4bdf6-fa27-4747-9bb3-f0cf883de20b/image.png">

_ResNet-50 vs ResNet-101_

<img src="https://velog.velcdn.com/images/heayounchoi/post/471f03d0-06db-4f8e-af90-e7ea3bc069ca/image.png">

_Multi-grid_

<img src="https://velog.velcdn.com/images/heayounchoi/post/62e1d999-9e9c-49cb-9987-74c79a80148a/image.png">

_Inference strategy on val set_

<img src="https://velog.velcdn.com/images/heayounchoi/post/e08a503a-eb27-455c-bf56-9efb9409d264/image.png">

**_Atrous Spatial Pyramid Pooling_**

_ASPP_

<img src="https://velog.velcdn.com/images/heayounchoi/post/e1867b8a-e077-45eb-b4d4-b5eafcf4e4e1/image.png">

- Multi-Grid가 (1,2,4)이고 ASPP가 동일할때 성능이 더 좋은걸로 봐서는 ASPP의 역할은 전체 context를 이해하는게 아닐까?
- ASPP를 rate=24에 대해서 했을때 성능이 떨어지는 이유는 보통 객체들은 r=18의 범위를 벗어나지 않기 때문 아닐까?

_Inference strategy on val set_

<img src="https://velog.velcdn.com/images/heayounchoi/post/970ffb00-79c0-47ee-b972-851529692e98/image.png">

_Comparison with DeepLabv2_
- Both our best cascaded model and ASPP model (in both cases without DenseCRF post-processing or MS-COCO pre-training) already outperform DeepLabv2 (77.69% with DenseCRF and pretrained on MS-COCO) on the PASCAL VOC 2012 val set.
- The improvement mainly comes from including and fine-tuning batch normalization parameters in the proposed models and having a better way to encode multi-scale context.

_Appendix_

_Qualitative results_

<img src="https://velog.velcdn.com/images/heayounchoi/post/3768e20d-d3ea-4e96-8e1d-b2f7c2a3d018/image.png">

_Failure mode_
- the model has difficulty in segmenting (a) sofa vs. chair, (b) dining table and chair, and (c) rare view of objects

_Pretrained on COCO_

_Test set result and an effective bootstrapping method_

<img src="https://velog.velcdn.com/images/heayounchoi/post/a52cf4d9-0d23-441c-9932-64dc6b87aae8/image.png">

- instead of performing pixel hard example mining, we resort to bootstrapping on hard images.
- In particular, we duplicate the images that contain hard classes (namely bicycle, chair, table, potted-plant, and sofa).

_Model pretrained on JFT-300M_

---

**Conclusion**

---

**Effect of hyper-parameters**

_New training protocol_
(1) larger crop size
(2) upsampling logits during training (instead of downsampling the groundtruths)
(3) fine-tuning batch normalization

<img src="https://velog.velcdn.com/images/heayounchoi/post/bca9ae29-2fd3-4f9d-b7f7-6a74bacb530a/image.png">

- boundary effect resulted from small crop size hurts the performance of DeepLabv3 which employs large atrous rates in the ASPP module

_Varying batch size_

<img src="https://velog.velcdn.com/images/heayounchoi/post/7e285b41-7f9b-43e0-b9bf-a3023ee52053/image.png">

_Output stride_

<img src="https://velog.velcdn.com/images/heayounchoi/post/f8a6a775-4cd9-4a47-821f-c75a6529bd7a/image.png">

- The value of output_stride determines the output feature map resolution and in turn affects the largest batch size we could use during training.
---

**Asynchronous training**
- training time with 32 replicas is significantly reduced to 2.74 hours

<img src="https://velog.velcdn.com/images/heayounchoi/post/3e7d9d72-04ad-4f79-bf4f-0371a6515449/image.png">

---

**DeepLabv3 on Cityscapes dataset**
- 잘 나왔다고 함

---

