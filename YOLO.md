# You Only Look Once: Unified, Real-Time Object Detection / YOLO

## Abstract
> ##### Sliding Window Approach
> - How it works: train a classifier on a fixed-size input to recognize an object -> to detect an object in a larger image, slide a window of the same fixed size across the image in various positions -> at each position, apply the classifier to the cropped portion of the image inside the window
> - This approach is computationally expensive, as the classifier has to evaluate every possible position and scale of the object in the image.
> ##### Region Proposals
> - A major improvement to the sliding window approach was the use of region proposal methods.
> - Algorithms like Selective Search, EdgeBoxes, or R-CNN first identify a manageable number of "interesting" bounding boxes (or regions) in the image where an object is likely to be. This reduces the number of windows the classifier has to evaluate.
> - Once these regions are proposed, a CNN classifier is applied to each region to classify the object within it.
> ##### R-CNN and its Variants
> - R-CNN (Regions with CNN features): Takes the region proposals from the image, applies a CNN to extract features from each region, and then classifies each region using SVMs. The bounding box positions are also refined using a linear regressor.
> - Fast R-CNN: Instead of applying a CNN to each region proposal separately, it applies the CNN to the entire image once, and then extracts features for each region from the feature map. This makes it much faster than the original R-CNN.
> - Faster R-CNN: Goes a step further by introducing the Region Proposal Network (RPN), which learns to propose regions using the same feature map used for classification, further speeding up the process.

- These methods often involve multiple stages and components, making them less efficient.
- By training end-to-end, every part of the model is optimized with the final goal in mind: accurate object detection. This holistic training approach can lead to better performance and efficiency.
- false positives: an instance where the model incorrectly predicts the presence of a target when it is not actually there

## Introduction
> ##### Deformable Part Models (DPM)
> - a method in object detection that represents objects by a collection of parts arranged in a deformable configuration
> - The parts can be adjusted (or "deformed") to achieve the best match to a given image, making the method particularly useful for detecting objects that can vary in appearance due to different poses or viewpoints.

## Unified Detection
### Network Design
> ##### Inception Modules
> - The core idea behind the Inception module is to capture information at various scales by applying filters of different sizes to the input, and then stacking the results together. This way, the model can learn from different spatial extents without committing to a single scale.
> - The module concurrently applies convolutions with different kernel sizes (like 1x1, 3x3, and 5x5) as well as a max-pooling operation. This allows the network to recognize patterns of different sizes and at different positions within the input.
> - Before applying the larger 3x3 and 5x5 convolutions, 1x1 convolutions are used to reduce the depth (number of channels) of the input. This helps in reducing the computational burden.
> - After the different filters are applied, the resulting feature maps are concatenated along the depth dimension. This means that instead of picking one scale or type of operation, the module retains information from all scales and operations.
> - Along with different convolutions, the module also applies a parallel max-pooling operation, which is then followed by a 1x1 convolution to reduce dimensionality. This result is also concatenated with the others.

### Training
- Caffe's Model Zoo: a repository and distribution system for pre-trained models developed in the Caffe deep learning framework
- Overpowering Gradient: Given the large number of grid cells that don't contain objects compared to those that do, the collective gradient from all these "empty" cells can be quite significant. This strong gradient, which pushes the confidence scores towards zero, can sometimes be so dominant that it overshadows or "overpowers" the gradient from cells that do contain objects. When this happens, the model might struggle to correctly learn the representation of actual objects because the feedback (gradient) from empty cells is diluting or countering the feedback from cells with objects. The model could start predicting lower confidence scores even for cells with objects, thinking it's better to be safe than sorry.
- When "training diverges", during the training process, the model is not learning the desired patterns from the data. Instead of improving its performance on the training data, the model's performance gets progressively worse.
- aspect ratio: the ratio of the width to the height of an image or a shape
- co-adaptation: a phenomenon in training deep neural networks where different neurons (or units) in the network become highly dependent on each other to correct their respective mistakes during the training process. Instead of each neuron learning unique and robust features independently, they "collaborate" in a way that may not generalize well to unseen data. This is undesirable because it can lead to overfitting, where the model performs well on the training data but poorly on new, unseen data.
- HSV: stands for Hue, Saturation, and Value. It's a color space that represents colors in a way that can be more intuitive and closer to how humans perceive and describe colors compared to the RGB (Red, Green, Blue) color space.
### Inference
- non-maximal suppression (NMS): a technique used in object detection to prune redundant or overlapping bounding boxes. After an object detection model processes an image, it often outputs multiple bounding boxes around the same object, each with an associated confidence score. NMS helps reduce this set of bounding boxes to retain only the most probable ones, effectively removing duplicates.
### Limitations of YOLO
## Comparison to Other Detection Systems
#### Deformable parts models
#### R-CNN
- Selective Search: The fundamental idea behind Selective Search is to merge similar regions based on various visual criteria to generate object proposals, rather than relying on exhaustive and computationally expensive sliding window methods.
> ##### how Selective Search works
> 1. Over-segmentation: The image is initially over-segmented into many small regions using a method like the Felzenszwalb-Huttenlocher algorithm. This results in hundreds of small segments that are more or less homogeneous in terms of color or texture.
> 2. Hierarchical Grouping: After over-segmentation, a greedy algorithm is used to hierarchically group (or merge) these segments based on visual similarity. The similarity can be computed using various criteria, including:
>	- Color Similarity: Based on the color distribution within regions.
>	- Texture Similarity: Based on texture distributions, which can be computed using methods like Gabor filters.
>	- Size Similarity: Smaller regions are preferred for merging to prevent the method from favoring larger regions excessively.
>	- Fit: A measure that encourages filling in gaps between regions.
> 3. Generating Proposals: During the hierarchical grouping process, all intermediate groupings of regions are considered as object proposals, which leads to the generation of thousands of region proposals for an image.
> 4. Diversification Strategies: The method can integrate different diversification strategies. For instance, by varying the color space used or by incorporating different similarity measures, you can obtain a wide variety of region proposals, increasing the chances of accurately capturing all objects.

#### Other Fast Detectors
- HOG computation: HOG, which stands for Histogram of Oriented Gradients, is a feature descriptor used primarily in object detection. The essence of the HOG computation is to extract information about edge directions and edge intensities in localized portions of an image, which can then be used to differentiate and detect objects of interest, especially when their appearance is relatively consistent across different instances.
- cascades: a technique often used to speed up object detection systems, and it's most famously associated with the Viola-Jones face detection algorithm. The basic idea behind cascades is to have a series of increasingly complex classifiers arranged in a sequential manner. At each stage or level of the cascade, simple and fast classifiers reject a majority of the negative examples (i.e., regions where the object of interest isn't present). Only the regions that pass one stage are then passed to the next, more complex classifier. This continues through the cascade, and at the end, if a region passes all stages, it's considered as containing the object of interest.
- "30Hz DPM" means that the DPM model can process 30 frames per second (fps).
#### Deep MultiBox
- MultiBox: an object detection method developed by researchers at Google. It's a precursor to some of the more advanced object detection methods like SSD (Single Shot MultiBox Detector). MultiBox differs from traditional object detection methods in that it simultaneously predicts both the bounding box locations and the class of objects, all in a single pass. It operates in a two-stage manner: proposal generation followed by classification. 
#### OverFeat
#### MultiGrasp
## Experiments
### Comparison to Other Real-Time Systems
### VOC 2007 Error Analysis
### Combining Fast R-CNN and YOLO
### VOC 2021 Results
### Generalizability: Person Detection in Artwork
## Real-Time Detection In The Wild
## Conclusion

---
#### 공부할 것
- ~~EdgeBoxes, R-CNN~~
- ~~SVM~~
- Region Proposal Network
- Deformable Part Models (DPM)
~~- GoogLeNet~~
- Caffe framework, Darknet framework
- Felzenszwalb-Huttenlocher algorithm
- Gabor filters
- HOG computation
- Viola-Jones face detection algorithm
- Single Shot MultiBox Detector
- MultiBox
- other Detection models

