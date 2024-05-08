### [Panoptic Segmentation](https://arxiv.org/pdf/1801.00868)

**Abstract**
- Panoptic segmentation unifies the typically distinct tasks of semantic segmentation (assian a class label to each pixel) and instance segmentation (detect and segment each object instance).
- The proposed task requires generating a coherent scene segmentation that is rich and complete, an important step toward real-world vision systems.
- We propose a novel panoptic quality (PQ) metric that captures performance for all classes (stuff and things) in an interpretable and unified manner.
---

**Introduction**

<img src="https://velog.velcdn.com/images/heayounchoi/post/2389cecb-35b2-4944-b514-8b02368f2308/image.png">

- Stuff classifiers are usually built on fully convolutional nets with dilations while object detectors often use object proposals and are region-based.
- In this work, we propose a task that: (1) encompasses both stuff and thing classes, (2) uses a simple but general output format, and (3) introduces a uniform evaluation metric.
- panoptic: including everything visible in one view
- task format: each pixel of an image must be assigned a semantic label and an instance id
- Pixels with the same label and id belong to the same object; for stuff labels the instance id is ignored.
---

**Related Work**

_Object detection tasks_

_Semantic segmentation tasks_

_Multitask learning_
- panoptic segmentation is not a multitask problem but rather a single, unified view of image segmentation

_Joint segmentation tasks_
- In the pre-deep learning era, there was substantial interest in generating coherent scene interpretations.
- The seminal work on image parsing proposed a general bayesian framework to jointly model segmentation, detection, and recognition.
- Later, approaches based on graphical models studied consistent stuff and thing segmentation.
- While these methods shared a common motivation, there was no agreed upon task definition, and different output formats and varying evaluation metrics were used, including separate metrics for evaluating results on stuff and thing classes.
- In recent years this direction has become less popular, perhaps for these reasons.

_Amodal segmentation task_
- the full extent of each region is marked, not just the visible
- this work focuses on segmentation of all visible regions only
---

**Panoptic Segmentation Format**

_Task format_
- Given a predetermined set of L semantic classes, the task requires a panoptic segmentation algorithm to map each pixel of an image to the semantic class of the pixel and its instance id.
- The instance ids group pixels of the same class into distinct segments.
- Ambiguous or out-of-class pixels can be assigned a special void label; not all pixels must have a semantic label.

_Stuff and thing labels_
- for stuff classes all pixels belong to the same instance
- the selection of which classes are stuff vs. things is a design choice left to the creator of the dataset

_Relationship to semantic segmentation_
- If the ground truth does not specify instances, or all classes are stuff, then the task formats are identical (although the task metrics differ).
- inclusion of thing classes, which may have multiple instances per image, differentiates the tasks

_Relationship to instance segmentation_
- instance segmentation allows overlapping segments, whereas the panoptic segmentation task permits only one semantic label and one instance id to be assigned to each pixel

_Confidence scores_
- not required
---

**Panoptic Segmentation Metric**
- existing metrics are specialized for either semantic or instance segmentation and cannot be used to evaluate the joint task involving both stuff and thing classes
- PQ metric involves two steps: (1) segment matching and (2) PQ computation given the matches

**_Segment Matching_**
- a predicted segment and a ground truth segment can match only if their intersection over union (IoU) is strictly greater than 0.5.
- This requirement, together with the non-overlapping property of a panoptic segmentation, gives a unique matching.

**_PQ Computation_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/0145f239-31ad-41ba-995e-51138a2d7077/image.png">

_Void labels_
- (a) out of class pixels, (b) ambiguous or unknown pixels
- predictions for void pixels are not evaluated

_Group labels_
- not used

**_Comparison to Existing Metrics_**

_Semantic segmentation metrics_
- common metrics for semantic segmentation include pixel accuracy, mean accuracy, and IoU
- IoU is the ratio between correctly predicted pixels and total number of pixels in either the prediction or ground truth for each class
- As these metrics ignore instance labels, they are not well suited for evaluating thing classes.

_Instance segmentation metrics_
- The standard metric for instance segmentation is Average Precision (AP).
- AP requires each object segment to have a confidence score to estimate a precision/recall curve.
- AP cannot be used for measuring the output of semantic segmentation.

_Panoptic quality_
- PQ treats all classes (stuff and things) in a uniform way.
- PQ is not a combination of semantic and instance segmentation metrics.
- Rather, SQ and RQ are computed for every class (stuff and things), and measure segmentation and recognition quality, respectively.
---

**Panoptic Segmentation Datasets**
- Cityscapes, ADE20k, Mapillary Vistas

_Cityscapes_
- ego-centric driving scenarios in urban settings

_ADE20k_
- densely annotated with an open-dictionary label set

_Mapillary Vistas_
- street-view images
---

**Human Consistency Study**

_Human annotations_

_Human consistency_

_Stuff vs. things_
- stuff and things have similar difficulty, although thing classes are somewhat harder

_Small vs. large objects_
- For small objects, RQ drops significantly implying human annotators often have a hard time finding small objects.

_IoU threshold_
- 0.5 good

_SQ vs. RQ balance_

---

**Machine Performance Baselines**

_Algorithms and data_
- to understand panoptic segmentation in terms of existing well-established methods, we create a basic PS system by applying reasonable heuristics to the output of existing top instance and semantic segmentation systems

_Instance segmentation_
- removing overlaps harms AP as detectors benefit from predicting multiple overlapping hypotheses
- improvements in a detector's AP will also improve its PQ

_Semantic segmentation_

_Panoptic segmentation_
- Due to the merging heuristic used, PQ of things stays the same while PQ of stuff is slightly degraded.

_Human vs. machine panoptic segmentation_
- For SQ, machines trail humans only slightly.
- On the other hand, machine RQ is dramatically lower than human RQ.

