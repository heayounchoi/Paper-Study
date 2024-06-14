### Learning Transferable Visual Models From Natural Language Supervision

**Abstract**
- Learning directly from raw text about images is a promising alternative which leverages a much broader source of supervision.
- We demonstrate that the simple pre-training task of predicting which caption goes with which image is an efficient and scalable way to learn SOTA image representations from scratch on a dataset of 400 million (image, text) pairs collected from the internet.
- After pre-training, natural language is used to reference learned visual concepts (or describe new ones) enabling zero-shot transfer of the model to downstream tasks.
- The model transfers non-trivially to most tasks and is often competitive with a fully supervised baseline without the need for any dataset specific training.
---

**Introduction and Motivating Work**
- Enabled by the large amounts of publicly available data of this form on the internet, we create a new dataset of 400 million (image, text) pairs and demonstrate that a simplified version of ConVIRT trained from scratch, which we call CLIP, for Contrastive Language-Image Pre-training, is an efficient method of learning from natural language supervision.
---

**Approach**

**_Natural Language Supervision_**
- Learning from natural language has several potential strengths over other training methods.
- It's much easier to scale natural language supervision compared to standard crowd-sourced labeling for image classification since it does not require annotations to be in a classic "machine learning compatible format" such as the canonical 1-of-N majority vote "gold label".
- Instead, methods which work on natural language can learn passively from the supervision contained in the vast amount of text on the internet.
- Learning from natural language also has an important advantage over most unsupervised or self-supervised learning approaches in that it doesn't "just" learn a representation but also connects that representation to language which enables flexible zero-shot transfer.

**_Creating a Sufficiently Large Dataset_**
- We constructed a new dataset of 400 million (image, text) pairs collected from a variety of publicly available sources on the Internet.
- To attempt to cover as broad a set of visual concepts as possible, we search for (image, text) pairs as part of the construction process whose text includes one of a set of 500,000 queries.
- We approximately class balance the results by including up to 20,000 (image, text) pairs per query.
- The resulting dataset has a similar total word count as the WebText dataset used to train GPT-2.
- We refer to this dataset as WIT for WebImageText.

**_Selecting an Efficient Pre-Training Method_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/841dcf2e-acc4-4ff6-9277-51233c5893f4/image.png">

- Our initial approach, similar to VirTexm jointly trained an image CNN and text transformer from scratch to predict the caption of an image.
- However, we encoutered difficulties efficiently scaling this method.
- A 63 million parameter transformer language model, which already uses twice the compute of its ResNet-50 image encoder, learns to recognize ImageNet classes three times slower than a much simpler baseline that predicts a bag-of-words encoding of the same text.
- Both these approaches share a key similarity.
- They try to predict the exact words of the text accompanying each image.
- This is a difficult task due to the wide variety of descriptions, comments, and related text that co-occur with images.
- Recent work in contrastive representation learning for images has found that contrastive objectives can learn better representations than their equivalent predictive objective.
- Other work has found that although generative models of iamges can learn high quality image representations, they require over an order of magnitude more compute than contrastive models with the same performance.
- Noting these findings, we explored training a system to solve the potentially easier proxy task of predicting only which text as a whole is paired with which image and not the exact words of that text.
- Starting with the same bag-of-words encoding baseline, we swapped the predictive objective for a contrastive objective and observed a further 4x efficiency improvement in the rate of zeo-shot transfer to ImageNet.
- Given a batch of N (image, text) pairs, CLIP is trained to predict which of the N x N possible (image, text) pairings across a batch actually occurred.
- To do this, CLIP learns a multi-modal embedding space by jointly training an image encoder and text encoder to maximize the cosine similarity of the image and text embeddings of the N real pairs in the batch while minimizing the cosine similarity of the embeddings of the $$N^2$$ - N incorrect pairings.
- We optimize a symmetric cross entropy loss over these similarity scores.

<img src="https://velog.velcdn.com/images/heayounchoi/post/b0ebff22-9a71-4760-9f7a-aba7601a9a2e/image.png">

- We train CLIP from scratch without initializing the image encoder with ImageNet weights or the text encoder with pre-trained weights.
- We use only a linear projection to map from each encoder's representation to the multi-modal embedding space.
- the temperature parameter which controls the range of the logits in the softmax is directly optimized during training as a log-parameterized multiplicative scalar to avoid turning as a hyper-paramter

**_Choosing and Scaling a Model_**
- We consider two different architectures, modified ResNet-50 and ViT, for the image encoder.
- The text encoder is a modified Transformer.
- we found CLIP's performance is less sensitive to the capacity of the text encoder

**_Training_**
- all results reported in this paper as "CLIP" use ViT-L/14@336px which we found to perform best
---

**Experiments**

**_Zero-Shot Transfer_**

_Motivation_

- In computer vision, zero-shot learning usually refers to the study of generalizing to unseen object categories in image classification.
- We instead use the term in a broader sense and study generalization to unseen datasets.
- While much research in the field of unsupervised learning focuses on the representation learning capabilities of machine learning systems, we motivate studying zero-shot transfer as a way of measuring the task-learning capabilities of machine learning systems.

_Using CLIP for Zero-Shot Transfer_

- CLIP is pre-trained to predict if an image and a text snippet are paired together in its dataset.
- To perform zero-shot classification, we reuse this capability.
- For each dataset, we use the names of all the classes in the dataset as the set of potential text pairings and predict the most probable (image, text) pair according to CLIP.

_Initial Comparison to Visual N-Grams_

<img src="https://velog.velcdn.com/images/heayounchoi/post/407ea9b3-958f-4625-8ccd-ba4e6aa7b280/image.png">

- The ability to match the performance of a strong, fully supervised baselines in a zero-shot setting suggests CLIP is a significant step towards flexible and practical zero-shot computer vision classifiers.

_Prompt Engineering and Ensembling_

- Most standard image classification datasets annotate images with just a numeric id of the label and contain a file mapping these ids back to their names in English.
- A common issue is polysemy.
- When the name of a class is the only information provided to CLIP's text encoder it is unable to differentiate which word sense is meant due to the lack of context.
- Another issue we encountered is that it's relatively rare in our pre-training dataset for the text paired with the image to be just a single word.
- we have also observed that zero-shot performance can be significantly improved by customizing the prompt text to each task
- We also experimented with ensembling over multiple zero-shot classifiers as another way of improving performance.

<img src="https://velog.velcdn.com/images/heayounchoi/post/4d132ca4-7d22-4b10-961f-fdf3446a067b/image.png">

_Analysis of Zero-Shot CLIP Performance_

<img src="https://velog.velcdn.com/images/heayounchoi/post/946b492f-d20a-4183-84e5-d8bdb99966b6/image.png">

- On fine-grained classification tasks, we observe a wide spread in performance.
- On "general" object classification datasets, performance is relatively similar with a slight advantage for zero-shot CLIP in all cases.
- Zero-shot CLIP significantly outperforms a ResNet-50 on two datasets measuring action recognition in videos.
- We speculate this is due to natural language providing wider supervision for visual concepts involving verbs, compared to the noun-centric object supervision in ImageNet.
- Looking at where zero-shot CLIP notably underperforms, we see that zero-shot CLIP is quite weak on several specialized, complex, or abstract tasks.
- However, we caution that it is unclear whether measuring zero-shot transfer, as opposed to few-shot transfer, is a meaningful evaluation for difficult tasks that a learner has no prior experience with.

<img src="https://velog.velcdn.com/images/heayounchoi/post/85f45450-711b-44b3-87c7-6be459553e26/image.png">

- we find that zero-shot CLIP matches the performance of 4-shot logistic regression on the same feature space
- This is likely due to an important difference between the zero-shot and few-shot approach.
- First, CLIP's zero-shot classifier is generated via natural language which allows for visual concepts to be directly specified.
- By contrast, "normal" supervised learning must infer concepts indirectly from training examples.
- Context-less example-based learning has the drawback that many different hypotheses can be consistent with the data, especially in the one-shot case.
- A single image often contains many different visual concepts.

<img src="https://velog.velcdn.com/images/heayounchoi/post/d11b5c1c-6cc7-49e8-a315-d1aa01a35294/image.png">

<img src="https://velog.velcdn.com/images/heayounchoi/post/98aff00d-8918-4010-b2bc-e3984b0a9fd3/image.png">

- If we assume that evaluation datasets are large enough that the parameters of linear classifiers trained on them are well estimated, then, because CLIP's zero-shot classifier is also a linear classifier, the performance of the fully supervised classifiers roughly sets an upper bound for what zero-shot transfer can achieve.
- On 5 datasets, both zero-shot accuracy and fully supervised accuracy are over 90%.
- This suggests that CLIP may be more effective at zero-shot transfer for tasks where its underlying representations are also high quality.

<img src="https://velog.velcdn.com/images/heayounchoi/post/284179f8-49f1-4b6b-bb79-656b975b021a/image.png">

- Over the past few years, empirical studies of deep learning systems have documented that performance is predictable as a function of important quantities such as training compute and dataset size.
- The GPT family of models has so far demonstrated consistent improvements in zero-shot performance across a 1000x increase in training compute.

**_Representation Learning_**
- While we have extensively analyzed the task-learning capabilities of CLIP through zero-shot transfer in the previous section, it is more common to study the representation learning capabilities of a model.

<img src="https://velog.velcdn.com/images/heayounchoi/post/157cd0ea-b433-45a0-be83-35d7eae84068/image.png">

- models trained with CLIP scale very well
- We also find that CLIP vision transformers are about 3x more compute efficient than CLIP ResNets, which allows us to reach higher overall performance within our compute budget.
- These results qualitatively replicate the previous findings which reported that vision transformers are more compute efficient than convnets when trained on sufficiently large datasets.
- CLIP models learn a wider set of tasks than has previously been demonstrated in a single computer vision model trained end-to-end from random initialization.
- These tasks include geo-localization, optical character recognition, facial emotion recognition, and action recognition.
- self-supervised systems do noticeably better on our broader evaluation suite
- These findings suggest continuing to expand task diversity and coverage in order to better understand the "general" performance of systems.

<img src="https://velog.velcdn.com/images/heayounchoi/post/1008752b-5367-41da-a009-f22021c4800f/image.png">

- a supervised representation collapses intra-class details and hurt accuracy on a fine-grained downstream task
- the dataset that the EfficientNet does best relative to CLIP on is the one it was trained on: ImageNet
- The EfficientNet also slightly outperforms CLIP on low-resolution datasets.
- We suspect this is at least partly due to the lack of scale-based data augmentation in CLIP.

**_Robustness to Natural Distribution Shift_**
- deep learning models are exceedingly adept at finding correlations and patterns which hold across their training dataset and thus improve in-distribution performance
- However many of these correlations and patterns are actually spurious and do not hold for other distributions and result in large drops in performance on other datasets.
- Effective robustness measures improvements in accuracy under distribution shift above what is predicted by the documented relationship between in-distribution and out-of-distribution accuracy.
- Relative robustness captures any improvement in out-of-distribution accuracy.

<img src="https://velog.velcdn.com/images/heayounchoi/post/953e4df9-3bad-47f2-a7b2-5cf5269a5248/image.png">

<img src="https://velog.velcdn.com/images/heayounchoi/post/f9b25889-a780-48d1-87e8-633f6673ab69/image.png">

- While zero-shot models can be much more robust, they do not necessarily mean that supervised learning on ImageNet causes a robustness gap.
- Other details of CLIP, such as its large and diverse pre-training dataset or use of natural language supervision could also result in much more robust models regardless of whether they are zero-shot or fine-tuned.
- As an inital experiment to potentially begin narrowing this down, we also measure how the performance of CLIP models change after adapting to the ImageNet distribution via a L2 regularized logistic regression classifier fit to CLIP features on the ImageNet training set.
- It is surprising to see a 9.2% increase in accuracy fail to translate into any improvement in average performance under distribution shift.
- ImageNetV2 closely followed the creation process of the original ImageNet dataset.

<img src="https://velog.velcdn.com/images/heayounchoi/post/d4a2517c-d572-4f3c-b28f-a2d81ad27f56/image.png">

- Across our experiments, high effective robustness seems to result from minimizing the amount of distribution specific training data a model has access to, but this comes at a cost of reducing dataset-specific performance.

---

**Comparison to Human Performance**

- We wanted to get a sense of how strong human zero-shot performance is at these tasks, and how much human performance is improved if they are shown one or two image samples.
- This can help us to compare task difficulty for humans and CLIP, and identify correlations and differences between them.
- humans went from a performance average of 54% to 76% with just one training example per class, and the marginal gain from an additional training example is minimal.
- The gain in accuracy going from zero to one shot is almost entirely on images that humans were uncertain about.
- This suggests that humans "know what they don't know" and are able to update their priors on the images they are most uncertain in based on a single example.

<img src="https://velog.velcdn.com/images/heayounchoi/post/b551ca43-ef7b-4189-813b-a8188be31e86/image.png">

- there are still algorithmic improvements waiting to be made to decrease the gap between machine and human sample efficiency
- Because these few-shot evaluations of CLIP don't make effective use of prior knowledge and the humans do, we speculate that finding a method to properly integrate prior knowledge into few-shot learning is an important step in algorithmic improvements to CLIP.

<img src="https://velog.velcdn.com/images/heayounchoi/post/db49ecb9-f0b5-4b2c-89fa-9e9dc8bda51a/image.png">

---

**Data Overlap Analysis**

<img src="https://velog.velcdn.com/images/heayounchoi/post/c9c56ace-b46f-4886-8171-06d4ac1c2a97/image.png">

- A concern with pre-training on a very large internet dataset is unintentional overlap with downstream evals.
- we document how much overlap occurs and how performance changes due to these overlaps
- There is a median overlap of 2.2% and an average overlap of 3.2%.
- Despite large overlap for Country211, there is only a 0.2% increase in accuracy.
- This may be because the training text accompanying an example is often not related to the specific task a downstream eval measures.
- there are potential concerns with this analysis: changes in accuracy could instead be due to changes in the class distribution or difficulty of the duplicates
- However, these results closely follow the findings of similar duplicate analysis in previous work on large scale pre-training.
---

**Limitations**
- On datasets with training splits, the performance of zero-shot CLIP is on average competitive with the simple supervised baseline of a linear classifier on top of ResNet-50 features.
- On most of these datasets, the performance of this baseline is now well below the overall state of the art.
- we estimate around a 1000x increase in compute is required for zero-shot CLIP to reach overall SOTA performance, which is infeasible to train with current hardware
- the performance of CLIP is poor on several types of fine-grained classification
- CLIP also struggles with more abstract and systematic tasks such as counting the number of objects in an image.
- for novel tasks which are unlikely to be included in CLIP's pre-training dataset, such as classifying the distance to the nearest car in a photo, CLIP's performance can be near random
- zero-shot CLIP still generalizes poorly to data that is truly out-of-distribution for it
- CLIP is still limited to choosing from only those concepts in a given zero-shot classifier
- This is a significant restriction compared to a truly flexible approach like image captioning which could generate novel outputs.
- CLIP also does not address the poor data efficiency of deep learning.
- Combining CLIP with self-supervision and self-training methods is a promising direction given their demonstrated ability to improve data efficiency over standard supervised learning.
- new benchmark of tasks designed explicitly to evaluate broad zero-shot transfer capabilities, rather than re-using existing supervised datasets, is necessary
- CLIP's image-text pairs are unfiltered and uncurated and result in CLIP models learning many social biases.
- While we have emphasized throughout this work that specifying image classifiers through natural language is a flexible and general interface, it has its own limitations.
- Many complex tasks and visual concepts can be difficult to specify just through text.
- Actual training examples are undeniably useful but CLIP does not optimize for few-shot performance directly.
- Future work is needed to develop methods that combine CLIP's strong zero-shot performance with efficient few-shot learning.

---

**Broader Impacts**

**_Bias_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/dd20dda2-7a34-4b02-b2a7-155988825a0b/image.png">

<img src="https://velog.velcdn.com/images/heayounchoi/post/ca06fb6c-a7a8-4806-8cb4-f082e1b46c3e/image.png">

<img src="https://velog.velcdn.com/images/heayounchoi/post/4418ecf5-2c57-4e19-8bc7-cd0885605dac/image.png">

<img src="https://velog.velcdn.com/images/heayounchoi/post/2ae38740-b0e9-4239-ae26-d378dc1906cb/image.png">

<img src="https://velog.velcdn.com/images/heayounchoi/post/90df2114-5657-4be4-b6a3-01d8e74270e3/image.png">

- Table 7 points to how class design has the potential to be a key factor determining both the model performance and the unwanted biases or behaviour the model may exhibit while also asks overarching questions about the use of face images to automatically classify people along such lines.

<img src="https://velog.velcdn.com/images/heayounchoi/post/4f0d0995-dcd7-497e-9b95-5d5c0104e674/image.png">

- Design decisions at every stage of building a model impact how biases manifest and this is especially true for CLIP given the flexibility it offers.
- In addition to choices about training data and model architecture, decisions about things like class designs and thresholding values can alter the labels a model outputs and as a result heighten or lower certain kinds of harm.
- People designing and developing models and AI systems have considerable power.

**_Surveillance_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/c8cc60b3-14e5-48ac-81a8-e5c59b47beba/image.png">

- large datasets and high performing supervised models exist for many in-demand surveillance tasks such as facial recognition
- As a result, CLIP's comparative appeal for such uses is low.
- Additionally, CLIP is not designed for common surveillance-relevant tasks like object detection and semantic segmentation.
- CLIP and similar models could enable bespoke, niche surveillance use cases for which no well-tailored models or datasets exist, and could lower the skill requirements to build such applications.

**_Future Work_**

---

**Related Work**

---

**Conclusion**

---

**Linear-probe evaluation**

**_Datasets_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/04a9c4c1-36fa-4b16-9c18-dd3275482311/image.png">

**_Models_**

_LM RN50_
- multimodal model that uses an autoregressive loss instead of a contrastive loss, while using the ResNet-50 architecture as in the smallest contrastive model
- To do so, the output from the CNN is projected into four tokens, which are then fed as a prefix to a language model autoregressively predicting the text tokens.

_CLIP-RN_
- Five ResNet-based contrastive CLIP models are included.

_CLIP-ViT_
- four CLIP models that use the Vision Transformer architecture as the image encoder

_EfficientNet_

_Instagram-pretrained ResNeXt_

_Big Transfer (BiT)_

_Vision Transformer (ViT)_

_SimCLRv2_

_BYOL_

_Momentum Contrast (MoCo)_

_VirTex_
- VirTex has a similar model design to CLIP-AR but is trained on a 1000x smaller dataset of high-quality captions from MSCOCO

_ResNet_

**_Evaluation_**
- We use image features taken from the penultimate layer of each model, ignoring any classification layer provided.
- We train a logistic regression classifier.

**_Results_**
- The best-performing CLIP model, using ViT-L/14 architecture and 336-by-336 pixel images, achieved the state of the art in 21 of the 27 datasets.

---

**Zero-Shot Prediction**

---

**Duplicate Detector**
- Our early attempts at duplicate detection and analysis used nearest neighbors in the model's learned embedding space.
- While it is intuitive to use a model's own notion of similarity, we encountered issues.
- We found the model's feature space is weighted very heavily towards semantic similarity.
- Many false positives occurred due to distinct objects that would be described similarly (soccer balls, flowers of the same species, etc...) having almost perfect similarity.
- We also observed the model was quite poor at assigning certain kinds of near-duplicates high similarity scores.
- We noticed repeatedly that images with high-frequency textures (such as fur or stripe patterns) pre-processed by different resizing algorithms (nearest neighbor vs bi-linear) could have surprisingly low similarity.
- This resulted in many false negatives.
- We built our own near-duplicate detector to fix this issue.
- We created a synthetic data augmentation pipeline that combined a variety of common image manipulations.

---

**Dataset Ablation on YFCC100M**

<img src="https://velog.velcdn.com/images/heayounchoi/post/1447cd08-ede0-4dee-95c9-47fe03ad77b2/image.png">

- To study whether our custom dataset is critical to the performance of CLIP, we trained a model on a filtered subset of the YFCC100M dataset and compared its performance to the same model trained on an equally sized subset of WIT.

---

**Selected Task and Dataset Results**

**_Image and Text Retrieval_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/f0723f8d-c5c9-40d0-b033-1e6bb67c3b0c/image.png">

- CLIP pre-trains for the task of image-text retrieval on our noisy web-scale dataset.
- Although the focus of this paper is on representation learning and task learning for the purpose of transfer to a wide variety of downstream datasets, validating that CLIP is able to achieve high transfer performance on exactly what it is pre-trained for is an important sanity check / proof of concept.

**_Optical Character Recognition_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/dc91f4f0-c33c-4e96-a009-c6d459bed253/image.png">

- Although visualizations have shown that ImageNet models contain features that respond to the presence of text in an image, these representations are not sufficiently fine-grained to use for the task of optical character recognition (OCR).
- we measured performance on 5 datasets requiring the direct and indirect use of OCR

**_Action Recognition in Videos_**

- Does the lack of broader supervision in ImageNet result in weaker transfer of ImageNet models to tasks involving the recognition of visual concepts that are not nouns?

<img src="https://velog.velcdn.com/images/heayounchoi/post/ce699236-9dbc-4a2f-a103-66ebfb80dd6e/image.png">

**_Geolocalization_**

- Another behavior we noticed during the development of CLIP was its ability to recognize many places and locations.

<img src="https://velog.velcdn.com/images/heayounchoi/post/ab1383cf-78eb-44a8-b808-5bf6f1850ace/image.png">

**_Robustness to Distribution Shift_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/ec20ff07-979d-4469-ab9f-116bd238efa9/image.png">
