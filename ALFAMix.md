### [Active Learning by Feature Mixing](https://arxiv.org/pdf/2203.07034)

**Abstract**

- The promise of active learning (AL) is to reduce labelling costs by selecting the most valuable examples to annotate from a pool of unlabelled data.
- We identify unlabelled instances with sufficiently-distinct features by seeking inconsistencies in predictions resulting from interventions on their representations.
- We construct interpolations between representations of labelled and unlabelled instances then examine the predicted labels.
- We show that inconsistencies in these predictions help discovering features that the model is unable to recognise in the unlabelled instances.
- We derive an efficient implementation based on a closed-form solution to the optimal interpolation causing changes in predictions.

---

**Introduction**

- High quality data annotations can be slow and expensive.
- Active learning (AL) aims to actively select the most valuable samples to be labelled in the training process iteratively, to boost the predictive performance.
- A popular setting called batch AL fixes a budget on the size of the batch of instances to be sent to an oracle for labelling.
- Various AL strategies have been proposed differing in predicting (1) how informative a particular unlabelled instance will be (i.e. uncertainty estimation) or (2) how varied a set of instances will be (i.e. diversity estimation), or both.
- Recent deep learning based AL techniques include, for example, the use of an auxiliary network to estimate the loss of unlabelled instances, the use of generative models like VAEs to capture distributional differences, and the use of graph convolutional networks to relate unlabelled and labelled instances.
- Despite much progress made, current AL methods still struggle when applied to deep neural networks, with high-dimensional data, and in a low-data regime.

<img src="https://velog.velcdn.com/images/heayounchoi/post/4bea5350-e94d-438f-9d00-2b10dcda77d2/image.png">

- We identify informative unlabelled instances by evaluating the variability of the labels predicted for perturbed versions of these instances.
- These perturbed versions are instantiated in feature space as convex combinations of unlabelled and labelled instances.
- This approach effectively explores the neighbourhood surrounding an unlabelled instance by interpolating its features with those of previously-labelled ones.
- under a norm-constraint on the interpolation ratio, we show that the interpolation is equivalent to considering (1) the difference between the features of the unlabelled instance and the labelled ones and (2) the gradient of the model w.r.t the features at the unlabelled point
- Discovering new features considering (1) and (2) leads us to finding an optimal interpolated point deterministically, at a minimal computing cost.
- Rather than using all the labelled data for these interpolations, we choose a subset we call anchors to capture the common features for each class.
- Subsequently, we construct a candidate set by choosing the instances from the unlabelled set that when mixed with these anchors lead to a change in the model's prediction for those instances.
- Then, to ensure selected instances are diverse, we perform a simple clustering in the candidate set and choose their centroids as the points to be queried.

----

**Related Work**

- diversity-based vs uncertainty-based vs hybrid sampling
- Diversity-based approaches aim to select samples that best represent the whole of the available unlabelled set.
- Uncertainty-based methods seek to identify the unlabelled samples that are most ambiguous to the current model that has been trained over the present labelled set based on the target objective function.
- These methods favour points that lie close to the decision boundary, but as they rely entirely on the predicted class likelihoods they ignore the value of the feature representation itself.
- ALFA-Mix interpolates in latent space
- Recently, a series of model-based active learning have been developed whereby a separate model is trained for active instance selection.
- these AL methods do not consider the diversity of the selected samples and are prone to selecting samples with repetitive patterns
- Hybrid AL methods exploit both diversity and uncertainty in their sample selection methodologies.

---

**Methodology**

**_Problem Definition_**

**_Feature Mixing_**

- Our intuition is that the model's incorrect prediction is mainly due to novel "features" in the input that are not recognisable.
- Thus, we approach the AL problem by first probing the features learned by the model.
- To that end, we use a convex combination (i.e. interpolation) of the features as a way to explore novel features in the vicinity of each unlabelled point.

<img src="https://velog.velcdn.com/images/heayounchoi/post/4b7700d6-2e9b-4a2c-b621-697b2bd7e27e/image.png">

- We consider interpolating an unlabelled instance with all the anchors representing different classes to uncover the sufficietnly distinct features by considering how the model's prediction changes.
- For that, we investigate the change in the pseudo-label for the unlabelled instance and the loss incurred with the interpolation.
- We expect that a small enough interpolation with the labelled data should not have a consequential effect on the predicted label for each unlabelled point.
- Using a first-order Taylor expansion w.r.t. z^u, the model's loss for predicting the pseudo-label of an unlabelled instance at its interpolation with a labelled one can be re-written as:

<img src="https://velog.velcdn.com/images/heayounchoi/post/3679552a-3106-49ab-a6a6-def8c0a2e392/image.png">

- Consequently, for the full labelled set, by choosing the max loss from both sides we have:

<img src="https://velog.velcdn.com/images/heayounchoi/post/065c9efc-cca3-487d-af82-f6305cd22b1d/image.png">

- Intuitively, when performing interpolation, the change in the loss is proportionate to two terms: (a) the difference of features of z^* and z^u proportionate to their interpolation a, and (b) the gradient of the loss w.r.t the unlabelled instance.
- The former determines which features are novel and how their value could be different between the labelled and unlabelled instance.
- On the other hand, the latter determines the sensitivity of the model to those features.
- That is, if the features of the labelled and unlabelled instances are completely different but the modele is reasonably consistent, there is ultimately no change in the loss, and hence those features are not considered novel to the model.

**_Optimising the Interpolation Parameter a**

- Since manually choosing a value for a is not-trivial, we devise a simple optimisation approach to choose the appropriate value for a given unlabelled instance.

<img src="https://velog.velcdn.com/images/heayounchoi/post/86f0ea41-b6c3-41cb-930a-80505b162cba/image.png">

- e is a hyper-parameter governing the magnitude of the mixing
- Intuitively, this optimisation chooses the hardest case for a for each unlabelled instance and anchor.

<img src="https://velog.velcdn.com/images/heayounchoi/post/77c2b01e-e1d2-4c3a-828c-b4fce1a46513/image.png">

- This approximation makes the optimisation of the interpolation parameter efficient and our experiments show that it will not have significant detrimental effects on the final results compared to directly optimising for a to maximize the loss.

**_Candidate Selection_**

- For AL it is reasonable to choose instances to be queried whose loss substantially change with interpolation according to Eq. (3).

<img src="https://velog.velcdn.com/images/heayounchoi/post/712c524f-bd2b-4178-b9c8-360599ae442b/image.png">

<img src="https://velog.velcdn.com/images/heayounchoi/post/6bf43069-3c2b-4eaf-8a4f-c287e34a0406/image.png">

- The size of the selected set I could potentially be larger than the budget B.
- In addition, ideally we seek diverse samples since most instances in I could be chosen from the same region (i.e. they might share the same novel features).
- To that end, we propose to cluster the instances in I into B groupd based on their feature similarities and further choose the closest samples to the centre of each cluster to be labelled by oracle.

<img src="https://velog.velcdn.com/images/heayounchoi/post/614bd641-2b14-4c9c-9d8f-708f7f62be63/image.png">

---

**Experiments and Results**

**_Baselines_**

- Random, Entropy, BALD, Coreset, Adversarial Deep Fool, GCNAL, BADGE, CDAL

**_Experiment Settings_**

_Setting and Datasets_

<img src="https://velog.velcdn.com/images/heayounchoi/post/00fcfbb4-0ada-4ce7-b886-220ae6e1941c/image.png">

_Video classification_

_Interpolation optimisation_

**_Overall Results_**

_Image and non-image results_

<img src="https://velog.velcdn.com/images/heayounchoi/post/8cd47042-4120-4a13-bafa-04bbd5ea543e/image.png">

_Video Classification results_

<img src="https://velog.velcdn.com/images/heayounchoi/post/8a1f622e-fb45-4a98-8dac-4b751dfd1087/image.png">

**_Ablation Study_**

_Learning Ablations_

_Diversification_

<img src="https://velog.velcdn.com/images/heayounchoi/post/505cfae1-ae2f-43c1-87cd-bedef1c58b89/image.png">

_Learning the Interpolation Parameter_

_Anchors_

_Acquisition Time_

<img src="https://velog.velcdn.com/images/heayounchoi/post/9a9f1272-2685-4d9e-8554-0fca9b96f3d6/image.png">

- we only back-propagate to a latent representation layer (not the whole network)

---

**Conclusions and Limitations**

---

appendix 읽어야함
