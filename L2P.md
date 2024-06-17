### [Learning to Prompt for Continual Learning](https://arxiv.org/pdf/2112.08654)

**Abstract**
- The mainstream paradigm behind continual learning has been to adapt the model parameters to non-stationary data distributions, where catastrophic forgetting is the central challenge.
- Typical methods rely on a rehearsal buffer or know task identity at test time to retrieve learned knowledge and address forgetting, while this work presents a new paradigm for continual learning that aims to train a more succinct memory system without accessing task identity at test time.
- Our method learns to dynamically prompt (L2P) a pre-trained model to learn tasks sequentially under different task transitions.
- In our proposed framework, prompts are small learnable parameters, which are maintained in a memory space.
- The objective is to optimize prompts to instruct the model prediction and explicitly manage task-invariant and task-specific knowledge while maintaining model plasticity.
- Surprisingly, L2P achieves competitive results against rehearsal-based methods even without a rehearsal buffer and is directly applicable to challenging task-agnostic continual learning.
---

**Introduction**
- Contrary to ordinary supervised learning that trains on independent and identically distributed data, continual learning tackles the problem of training a single model on non-stationary data distributions where different classification tasks are presented sequentially.
- However, since the model only has access to the current data in an individual phase of the learning cycle, it is prone to overfit on the currently available data and suffers from performance deterioration on the previously trained data due to catastrophic forgetting.
- motivated by the episodic memory in the hippocampus according to the Complementary Learning Systems (CLS) theory, many SOTA methods rely on a rehearsal buffer to re-train a portion of past examples
- However, they suffer from substantial performance deterioration with smaller buffer size and become ineffective when a rehearsal buffer is not allowed - for example, in real-world scenarios where data privacy matters.
- This suggests that simply buffering past data and re-train the model may not be the best approach to retrieve past knowledge.
- Without accessing a rehearsal buffer, another branch of works bypass the forgetting issue by assuming known task identity at test time, so that they are able to attach task-independent modules to the shared model for inference.
- However, knowing task identity at test time restricts practical usage.
- The limitations of prior work bring up critical questions in continual learning:
- (1) Whether the form of episodic memory can go beyond buffering past data to more intelligent and succinct episodic memory system?
- (2) How to automatically select relevant knowledge component for arbitrary sample without knowing its task identity?
- for the first question,
- Prompting techniques design model textual inputs with templated or learnable prompt tokens containing additional task-specific information, such that the pre-trained language model can process parameterized inputs in order to perform prompt-specific prediction.
- Intuitively, prompt-based learning reformulates learning downstream tasks from directly adapting model weights to designing prompts that "instruct" the model to perform tasks conditionally.
- A prompt encodes task-specific knowledge and has the ability to utilize pre-trained frozen models more effectively than ordinary fine-tuning.
- Thus, it is promising to leverage prompts to learn knowledge, and further store learned knowledge, in the continual learning context.
- for the second question,
- it is not clear how to apply prompting
- if we train different prompts for different tasks in the continual learning context, test-time task identity is still required for making predictions using an appropriate task-specific prompt
- as a transfer learning technique, the target of prompting is to make frozen pre-trained models achieve good performance on down-streaming individually, not sequentially
- if we instead maintain a single shared prompt for all tasks, the problem of catastrophic forgetting may still exist

<img src="https://velog.velcdn.com/images/heayounchoi/post/75d06b32-7edf-4641-b15a-89901af982f1/image.png">

- we propose a new continual learning method called Learning to Prompt for Continual Learning (L2P), which is orthogonal to popular rehearsal-based methods and applicable to practical continual learning scenarios without known task identity or boundaries
- L2P leverages the representative features from pre-trained models; however, instead of tuning the parameters during the continual learning process, L2P keeps the pre-trained model untouched, and instead learns a set of prompts that dynamically instruct models to solve corresponding tasks.
- Specifically, the prompts are structured in a key-value shared memory space called the prompt pool, and we design a query mechanism to dynamically lookup a subset of task-relevant prompts based on the instance-wise input features.
- The prompt pool, which is optimized jointly with the supervised loss, ensures that shared prompts encode shared knowledge for knowledge transfer, and unshared prompts encode task-specific knowledge that help maintain model plasticity.
- Our design explicitly decouples shared and task-specific knowledge, thus largely reducing the interference between task-specific knowledge during optimization, leading to minimal catastrophic forgetting without the necessity of a rehearsal buffer.
- The instance-wise query mechanism removes the necessity of knowing the task identity or boundaries, enabling the most challenging, yet under-investigated task-agnostic continual learning.

<img src="https://velog.velcdn.com/images/heayounchoi/post/7163a04c-f155-4d8d-87be-a40748d05732/image.png">

- The selected prompts are then prepended to the input embeddings, which implicitly add task-relevant instruction to pre-trained models, so that the model recalls the most relevant features to conduct corresponding tasks.

---

**Related Work**

_Continual learning_
- three main categories: regularization-based, rehearsal-based, architecture-based
- regularization-based
- limit the plasticity of the model by limiting the learning rate on important parameters for previous tasks
- cannot get satisfactory performance under challenging settings or complex datasets
- rehearsal-based
- construct a data buffer to save samples from older tasks to train with data from the current task
- the performance of rehearsal-based methods generally deteriorates with smaller buffer size, and rehearsal-based methods are eventually not applicable to scenarios where data privacy should be taken into account
- Different from directly saving data from past knowledge to re-train the model, our method stores past knowledge in small learnable prompt parameters to instruct the model to deal with current task, and in turn accumulate current knowledge to the prompts.
- architecture-based
- aims at having separate components for each task
- most methods, which require task identity to condition the network at test-time, are not applicable to more realistic class-incremental and task-agnostic settings when task identity is unknown
- L2P does not require test-time task identity and only adds negligible amount of additional parameters

_Prompting for transfer learning_
- The high-level idea of prompting is to apply a function to modify the input text, so that the language model gets additional information about the task.
- Prompts capture task-specific knowledge with much smaller additional parameters, than its competitors, such as Adapter and LoRA.

---

**Prerequisites**

**_Continual learning protocols_**

- Different from class-incremental, task-incremental learning assumes task identity is known at test time and are often regarded as the simplest setting.
- Different from task- and class- incremental settings where each task has different classes, domain-incremental learning maintains the same set of classes for every task and only changes the distribution of x by task.
- In the more challenging task-agnostic setting, task data in D changes smoothly, and the task identity t is unknown.

**_Prompt-based learning and baselines_**

- In contrast to traditional supervised fine-tuning, this type of methods design task-specific prompt functions to instruct pre-trained models perform corresponding tasks conditionally.

---

**Learning to Prompt (L2P)**

**_From prompt to prompt pool_**

- motivations
- (1) the task identity at test time is unknown so training task-independent prompts is not feasible
- (2) even if the task-independent prompt can be known at test time, it prevents possible knowledge sharing between similar tasks
- (3) while the naive way of learning a single shared prompt for all tasks enables knowledge sharing, it still causes severe forgetting issue

**_Instance-wise prompt query_**

- We design a key-value pair based query strategy to dynamically select suitable prompts for different inputs.
- querying prompts is done in an instance-wise fashion, which makes the whole framework task-agnostic, meaning that the method works without needing clear task boundaries during training, nor task identity at test time

_Optionally diversifying prompt-selection_

- in real-world scenarios and experimental datasets, it is quite common that the task transition is discrete and so task boundaries are known at train time
- We find that adding such a prior into our framework can help the model learn better task-specific prompts, especially when tasks have high diversity.
- we propose a simple extension to add task boundary prior, which is optional for L2P

<img src="https://velog.velcdn.com/images/heayounchoi/post/1da050cf-8fe7-4a49-9e37-4891fe1d5be2/image.png">

**_Optimization objective for L2P_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/a99a0f6d-ddaf-4bd4-868a-8f2f0dbd4ae5/image.png">

---

**Experiments**

- we mainly consider (1) the class-incremental setting, where the task identity is unknown during inference; (2) the domain-incremental setting, where the input domain shifts over time; (3) the task-agnostic setting, where there is no clear task boundary

**_Comparing methods_**

_Baseline methods_

_SOTA rehearsal-based methods_

_SOTA architecture-based methods_

_Our methods_

**_Datasets and experimental details_**

_Datasets_

_Evaluation metrics_

_Training details_

**_Main results_**

_Results on class-incremental learning_

<img src="https://velog.velcdn.com/images/heayounchoi/post/3dc9c7e3-acb0-4a54-af48-472e1813766f/image.png">

<img src="https://velog.velcdn.com/images/heayounchoi/post/e0000c2c-30e6-47e8-8a34-a57111939195/image.png">

_Results on domain-incremental learning_

<img src="https://velog.velcdn.com/images/heayounchoi/post/73e67ba6-39e2-4976-8e79-31c6cc492efe/image.png">

_Results on task-agnostic learning_

<img src="https://velog.velcdn.com/images/heayounchoi/post/d030d128-b68c-4bef-9f33-112bbeb5701b/image.png">

- We believe that the smoother transition of tasks implicitly help L2P consolidate knowledge into prompts.

**_Effectiveness of core designs_**

_Effect of prompt related components for L2P_

<img src="https://velog.velcdn.com/images/heayounchoi/post/674af2bb-9f67-45fc-9d7c-4af7134c7473/image.png">

- a single prompt suffers severe catastrophic forgetting and knowledge interference between tasks, while our design of prompt pool encodes task-invariant and task-specific knowledge well
- learnable keys play an important role to decouple the query and prompt learning processes
- removing diversified prompt selection allows instances from different tasks to choose prompts freely
- however when tasks are diverse, adding this strategy indeed reduces unnecessary knowledge sharing and thus mitigating interference between unrelated tasks

<img src="https://velog.velcdn.com/images/heayounchoi/post/c1a25913-0c7c-4850-8c2e-215cb63518c6/image.png">

_Effect of hyperparameters for L2P_

<img src="https://velog.velcdn.com/images/heayounchoi/post/911b7dc9-8ec2-42bc-82ff-834f5cbeed1d/image.png">

- We hypothesize that a reasonable capacity of a single prompt is critical to encode a certain aspect of shared knowledge.
- a large enough pool size is needed to encode task-specific knowledge when tasks are diverse

---

**Conclusion**

---

**Potential negative societal impact**

- Our method takes a well-pretrained model as a backbone, thus any bias and fairness issues in the original model may be carried over during the continual learning process.

---

**Limitations**

- exploration on other modalities
- assumes there are pre-trained sequence-based models
- requires more complex benchmarks to evaluate the ability of task-agnostic continual learning methods

---

**Dataset details and licensing information**

_Split CIFAR-100 (class-incremental)_
- splits the original CIFAR-100 into 10 tasks, 10 disjoint classes per task

_5-datasets (class-incremental)_
- CIFAR-10, MNIST, Fashion-MNIST, SVHN, and notMNIST

_CORe50 (domain-incremental)_
- collection of 50 objects collected in 11 distinct domains, where 8 of them are used for training, and the rest are considered as a single test set

_Gaussian scheduled CIFAR-100 (task-agnostic)_
- The distribution of data shifts gradually throughout the learning process, the probability that a class is present in a batch follows a Gaussian distribution centered with intervals.

---

**Algorithm details**

<img src="https://velog.velcdn.com/images/heayounchoi/post/2ea67245-b34c-441a-b83c-01ec41686e25/image.png">
