### [BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/pdf/1810.04805)

**Abstract**
- language representation model
- Bidirectional Encoder Representations from Transformers
- designed to pre-train deep bidirectional representations from unlabeled text by jointly conditioning on both left and right context in all layers
- pre-trained BERT model can be fine-tuned with just one additional output layer to create SOTA models for a wide range of tasks, such as question answering and language inference, without substantial task-specific architecture modifications
---

**Introduction**
- two existing strategies for applying pre-trained language representations to downstream tasks: feature-based and fine-tuning
- The feature-based approach, such as ELMo, uses task-specific architectures that include the pre-trained representations as additional features.
- The fine-tuning approach, such as the Generative Pre-trained Transformer (OpenAI GPT), introduces minimal task-specific parameters, and is trained on the downstream tasks by simply fine-tuning all pre-trained parameters.
- They both use unidirectional language models to learn general language representations.
- current techniques restrict the power of the pre-trained representations, especially for the fine-tuning approaches
- unidirectional can be harmful when applying fine-tuning based approaches to token-level tasks such as question answering, where it is crucial to incorporate context from both directions
- BERT alleviates the previously mentioned unidirectionality constraint by using a "masked language model" (MLM) pre-training objective.
- The masked language model randomly masks some of the tokns from the input, and the objective is to predict the original vocabulary id of the maksed word based only on its context.
- In addition to the masked language model, we also use a "next sentence prediction" task that jointly pre-trians text-pair representations.
---

**Related Work**

**_Unsupervised Feature-based Approaches_**

**_Unsupervised Fine-tuning Approaches_**

**_Transfer Learning from Supervised Data_**

---

**BERT**

<img src="https://velog.velcdn.com/images/heayounchoi/post/6dd8cb45-cb9c-4602-bbda-babcb6b77373/image.png">

- During pre-training, the model is trained on unlabeled data over different pre-training tasks.
- For fine-tuning, the BERT model is first initialized with the pre-trained parameters, and all of the parameters are fine-tuned using labeled data from the downstream tasks.

_Model Architecture_
- multi-layer bidirectional Transformer encoder

_Input/Output Representations_
- To make BERT handle a variety of down-stream tasks, our input representation is able to unambiguously represent both a single sentence and a pair of sentences in one token sequence.
- Throughout this work, a "sentence" can be an arbitrary span of contiguous text, rather than an actual linguistic sentence.
- A "sequence" refers to the input token sequence to BERT, which may be a single sentence or two sentences packed together.

<img src="https://velog.velcdn.com/images/heayounchoi/post/eb3adf0c-c290-4b79-9e00-40c87a3f8e50/image.png">

**_Pre-training BERT_**
- pre-train BERT using two unsupervised tasks

_Task #1: Masked LM_
- Intuitively, it is reasonable to believe that a deep bidirectional model is strictly more powerful than either a left-to-right model or the shallow concatenation of a left-to-right and a right-to-left model.
- simply mask some percentage of the input tokens at random, and then predict those masked tokens (Masked LM / MLM)
- In contrast to denoising auto-encoders, we only predict the masked words rather than reconstructing the entire input.
- since MASK token does not appear during fine-tuning, we replace the i-th token with (1) MASK token 80% of the time (2) a random token 10% of the time (3) the unchanged i-th token 10% of the time

_Task #2: Next Sentence Prediction (NSP)_
- In order to train a model that understands sentence relationships, we pre-train for a binarized next sentence prediction task that can be trivially generated from any monolingual corpus.
- C is used for next sentence prediction (NSP)
- in prior work, only sentence embeddings are transferred to down-stream tasks, where BERT transfers all parameters to initialize end-task model parameters

_Pre-training data_
- It is critical to use a document-level corpus rather than a shuffled sentence-level corpus in order to extract long contiguous sequences.

**_Fine-tuning BERT_**
- For each task, we simply plug in the task-specific inputs and outputs into BERT and fine-tune all the parameters end-to-end.
- Compared to pre-training, fine-tuning is relatively inexpensive.
- All of the results in the paper can be replicated in at most 1 hour on a single Cloud TPU, or a few hours on a GPU, starting from the exact same pre-trained model.
---

**Experiments**

**_GLUE_**
- The General Language Understanding Evaluation (GLUE) benchmark is a collection of diverse natural language understanding tasks.
- To fine-tune on GLUE, we represent the input sequence (for single sentence or sentence pairs), and use the final hidden vector C corresponding to the first input token (CLS) as the aggregate representation.
- The only new parameters introduces during fine-tuning are classification layer weights W, where K is the number of labels.
- We compute a standard classification loss with C and W.

<img src="https://velog.velcdn.com/images/heayounchoi/post/c842a785-b658-42d9-825f-dae37d38334a/image.png">

- MNLI (Multi-Genre Natural Language Inference): premise와 hypothesis 문장을 비교해서 hypothesis가 premise를 지지하는지, 반대하는지, 중립인지를 분류
- QQP (Quora Question Pairs): 두 질문 사이의 의미적 유사성을 결정
- QNLI (Question Natural Language Inference): 질문과 문답이 상호 작용하여 추론
- SST-2 (Stanford Sentiment Treebank): 각 문장이 긍정적인지 부정적인지 classification
- CoLA (Corpus of Linguistic Acceptability): 주어진 문장이 올바른 문법과 의미 구조를 가지고 있는지 평가
- STS-B (Semantic Textual Similarity Benchmark): 두 문장 사이의 의미적 거리 측정
- MRPC (Microsoft Research Paraphrase Corpus): 문장이 서로 동의어로 재구성되었는지 판별
- RTE (Recognizing Textual Entailment): 전제가 가설을 지지하는지, 반대하는지, 중립인지 결정

**_SQuAD v1.1_**
- The Stanford Question Answering Dataset is a collection of 100k crowd-sourced question/answer pairs.
- Given a question and a passage from Wikipedia containing the answer, the task is to predict the answer text span in the passage.
- We only introduce a start vector S and an end vector E during fine-tuning.
- The probability of word i being the start of the answer span is computed as a dot product between $$T_i$$ and S followed by a softmax over all of the words in the paragraph.
- The analogous formula is used for the end of the answer span.

<img src="https://velog.velcdn.com/images/heayounchoi/post/3b1a2039-321b-4ce4-a0eb-155ec55dc3a2/image.png">

- EM (Exact Match) score: 정확한 일치를 나타내는 지표
- F1 score: precision과 recall의 조화 평균

**_SQuAD v2.0_**
- The SQuAD 2.0 task extends the SQuAD 1.1 problem definition by allowing for the possibility that no short answer exists in the provided paragraph, making the problem more realistic.
- We treat questions that do not have an answer as having an answer span with start and end at the CLS token.
- For prediction, we compare the score of the no-answer span to the score of the best non-null span.

<img src="https://velog.velcdn.com/images/heayounchoi/post/a3cc47ce-4cdb-40b5-ad65-b7e4ff2e42cf/image.png">

**_SWAG_**
- The Situations With Adversarial Generations (SWAG) dataset contains 113k sentence-pair completion examples that evaluate grouded common-sense inference.
- Given a sentence, the task is to choose the most plausible continuation among four choices.
- When fine-tuning, we construct four input sequences, each containing the concatenation of the given sentence and a possible continuation.
- The only task-specific parameters introduces is a vector whose dot product with the CLS token representation C denotes a score for each choice which is normalized with a softmax layer.

<img src="https://velog.velcdn.com/images/heayounchoi/post/caeb9260-d2c6-4682-8704-bd3aee53e4ce/image.png">

---

**Ablation Studies**

**_Effect of Pre-training Tasks_**

_No NSP_

_LTR & No NSP_
- left-context-only model which is trained using a standard Left-to-Right (LTR) LM, rather than an MLM

<img src="https://velog.velcdn.com/images/heayounchoi/post/a581f8ce-0710-4ca8-943d-840e8340da3c/image.png">

- compared to ELMo: (a) twice computationally expensive than a single bidirectional model; (b) RTL model would not be able to condition the answer on the question; (c) deep bidirectional model can use both left and right context at every layer

**_Effect of Model Size_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/e9286874-82f4-40a4-8d7e-d880fc21e202/image.png">

- scaling to extreme model sizes also leads to large improvements on very small scale tasks, provided that the model has been sufficiently pre-trained
- we hypothesize that when the model is fine-tuned directly on the downstream tasks and uses only a very small number of randomly initialized additional parameters, the task-specific models can benefit from the larger, more expressive pre-trained representations even when downstream task data is very small

**_Feature-based Approach with BERT_**
- feature-based approach: fixed features are extracted from the pre-trained model
- adv of feature-based approach: (1) not all tasks can be easily represented by a Transformer encoder architecture, and therefore require a task-specific model architecture to be added (2) there are major computational benefits to pre-compute an expensive representation of the training data once and then run many experiments with cheaper models on top of this representation

<img src="https://velog.velcdn.com/images/heayounchoi/post/c55741fa-4ea0-4a28-b06a-4aa2b6732378/image.png">

---

**Conclusion**
- Recent empirical improvements due to transfer learning with language models have demonstrated that rich, unsupervised pre-training is an integral part of many language understanding systems.
- In particular, these results enable even low-resource tasks to benefit from deep unidirectional architectures.
- Our major contribution is further generalizing these findings to deep bidirectional architectures, allowing the same pre-trained model to successfully tackle a broad set of NLP tasks.
---

**Additional Details for BERT**

**_Illustration of the Pre-training Tasks_**

_Maksed LM and the Masking Procedure_
- Compared to standard language model training, the masked LM only make predictions on 15% of tokens in each batch, which suggests that more pre-training steps may be required for the model to converge.
- MLM does converge marginally slower than a left-to-right model (which predicts every token), but the empirical improvements of the MLM model far outweigh the increased training cost.

_Next Sentence Prediction_

<img src="https://velog.velcdn.com/images/heayounchoi/post/ecbed658-6b48-42d7-88fc-2d11d1501521/image.png">

**_Pre-training Procedure_**

**_Fine-tuning Procedure_**
- We observed that large data sets were far less sensitive to hyperparameter choice than small data sets.

**_Comparison of BERT, ELMo, and OpenAI GPT_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/d3136ec4-9a01-43e9-889e-3c85a34a0966/image.png">

**_Illustrations of Fine-tuning on Different Tasks_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/5fa1cf97-79ac-4354-aa72-3288f87650a4/image.png">

---

**Detailed Experimental Setup**

**_Detailed Descriptions for the GLUE Benchmark Experiments_**

_MNLI_

_QQP_

_QNLI_

_SST-2_

_CoLA_

_STS-B_

_MRPC_

_RTE_

_WNLI_

---

**Additional Ablation Studies**

**_Effect of Number of Training Steps_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/d50f3ce2-6382-447b-b3c5-a5093b27f2c1/image.png">

**_Ablation for Different Masking Procedures_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/74cde8fe-e4ab-4ded-b9cf-c7f8684d0604/image.png">

