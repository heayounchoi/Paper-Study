### [RoboMP^2: A Robotic Multimodal Perception-Planning Framework with Multimodal Large Language Models](https://arxiv.org/pdf/2404.04929)

**Abstract**
- RoboMP2 is a framework for robotic manipulation which consists of a Goal-Conditioned Multimodal Perceptor (GCMP) and a Retrieval-Augmented Multimodal Planner (RAMP).
- GCMP captures environment states by employing a tailored MLLMs for embodied agents with the abilities of semantic reasoning and localization.
- RAMP utilizes coarse-to-fine retrieval method to find the k most-relevant policies as in-context demonstrations to enhance the planner.
---

**Introduction**
- end-to-end policies and human-selected prompts lack generalization and flexibility on unseen tasks
- Environment perception and task planning are two fundamental components for embodied agents to complete a task.
- Most of the existing work typically employs mainstream vision models as environment perceptors, such as YOLOv5 and CLIP.
- These models work well in simple scenarios where the categories of the objects are pre-defined or the relationships among objects are easy to be captured.
- However, they lack the capability to identify and locate objects in unseen scenarios or objects with intricate spatial relationships.

<img src="https://velog.velcdn.com/images/heayounchoi/post/57e3ddae-08ca-426a-8775-885b53a4b0de/image.png">

- The existing perceptors struggle to accurately identify and locate the specified yellow block since these models cannot understand the semantic information of the complex referring expression.
- Therefore, it necessitates the development of robot perceptors with multimodal understanding and reasoning capabilities.
- MLLMs have shown remarkable semantic understanding and vision perception abilities in various tasks.
- MLLM-based GCMP detects objects with a given semantic-complex reference goal.
- In addition to the multimodal environment perceptor, the planning for subsequent execution is also critical.
- The existing policies mainly include end-to-end models and prompt-based approaches.
- The end-to-end policy integrates the perception and planning into a single model, thus requiring closed-loop robot data.
- However, in the real world, the closed-loop data is very limited due to the expensive human labor for collection.
- Consequently, these models are proven to overfitting the data-collection scenario and show limited generalization in unseen environments or on new tasks.
- On the other hand, current prompt-based methods rely on manually designed and selected prompt templates to prompt LLMs to generate plans for a given task.
- They inherently lack generalization in diversities of tasks that are highly different from the tasks given in the demonstrations of the prompt templates.
- To cover different types of tasks, these methods adopt large amounts of templates, which results in in-context attention distraction.
- These approaches typically generate plans based solely on the text instruction, overlooking critical environmental information.
- To address these issues, we propose a retrieval-augmented method which prompts MLLMs to generate plans via adaptively choosing the most relevant policies as demonstrations.
- main contributions:
- 1) Different from the existing robot perceptors that can only identify objects with pre-defined classes or simple references, GCMP owns the comprehension abilities to perceive targeted objects with complex references.
- 2) Different from the existing code planners that simply generate code based solely on a text instruction with manually selected templates, RAMP integrates multimodal environment information into the code generation process, and develops a retrieval-augment strategy to mitigate the interference of redundant in-context examples.
---

**Related Work**

_MLLMs for Robotic Manipulation_
- Robotic manipulation aims to complete a specific task by interacting with the environment.
- In recent, imitation learning has achieved a great success in robotic learning, but due to the task complexity, it needs large amounts of data to train a robot agent to achieve strong generalization capability.
- Therefore, many efforts have been made on prompting LLMs to generate policies in a zero-shot manner to control robot agent.
- It has been proven that the prompt is crucial when using LLMs to generate text in a zero-shot manner.
- Previous studies merely utilized the textual information to prompt LLMs, ignoring the significant multimodal information of the environment.

_Retrieval Augmented Generation with MLLMs_
- Retrieval augmented generation (RAG) was first introduced to serve as more informative inputs to unlesh the extensive knowledge of LLMs.
- Due to its effectiveness, it was subsequently introduced to the multimodal domain.
- It assists models generate answers by retrieving contents related to the original input as supplement context.
- Considering the plans for robotic manipulation tasks are predominantly involved in the executed actions and target objects, we incorporate a task rewriting module.
- This module is introduced to extract essential textual information from task instructions.
---

**Framework**

<img src="https://velog.velcdn.com/images/heayounchoi/post/3b90a1d7-19d8-4770-afef-154fa9ae1656/image.png">

**_Goal-Conditioned Multimodal Perceptor_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/891a41f5-cf61-4145-bcbe-4bb0cf93145c/image.png">

_Environment Perception for Manipulation_
- existing perceptors cannot understand the instruction semantics or the spatial relationships among different objects
- commonly used reference expressions that can hardly be handled by the existing perceptors:
- 1) Object Perception Based on Attributes: requires the perceptor to have a powerful attribute-aware perception ability for not leaving out any of the referred objects
- 2) Object Perception Based on Spatial Relationships
- 3) Object Perception Based on Knowledge Reasoning:

_Training of Multimodal Perceptor_

<img src="https://velog.velcdn.com/images/heayounchoi/post/289c5487-459a-4ff4-abe2-73b66d953340/image.png">

- input: image, referential expression
- output: coordinates for manipulation

**_Retrieval-Augmented Multimodal Planner_**
- Existing approaches typically prompt LLMs to generate plans according to a textual task instruction and a manually selected prompt template.
- However, they suffer from two issues: (1) utilizing human selected templates for a given instruction while hardly generalizing to new tasks; (2) only using text information while ignoring multimodal environment information.
- To address these challenges, we introduce RAMP which is composed of a coarse retriever, a instruction and a fine reranker to adaptively find the k most-relevant policies as in-context demonstrations, thereby boosting the generalization of the policy planning.

_Coarse Retriever_
- To cover diversities of tasks, a prompt typically includes many in-context examples as demonstrations to prompt LLMs to generate task plans.
- This leads to attention diffusion across these examples.
- coarse retriever finds the most relevant policies from a codebase

<img src="https://velog.velcdn.com/images/heayounchoi/post/9a21dfe6-e906-4d47-a83b-7da3210e764c/image.png">

_Fine Reranker_
- Despite relevant code snippets could be recalled through the coarse retriever, there remains an issue of the order among candidates.
- We introduce a rewriting module and a reranker module respectively to extract its core of the task instruction and order these relevant demonstrations.

_Instruction Rewriter_
- The robotic policy is mainly related to the action type and manipulation objects.
- Thus, we introduce a instruction rewriting module to eliminate distracting expressions from the task description, obtaining its core robotic operation instructions.

<img src="https://velog.velcdn.com/images/heayounchoi/post/22572a96-8b1c-4a2e-bcb2-592d0e6ad963/image.png">

_Semantic Reorder_
- Due to the design of positional encoding in the transformer architecture, researchers have found that model generation tends to primarily focus on the content at the beginning and the end of the paragraph.
- reorder module aims to sort the retrieved candidate code snippets
**- A straightforward solution is to simply concatenate it at the beginning of the whole candidate.** ?

<img src="https://velog.velcdn.com/images/heayounchoi/post/33d9e955-216d-469e-a0b9-211241ca89a0/image.png">

_Multimodal Generation Module_
- After adaptively selecting k most relevant examples through the coarse-to-fine retrieval method, we combine them with a template to construct the complete prompt, including the third-party libraries, function definitions, and the task instruction.
- we use the GPT4V as the multimodal generator with the input 

<img src="https://velog.velcdn.com/images/heayounchoi/post/2e76a936-d1e9-440a-b3db-f9f392252ea2/image.png">

and full prompt template.

<img src="https://velog.velcdn.com/images/heayounchoi/post/3713b745-b710-425e-af1d-0abcb9c3cead/image.png">

<img src="https://velog.velcdn.com/images/heayounchoi/post/99793360-70b2-469e-b0ad-38579c2c9d5f/image.png">

---

**Experiment**

**_Data and Evaluation Metrics_**

**_Baselines_**

_End-to-end models_

_Prompt-based methods_

**_Implementation Details_**

**_Results_**

_Experimental Results on VIMABench_

_Real-world Experimental Results_

**_Ablation Study_**

_Effects of the Coarse-to-fine Retriever_

_Effectiveness of Multimodal Planner_

_Comparison of Multimodal Perceptors_

**_Qualitative Results_**

---

**Conclusion**

---

**Impact Statements**
