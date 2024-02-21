# Multi-level Compositional Reasoning for Interactive Instruction Following
[Introduction]
- 연관 분야인듯: visual navigation, object interaction, interactive reasoning
- reasoning for navigation: needs to detect navigable space and explore to reach a target location
- reasoning for object interaction: requires detecting objects and analysing their distances and states
- human cognition process learns to divide a task into sub-objectives, which enables humans to facilitate complex reasoning
- 논문에서 소개할 것:
    - MCR-Agent:
        - 1) a policy composition controller (PCC) that specifies a sub-policy sequence
        - 2) a master policy (MP) that specialises in navigation
        - 3) a set of interaction policies (IP) that execute interactions
    - obejct encoding module (OEM): provides target object information which is used as a navigational subgoal monitor

[Related Work]
- CNN-LSTM based baseline agent with progress tracking
- modular strategy for factorising action prediction and mask generation
- system that encodes language and visual state, and performs action prediction using independently trained modules
- transformer-based hierarchical agent
- transformer-based agent that uses object landmarks for navigation
- transformer-based agent that uses a multimodal transformer for exploiting the multiple input modalities
- constructing semantic maps and leveraging the relative localization for improved navigation
- using 3D map to encode spatial semantic representation
- SLAM-based approach that keeps observed information in a 2D top-down map
- planning-based approach that keeps a semantic spatial graph to encode visual inputs and the agent’s poses
- modular policy with two levels of hierarchy (does not perform well on a long-horizon task)
- this paper’s policy operates at three hierarchical levels, exploiting the fact that navigation and interaction are semantically diverse activities that require independent processing

[Model]
- navigation needs to reason about the temporal history and global environment information
- interaction with objects requires focusing on local visual cues for precise object localization
- three levels of compositional learning:
    - 1) high-level policy composition controller (PCC) that uses language instructions to generate a sequence of sub-objectives
    - 2) master policy that specialises in navigation and determines when and where the agent is required to perform interaction tasks
    - 3) interaction policies (IP) that are a collection of subgoal policies that specialise in precise interaction tasks
{Policy Composition Controller}
- 단계별 instruction에 대해 subgoal을 예측함
- progress를 track 할 수 있음
- first encode the language instructions with a Bi-LSTM, followed by a self-attention module
- each encoded step-by-step language instruction is used as input for the PCC to generate the subgoal sequences
- PCC module은 imitation learning으로 학습시킨다고 함
{Master Policy}
- not only performs navigation but simultaneously also marks the locations for object interaction along the way
- navigation actions와 manipulate의 distribution을 학습해서 environment를 navigate 함
- comprises of two modules:
    - 1) an object encoding module that provides information about the object the agent needs to locate for interaction
    - 2) a navigation policy that outputs the navigation action sequence based on the multi-modal input for traversing the environment
(Subtask Language Encoding)
- regard the subtask instruction as a combination of (1) navigation to discover the relevant object and (2) corresponding interactions
- encode instruction combinations in a similar manner as PCC
(Object Encoding Module (OEM))
- input: subtask language instruction
- output: target object
- acts as a navigation subgoal monitor that indicates the end of the navigation subgoal and shifts control to the next interaction policy
- composed of Bi-LSTM with a two-layer perceptron which ouputs the object class
- during navigation, subgoal monitor uses a pretrained object detector that validates if the relevant object is present in the current view or not
(Navigation Policy)
- uses visual features, subtask instruction features, object encoding, and the embedding of the preceding time step action as inputs
- goal is to locate the correct object for interaction
(Loop Escape)
- subgoal progress monitor and overall progress monitor to train the navigation policy
- heuristic loop escape module for escaping the deadlock conditions
{Interaction Policy}
- to abstract a visual observation to a consequent action, the agent requires a global scene-level comprehension of the visual observation
- for localisation task, the agent needs to focus on both global and local object-specific information
- due to the constrasting nature of the two tasks, seperate streams for action prediction and object localization are exploited
    - 1) action policy module: responsible for predicting the sequence of actions corresponding to the interaction subgoal
    - 2) interaction preception module: generates the pixel-level segmentation mask for objects that the agent needs to interact with at a particular time step
- each interaction has its own properties and the navigation information history is irrelevant to the task. therefore, for each interaction subgoal, a hidden state can be kept isolated
