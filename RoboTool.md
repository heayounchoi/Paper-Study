### [Creative Robot Tool Use with Large Language Models](https://arxiv.org/pdf/2310.13065)

**Abstract**
- This paper investigates the feasibility of imbuing robots with the ability to creatively use tools in tasks that involve implicit physical constraints and long-term planning.
- Leveraging LLMs, we develop RoboTool, a system that accepts natural language instructions and outputs executable code for controlling robots in both simulated and real-world environments.
- RoboTool incorporstes four components: (i) Analyzer, (ii) Planner, (iii) Calculator, (iv) Coder.
- Results show that RoboTool can not only comprehend explicit or implicit physical constraints and environmental factors but also demonstrate creative tool use.
- Unlike traditional Task and Motion Planning (TAMP) methods that rely on explicit optimization, our LLM-based system offers a more flexible, efficient, and user-friendly solution for complex robotics tasks.
---

**Introduction**
- TAMP methods with LLMs are able to bypass the computation burden of the explicit optimization process in classical TAMP.
- It is still unclear how to use LLMs to solve more complex tasks that require reasoning with implicit constraints imposed by the robot's embodiment and its surrounding physical world.

<img src="https://velog.velcdn.com/images/heayounchoi/post/4af5bb21-25db-4f48-a0cd-16a471ead279/image.png">

- By providing LLMs with adequate numerical semantic information in natural language, we observe that LLMs can identify the activated constraints induced by the spatial layout of objects in the scene and the robot's embodiment limits, suggesting that LLMs may maintain knowledge and reasoning capability about the 3D physical world.

<img src="https://velog.velcdn.com/images/heayounchoi/post/c698e398-370b-4322-bfa3-273e18e8bca4/image.png">

- key contributions of this paper:
> - RoboTool: a creative robot tool user based on pretrained LLMs that can solve long-horizon hybrid discrete-continuous planning problems with environment- and embodiment-related constraints in a zero-shot manner
> - evaluation benchmark to test tool selection, sequential tool use, and tool manufacturing

---

**Related Works**

_Language Models for Task and Motion Planning (TAMP)_
- Most of the literature built upon hierarchical planning, where LLMs only provide a high-level plan that invokes human-engineered control primitives or motion planners.
- This work follows the hierarchical planning setting and aim to develop an LLM-based planner to solve tasks with constraints that require creative tool-use behaviors.
- Previous works grounded an LLM planner with a real-world affordance function, which requires extra training from massive offline data or domain-specific knowledge, to propose feasible and appropriate plans.
- This work relies entirely on LLM's capability of deriving the affordance from the language input and do not require separate pretrained affordance functions.

_Robot Tool Use_
- 이전에는 GPT-4가 없었다,, 로봇이 머리가 없었다

---

**Methodology**

**_Problem Formulation_**

_Language Description as Input_

_Hierarchical Policies for Robot Tool Use_

**_RoboTool: Creative Robot Tool Use with Large Language Models_**

_Analyzer_

<img src="https://velog.velcdn.com/images/heayounchoi/post/25993be9-0c45-4f40-93c5-f6b8afcabb0f/image.png">

_Planner_

<img src="https://velog.velcdn.com/images/heayounchoi/post/d8e40cf3-487b-45d7-8faf-bdc8b91e529b/image.png">

<img src="https://velog.velcdn.com/images/heayounchoi/post/85e1574f-c629-4910-a53d-c49f6907dd10/image.png">

_Calculator_

_Coder_

---

**Creative Robot Tool Use Benchmark**

_Tool selection_

_Sequential tool use_

_Tool manufacturing_

---

**Experiment Results**

**_Experiment Setup_**

_Robotic Arm_

_Quadrupedal Robot_

**_Baselines_**

_Coder_

_Planner-Coder_

_RoboTool without Analyzer_

_RoboTool without Calculator_

**_Can RoboTool Achieve Creative Tool Use?_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/eeafc3ae-b9ca-4977-8954-a8274747c2d8/image.png">

- RoboTool's performance in the real world drops by 0.1 in comparison to the simulation result, mainly due to the perception errors and execution errors associated with parameterized skills, such as the quadrupedal robot falling down the soft sofa.

_Prior Knowledge_

_Long-horizon Planning_

_Hidden Mechanism Identification_

**_Error Breakdown_**

<img src="https://velog.velcdn.com/images/heayounchoi/post/07cd4755-7803-450c-aece-2d35fd834190/image.png">

**_How Does Analyzer Affect the Tool-Use Capability?_**

_Key Concept Identification Accuracy_

_Discriminative Tool-use Capability_
- using tools when necessary and ignoring tools when the robot can directly finish tasks without the need to manipulate other objects

---

**Conclusion**

_Limitations_

---

**Additional Experiment Results**

<img src="https://velog.velcdn.com/images/heayounchoi/post/4c791d4a-42e9-42d2-b55a-c31eb49c3645/image.png">

- logical error랑 numerical error가 좀 있음
- logical error focuses on planning error, such as using tools in the wrong order or ignoring the constraints provided
- numerical error includes calculating the wrong target positions or adding incorrect offsets (Calculator)

---

**Real-World Setup for Quadrupedal Robot**

**_move-to-position_**
- robot navigates to the target position from its current location, avoiding obstacles present in the scene

**_push-to-position_**
- robot pushes an object to the target location following this sequence:
- 1) Rotate Object
- 2) Push along y-axis
- 3) Push along x-axis

**_climb-to-position_**
- robot climbs to the desired location

**_get-position_**
- position of each object is estimated using AprilTags affixed to them

**_get-size_**
- bboxes of the objects are pre-measured and stored in a database

---

**Real-World Setup for the Robotic Arm**
- We assume the graspable point of each object is given to RoboTool.
- In this work, we focus on the high-level planning capability of LLMs rather than the low-level grasping policy.

**_move-to-position_**
- Milk-Reachiing: Due to the geometric features of the object hammer, which its center does not represent the grasping point of the object, we added an object-specific offset in both x and y axes to the motion planner when grasping the hammer.
- Can-Grasping: Under the object settings, we have pre-scripted a collision-free path given the target position.
- Button-Pressing: For the magnetic cube geometrics, only the flat surface can be attached firmly.

**_get-position_**

**_get-size_**

**_open & close gripper_**
