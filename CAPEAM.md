# Context-Aware Planning and Environment-Aware Memory for Instruction Following Embodied Agents, CAPEAM

## Abstract
![](https://velog.velcdn.com/images/heayounchoi/post/06ad6db1-956b-446d-a3dd-a1c2282d6a77/image.png)
- 최신의 embodied agents는 전문가를 모방하거나, 지식이 없는 상태로 잘못 훈련돼있어 환경 속에서 움직이거나, 객체와 소통하는 일에 실수를 겪는다.
- CAPEAM을 활용하면 공간의 변화나 객체의 상태 변화의 의미적 맥락을 파악해 다음 행동을 수행할 수 있다.
- 목격한 환경이거나, 처음 목격하는 환경에서 성능이 좋아지는 것을 확인했다.
----
## Introduction
- 공부해볼 분야: navigation, object interaction, interactive reasoning, interactive instruction following
- task를 (1) tast-relevant prediction(context prediction)과 (2) detailed action planning that considers the contextual memory로 나눈다. 여기서 context란, agent가 조작해야할 객체를 의미한다.  <= context-aware planning(CAP)
- 또다른 challenge는 객체의 상태가 변화하면서 agent가 작업 수행에 어려움을 겪는 경우인데, environment-aware memory에 객체의 상태와, 마스킹을 활용한 visual 변화를 저장함으로써 해결될 수 있다 <= environment-aware memory(EAM)
----
## Related Work

### Action Planning
- 기존의 시도: 데이터로 승부하기, 템플릿 만들어두기 -> CAP를 활용해 task와 관련된 객체만 예측해 효율적인 planning 가능
- LLM은 physical grounding이 부족해서 planning을 잘 하더라도 agent가 제대로 수행하지 못할 수 있다.

### Memory for Object Interaction
- 기존의 시도: 객체들의 위치 정보 기억하기(이미 task 수행에 사용된 객체인지에 대한 정보는 저장하지 않음) -> EAM에 interact가 있던 객체인지 기억해서 misinteraction을 줄임
- 기존의 시도: task 수행 중 객체의 형태가 바뀜으로써 생기는 문제를 마스크를 씌워서 해결(형태가 바뀐 객체를 인식하지 못하면 에러) -> 이전의 객체 행태를 기억함으로써 occlusion으로 인한 형태 변화에 영향을 줄임

### Semantic Spatial Representation
- 기존의 시도: 여러 브랜치로 행동과 마스크 예측을 나누고, 각 브랜치가 학습해서 sequence를 만들어내는 식(unseen environment에서 성능이 좋지 않음) -> 3D 환경을 정확하게 인식할 수 있도록 agent의 history를 room layout과 같은 semantic map에 기록
----
## Approach
- CAPEAM은 학습 + 네비게이션 설계 알고리즘 모델에 context-aware planning으로 task context
를 까먹는 문제를 해결하고, environment-aware memory로 객체의 상태를 기억한다.

### Context-Aware Planning
![](https://velog.velcdn.com/images/heayounchoi/post/5f3a8ac3-bd54-4fba-9bf6-569556bb92fc/image.png)

- sub-goal planner 단계에서 sub-goals를 만들고, detailed planner 단계에서 sub-goal을 수행하기 위한 자세한 일을 계획한다.
- sub-goal planner는 context predictor와 sub-goal frame sequence generator로 구성된다. Context predictor는 task와 관련된 객체를 3개 예측하고, sub-goal frame sequence generator는 특정 객체에 국한되지 않는 행동 sub-goal을 만든다.
- 예측된 3개의 객체들은 1) main object to be manipulated, 2) container that holds the object, 3) target object where the object is to be placed in the task로 구성된다.
- 예측된 객체들을 sub-goal frames와 통합해 할일을 계획함으로써 agent는 task를 수행하는 동안 context를 기억할 수 있다.

#### Sub-Goal Planner
  
![](https://velog.velcdn.com/images/heayounchoi/post/37a04a41-5b46-46a6-9771-42fbb06d3e71/image.png)

- l: input
- fsub(): sub-goal planner
- An: human-interpretable action
- On: object targeted for manipulation in the execution of An
- Rn: location where On can be found
- N: number of subgoals in a plan

[Context Prediction]
- BERT 모델을 사용해서 3개의 객체들을 예측한다.

[Sub-Goal Frame Sequence Generator]
- sub-goal frame sequence generator은 task에 집중할 수 있도록 객체의 이름 대신 meta-class를 사용하고, 추후에 meta-class를 context prediction에서 예측한 contexts로 채운다.

#### Detailed Planners
- sub-goal을 sequence of detailed action으로 만드는데, self-attention LSTM을 사용한 지도 학습 방식으로 한다.

### Environment-Aware Memory

[Semantic Spatial Map]
- 예측된 depth map들과 object mask들을 사용해 3D로 back-projecting하여 semantic spatial map을 만든다.
- 과거 environmental information도 저장한다.

[Retrospective Object Recognition]
- 객체와 여러번 interact해서  visual appearance가 바뀌어 recognize 할 수 없는 경우 최근의 mask를 사용해 같은 객체를 인식할 수 있다.

[Object Relocation Tracking]
- 동일한 객체와 여러번 interact 하게 되는 문제를 피하기 위해서 semantic map에 이동된 객체 정보를 삭제한다.
- agent는 memory와 semantic map을 비교함으로써 이미 이동된 객체를 인지할 수 있다.

[Object Location Caching]
- 상태가 변화한 객체의 2D location과 segmentation masks를 메모리에 cache함으로써 객체를 다시 찾기 위해 환경을 explore 할 필요를 줄인다.

### Action Policy
- 확장된 obstacle map의 별개의 공간에 deterministic 알고리즘을 사용한 navigation paths를 계획한다.
----
## Limitation and Future Work
- context가 바뀌면 에러가 날수도 있음(single task execution 사이에도 context는 바뀔 수 있음)
- 환경에서 input을 받아서 context를 수정할 수 있는 방향으로의 개선이 필요함 
<br>
<br>

Reference
[Context-Aware Planning and Environment-Aware Memory for Instruction Following Embodied Agents](https://openaccess.thecvf.com/content/ICCV2023/papers/Kim_Context-Aware_Planning_and_Environment-Aware_Memory_for_Instruction_Following_Embodied_Agents_ICCV_2023_paper.pdf)
