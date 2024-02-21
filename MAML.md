# Meta-Learning

**Model Agnostic Meta-Learning for Fast Adaptation of Deep Networks**

## Abstract

- 본 논문에서는 model-agnostic한 상황에서 적용할 수 있는 meta-learning algorithm을 제안함.
- gradient descent를 사용하여 학습하는 모든 모델들과 호환 가능
- classification, regression, reinforcement learning(RL) 등을 포함한 다양한 learning 문제들에 적용 가능
- meta learning의 목표는 모델을 다양한 learning task들에 대해 학습하여, 적은 수의 학습 sample들만을 사용하더라도 새로운 learning task를 해결할 수 있도록 만드는 것

## Introduction

- key idea: 새로운 task에서 적은 양의 data를 사용하여 한번 혹은 그 이상의 gradient updates를 진행한 parameter가, 해당 task에서 최대한의 performance를 보여줄 수 있도록 model의 initial parameter를 학습하는 것

## Model Agnostic Meta-Learning

- 목표는 빠르게 적응할 수 있는 model을 학습하는 것인데, 이러한 problem setting은 종종 few-shot learning이라고도 불림
- 논문의 아이디어는 “어떠한 internal representations보다도 더 빨리 바뀔 수 있는 representations가 존재할 것”이라는 직관을 바탕으로 함
- 학습 과정:
    1. 태스크 샘플링: 여러 태스크들의 분포에서 태스크를 무작위로 선택. 각 태스크는 일련의 학습 데이터와 검증 데이터를 가지고 있음
    2. Fast Adaptation:
        - Inner Loop: 몇 개의 그래디언트 업데이트 스텝으로, 모델의 파라미터를 선택된 태스크에 최적화함. 초기 파라미터에서 출발하여, 태스크의 손실 함수에 대한 파라미터의 그래디언트를 계산하여 업데이트함
    3. 메타 최적화:
        - Outer Loop: 내부 루프에서 얻어진 태스크별 최적화된 파라미터를 사용하여 초기 파라미터를 메타 최적화함. 이는 모든 태스크에 대한 손실의 합을 최소화하는 방향으로 진행됨
        - 최종 목표: 최적화된 초기 파라미터는 새로운 태스크에 대해 빠르게 적응할 수 있는 범용성을 가지고 있어야 함
    4. Meta-Testing:
        - 새로운, 학습하지 않은 태스크에 대해 메타 학습된 모델을 평가함. 이 단계에서 모델의 빠른 적응력과 일반화 능력을 테스트함

## Species of MAML

### Supervised Regression and Classification

- few-shot learning은 제한된 수의 데이터 포인트를 통해 학습하는 것을 목적으로 함
- 목표는 모델이 이전에 비슷한 종류의 태스크로부터 얻은 데이터를 활용하여 새로운 태스크를 효과적으로 해결할 수 있게 하는 것
- 이 과정에서 메타러닝이 사용되며, 모델은 적은 수의 input-output 쌍을 가지고도 원하는 태스크를 수행할 수 있도록 학습함
- classification 예시: 강아지 사진 분류를 한다고 했을 때, 모델이 몇 개의 강아지 사진만 보고도 강아지의 종류를 정확히 분류할 수 있도록 학습. 이전에 모델은 다양한 객체들을 많이 보았기 때문에 새로운 강아지 사진에 대해서도 잘 분류할 수 있음
- regression 예시: 회귀에서는 연속적인 함수를 학습하는 것을 목적으로 함. 모델은 비슷한 통계적 특성을 가진 여러 함수들로부터 학습을 진행한 후, 새로운 함수에 대해 몇 개의 데이터 포인트만 보고도 해당 함수의 출력값을 예측할 수 있어야 함
- 메타러닝에서의 적용:
    - 메타러닝에서는 이러한 few-shot learning 문제에 접근하기 위해 메타러닝의 개념을 적용함
    - H=1은 각 태스크에서 단일 관측치만 사용한다는 것을 의미함. 예를 들어, 강아지 사진 분류에서는 한 장의 사진만 보고 종류를 분류해야 함
    - 연속된 데이터가 아닌 개별 데이터 포인트에 대해 처리하기 때문에 각 태스크는 독립적이고, 이전 데이터의 시퀀스가 영향을 미치지 않음. 따라서 timestep이라는 개념이 적용되지 않음
    - 각 태스크는 동일하게 분포된 관측치를 생성하고, 태스크의 손실은 모델의 출력과 목표 값 사이의 오차로 정의됨
    - supervised classification 및 regression에 가장 자주 사용되는 두 가지 loss function은 cross-entropy와 MSE.

### Reinforcement Learning

- reinforcement learning: agent가 환경과 상호작용하며 시행착오를 통해 최적의 전략(policy)을 학습하는 과정. 이 에이전트는 reward를 최대화하는 방식으로 행동을 결정하게 됨. RL의 핵심 목표는 장기적으로 가장 높은 보상을 얻을 수 있는 방법을 찾는 것
- few-shot meta learning은 RL의 개념을 확장해, 에이전트가 매우 적은 수의 경험만으로 새로운 태스크에 빠르게 적응할 수 있도록 하는 것을 목표로 함
- ex) 새로운 게임이나 작업 환경에 처음 도입되는 에이전트가 이전에 비슷한 게임이나 작업에서 얻은 지식을 활용해 매우 적은 시도 또는 경험 후에도 효과적으로 작업을 수행하는 것
- 전통적인 RL은 수많은 시도와 오랜 학습 시간을 요구함