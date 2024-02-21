# Zero-Shot Learning

- 일반적으로 딥러닝은 학습에 사용된 클래스만을 예측할 수 있음
- 따라서 unseen data가 입력되면 seen class로 예측해버림
- zero shot은 train set에 포함되지 않은 unseen class를 예측하는 분야
- unseen data를 입력 받아도, seen data로 학습된 지식을 전이하여 unseen data를 unseen class로 예측할 수 있음
- semantic information를 활용해서 예측할 수 있음
- semantic information은 여러 종류가 될 수 있는데, 한 가지 예시로는 data의 class를 word embedding으로 사용하는 것
- one-hot vector를 class로 주어 학습된 딥러닝 모델은 강아지와 고양이의 관계를 모르지만, class를 word embedding된 word vector로 주어 학습한다면 이 딥러닝 모델은 강아지와 고양이의 관게를 알 수 있음
- zero shot learning은 test 시에 unseen data만을 입력 받아 class를 예측하고, generalized zero shot learning은 test 시에 unseen, seen data를 둘 다 입력 받아 예측을 수행함

## Generalized Zero Shot Learning(GZSL)

(1) Transductive GZSL

- seen, unseen의 visual feature와 semantic information을 모두 활용
- 실제 환경에선 unseen에 대한 visual feature를 얻는 것이 불가능하므로 real world 환경에선 적합하지 않음

(2) Transductive semantic GZSL

- seen의 visual feature, semantic information과 unseen의 semantic information을 활용
- 한 가지 예시로 generative-based methods에서 unseen에 대한 image를 생성해 unseen의 visual feature를 활용하는 방법이 있음
- generative model로 생성한 unseen data를 training set에 넣어 함께 학습시켜 unseen에 대한 예측도 수행할 수 있도록 하는 방법
- 다른 방법으로는 embedding based method인데, semantic embedding space로 mappting 하는 방법
- unseen에 대한 visual feature를 입력 받았을때, seen으로 학습한 embedding space 상에서 적합한 위치에 projection 하게 되는데, 이에 해당하는 semantic information이 unseen class가 되는 것

(3) inductive GZSL

- seen의 visual feature와 semantic information만을 입력 받아 학습

## GZSL Methods
![Untitled](https://github.com/heayounchoi/Paper-Study/assets/118031423/2ae300f6-3326-4da2-84ec-76fb3520cbfa)

## Embedding Space

- GZSL은 seen class의 저차원 visual feature를 그에 해당하는 semantic vector로 mapping/embedding 하는 함수를 학습하는 것으로 볼 수 있음
- visual feature 사이에는 큰 variation이 존재하는데, 강아지와 고양이의 visual feature의 연관성이 적기 때문
- semantic information은 둘의 연관성을 잘 알고 있으므로 variation을 줄일 수 있음

a) visual → semantic embedding

- GZSL이 학습한 함수가 visual space의 vector를 semantic embedding으로 mapping 하는 방법

b) semantic → visual embedding

- semantic vector를 visual embedding으로 mapping 하는 함수를 학습하는 방법

c) latent embedding

- visual feature와 semantic representation을 common space L(latent space)로 projection 하여 동일한 클래스에 해당하는 visual, semantic vector는 common space L 내에서 가깝게 project 하는 함수를 학습하는 방법

## Challenging Issues

(1) hubness problem

- ZSL와 GZSL은 visual feature를 semantic space로 project하고, 고차원 space에서 nearest neighbor search를 하여 unseen에 대한 class를 예측하므로 nearest neighbors에 치명적인 차원의 저주 문제가 발생. 차원의 저주란 차원이 높아질수록 data 사이의 거리가 멀어지는 문제

(2) domain shift problem
![Untitled_1](https://github.com/heayounchoi/Paper-Study/assets/118031423/d0a3a458-71e8-4e05-8500-0dbf2e653880)

- seen class의 data로 ZSL, GZSL의 mapping 함수가 학습되었으므로 unseen class에 대한 adaptation이 없다면 domain shift problem이 발생함. 이 문제는 주로 unseen의 아무런 정보없이 학습하는 inductive based method에서 발생함

(3) biased towards the seen classes

- GZSL 방법은 seen class의 data로 모델을 학습하여 unseen과 seen class를 분류하기 때문에, biased towards the seen classes 문제가 발생함. unseen class의 data를 seen class로 분류하는 것.
