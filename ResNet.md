# Deep Residual Learning for Image Recognition / ResNet

## Abstract
- neural networks가 깊어질수록 성능은 더 좋지만, train은 더 어려워짐.
- 이 논문에서는 잔차를 이용한 잔차학습 (residual learning framework)를 이용해 깊은 신경망에서도 학습이 쉽게 이뤄질 수 있다는 것을 보이고 방법론을 제시함.
- 여기서 residual learning이란 쉽게 말해 이전 레이어의 결과를 다시 이용하는 것. 즉, 입력 레이어를 다시 이용하는 (참조하는) residual function을 사용하여 더 쉬운 최적화와 깊은 네트워크에서의 정확도 향상이 가능했다고 함.
- 결과적으로 152개의 레이어를 쌓아서 VGGNet보다 좋은 성능을 내면서 복잡성은 줄임. 3.57% 에러 달성.

## Introduction
- 심층 신경망은 추상화에 대한 low / mid / high level의 특징을 classifier와 함께 multi-layer 방식으로 통합함. 여기서 각 추상화 레벨은 쌓인 레이어의 수에 따라 더욱 높아질 수 있음. 즉, 높은 추상화 특징은 high layer에서 파악할 수 있다는 것.
![](https://velog.velcdn.com/images/heayounchoi/post/eb3b0c23-484a-4d7b-8515-323de13f4e42/image.png)
- 최근 연구 결과에 따르면 네트워크의 깊이는 매우 중요한데, 실제로 많은 모델들이 깊은 네트워크를 통해 좋은 성능을 보였다고 함.
- 네트워크의 깊이가 중요해지면서, 레이어를 쌓는만큼 더 쉽게 네트워크를 학습시킬 수 있는지에 대한 의문이 생기기 시작했고, 그중에서도 그래디언트 소실/폭발 문제가 큰 방해 요소였음. 하지만 normalization 기법과 intermediate normalization layer로 괜찮아짐.
> #### Intermediate Normalization Layers
> - 모델의 중간에 위치한 레이어로서, 데이터나 활성화 값을 일정한 범위나 분포로 정규화하여 학습을 안정화하고 속도를 높임.
> - ex) Batch Normalization, Layer Normalization, Instance Normalization, Group Normalization, ...
- 하지만 심층 신경망의 경우, 성능이 최고 수준에 도달할 때 degradation problem 발생. test error만이 아닌 training error이 함께 높아짐으로 과적합 때문이 아닌 레이어 수 추가로 인한 문제라는 것을 알 수 있음.
> #### Degradation Problem
> - 깊은 신경망에서는 이론적으로 깊이가 늘어날수록 훈련 오차는 줄어들어야 함. 하지만 실제로는 네트워크의 깊이가 특정 수준을 넘어가면 훈련 손실이 오히려 증가하는 현상이 발생하기도 함.
> - 그래디언트 소실/폭발이나 과적합과는 관련이 없는, 깊이 자체가 학습의 어려움을 가져오는 문제를 의미. 정확한 원인은 밝혀진바 없다고 함.
> #### degradation problem에 대한 가설과 해석
> - vanishing gradients가 성능 저하에 일부 기여할 수 있다는 가설. 하지만 이 문제는 활성화 함수나 초기화 전략의 변경으로 완화될 수 있고, 그럼에도 불구하고 성능 저하 문제는 여전히 발생함.
> - optimization difficulty: 깊은 네트워크의 손실 함수의 표면이 더 복잡하고 까다로울 수 있으므로, 최적화가 더 어려울 수 있다는 가설 존재.
> - feature propagation: 네트워크의 깊이가 증가함에 따라, 중요한 특징들이 네트워크를 통해 효과적으로 전파되지 않을 수 있다는 해석. 
- 이 논문에서는 성능 저하 문제를 optimization difficulty를 원인으로 봄.
- construction으로 최소 얕은 네트워크 정도의 성능을 보이는 건 가능했는데, 추가된 레이어가 아무런 변화나 학습을 하지 않고 단순히 입력을 그대로 전달하는 경우(identity mapping)였음.
- 이 논문에서는 deep residual learning framework를 제시함.
- unreferenced mapping보다 residual mapping에 최적화하는 것이 더 쉬울 것이라고 판단.
- 극한으로 끌고 가서 만약 identity mapping이 최적일 경우, residual을 0으로 만드는 것이 nonlinear layer들로 identity mapping을 하는것보다 쉬울 것.
> #### 왜 이런 접근 방식이 좋은가
> - 이런 접근 방식은 네트워크가 깊어져도 각 레이어가 학습할 작업을 더 간단하게 만들어, 깊은 네트워크의 학습을 더욱 효과적으로 만듬.

![](https://velog.velcdn.com/images/heayounchoi/post/57a03493-d71b-498b-8d1f-cf3c4a4595fe/image.png)
- F(x)+x는 feedforward neural network with shortcut connections라고 볼 수 있음.
- shortcut connections는 몇 단계의 레이어를 건너뛰어서 stacked layers의 output으로 더해지는 것. 그래서 추가적인 파라미터가 생기는 것도 아니고, 기존에 SGD로 학습시키던 것처럼 학습이 가능함.

## Related Work
### Residual Representations
- 벡터 양자화에 있어 residual vector를 인코딩하는 것이 original vector보다 훨씬 효과적.
> #### residual vector
> - 원본 벡터에서 어떤 대표 벡터를 뺀 나머지 부분. 이 '나머지' 부분이 때로는 원본 데이터를 표현하는데 더 효율적일 때가 있음.
- 벡터 양자화란 큰 데이터 벡터(리스트 또는 배열 같은 것)를 더 작고 단순한 형태의 벡터로 바꾸는 과정을 의미.
- low-level 비전 및 컴퓨터 그래픽 문제에서 편미분 방정식을 풀기 위해 멀티 그리드 방식을 많이 사용해왔는데, 이 방식은 시스템을 여러 scale의 하위 문제로 재구성하는 것.
- 멀티 그리드 방식 대신에 두 scale 간의 residual 벡터를 가리키는 변수에 의존하는 방식이 있는데, 이를 계층 기반 pre-conditioning이라고 함. 이 방식은 기존 방식보다 훨씬 빨리 수렴하는 특징이 있음.
- 즉, 합리적인 문제 재구성과 전제 조건(pre-conditioning)은 최적화를 더 간단하게 수행해준다는 것을 의미.
> 멀티그리드는 top-down, 계층 기반 pre-conditioning은 bottom-up 접근 방식으로 볼 수 있음. 멀티그리드 방식에서는 큰 문제에서 시작해서 솔루션을 얻은 후, 그것을 더 세부적인 문제의 초기 추정값으로 사용함. 반면 계층 기반 pre-conditioning은 작은 문제들로 시작해서 그 해결책을 합쳐서 더 큰 문제의 해결책을 구성함.
### Shortcut Connections
- ResNet의 shortcut connection은 highway networks와는 달리, parameter가 전혀 추가되지 않으며, gated shortcut처럼 닫히지 않기 때문에 지속적으로 residual function을 학습함.

## Deep Residual Learning
### Residual Learning
- 실제로는 identity mapping이 최적의 해답일 확률은 낮음. 하지만 이러한 아이디어는 문제를 precondition 할 수 있음.
- If the optimal function is closer to an identity mapping than to a zero mapping, it should be easier for the solver to find the perturbations with reference to an identity mapping, than to learn the function as a new one.
- 실험 결과에 따르면 residual functions는 주로 small responses를 갖고 있고, 이는 identity mapping이 합리적인 preconditioning을 제공한다는 의미.
### Identity Mapping by Shortcuts
![](https://velog.velcdn.com/images/heayounchoi/post/bfa27b22-7ef8-453c-b5e1-9249999f29ee/image.png)
- F+x 연산을 위해 x와 F의 차원이 같아야 하는데, 이들이 서로 다를 경우 linear projection인 $W_s$를 곱하여 차원을 같게 만들 수 있음. 여기서 $W_s$는 차원을 매칭 시켜줄 때에만 사용.
### Network Architectures
#### Plain Network
- baseline 모델로 사용한 plain net은 VGGNet에서 영감을 받음. conv filter의 사이즈가 3 x 3이고, 다음 2가지 규칙에 기반하여 설계.
1. Output feature map의 size가 같은 layer들은 모두 같은 수의 conv filter를 사용.
2. Output feature map의 size가 반으로 줄어들면 time complexity를 동일하게 유지하기 위해 필터 수를 2배로 늘려준다.
> #### Time Complexity
> - 알고리즘이 어떤 문제를 해결하는데 필요한 단계의 수나 연산의 수를 입력 크기와 관련하여 표현한 것.
> - 피쳐맵의 크기가 절반으로 줄어들면, 연산량도 크게 줄어들게 됨. 특징을 잘 추출하고 정보를 잃지 않기 위해서는 더 많은 필터를 사용해서 더 다양한 특징을 추출해야 함. 따라서 필터의 수를 늘려줌으로써 연산량 유지.
- downsampling은 stride가 2인 conv filter를 사용. 
- 마지막으로, 모델 끝단에 global average pooling layer를 사용하고, 사이즈가 1,000인 FC layer와 Softmax를 사용. 
- 결과적으로 전체 layer 수는 34인데 이는 VGGNet보다 적은 필터와 복잡성을 가짐. (VGGNet은 4,096 사이즈 FC layer가 2개 추가됨)
> #### Downsampling
> - 이미지나 신호의 해상도를 줄이는 과정. 주로 피쳐맵의 공간적 해상도를 줄이기 위해 사용됨.
> - 풀링을 사용해서 다운샘플링을 할 수 있지만, 필터를 사용해서 다운샘플링을 하면 다운샘플링을 하는 동시에 피쳐맵의 특징도 학습할 수 있음. 
> #### GAP(Global Average Pooling)
> - 각 피쳐맵의 모든 값을 평균 내어 하나의 숫자로 변환. GAP는 고정된 길이의 벡터를 생성하기 때문에 네트워크의 끝 부분에서 FC layer에 연결하기 전 사용됨.
![](https://velog.velcdn.com/images/heayounchoi/post/a867e40c-be2b-43f0-b455-98870b57f14c/image.png)
#### Residual Network
- Residual network는 plain 모델에 기반하여 shortcut connection을 추가하여 구성. 
- 이때 input과 output의 차원이 같다면, identity shortcut을 바로 사용하면 되지만, dimension이 증가했을 경우 두 가지 선택권이 있음.
1. zero padding을 적용하여 차원을 키워주기.
2. 앞서 다뤘던 projection shortcut을 사용. (1 x 1 convolution)
- shortcut이 feature map을 2 size씩 건너뛰므로 stride를 2로 설정.
![](https://velog.velcdn.com/images/heayounchoi/post/60f725e6-ce08-41a3-82f5-ea4d32f794c8/image.png)
### Implementation
1. 짧은 쪽이 256, 480 사이가 되도록 random 하게 resize 수행
2. resize 된 이미지 또는 이를 horizontal flip 한 이미지에서 224x224 사이즈로 random하게 crop 후 centering
3. standard color augmentation 적용
4. batch normalization right after each convolution and before activation
5. He 초기화 방법으로 가중치 초기화 (ReLU와 사용할때 효과적인 초기화 방법)
6. Optimizer : SGD (mini-batch size : 256) 
7. Learning rate : 0.1에서 시작 (학습이 정체될 때 10씩 나눠준다)
8. trained for up to 60 X 10^4 iterations
9. Weight decay : 0.0001
10. Momentum : 0.9
11. 60 X 10^4 반복 수행
12. dropout 미사용
- 테스트 단계에서는 10-cross testing 방식을 적용
- VGGNet처럼 테스트 단계에서 신경망의 fc layer를 convolution layers로 변환해서 사용.
- multiple scale을 적용해 짧은 쪽이 {224, 256, 384, 480, 640} 중 하나가 되도록 resize 한 후, 평균 score을 산출.
> #### 10-cross validation
> - the process of performing validation using the 10-crop technique during the model's training phase
> - used repeatedly during training to tune and validate the model
> - Using 10-crop validation means that for each image in the validation set, 10 different crops (4 corners, center, and their horizontal flips) are taken and processed through the model. The model's predictions for these 10 crops are averaged to get a final prediction for that image.
> #### 10-cross testing
> - 10-crop technique applied at the testing stage
> - Like in validation, for each image in the test set, 10 crops are derived, processed through the model, and their predictions are averaged to produce a final prediction.

## Experiments
### ImageNet Classification
- trained on the 1.28 million training images
- evaluated on the 50k validation images
- final result on the 100k test images
![](https://velog.velcdn.com/images/heayounchoi/post/55452ef2-46f3-4e4f-b290-f6002a2f088e/image.png)
#### Plain Networks
- 먼저, plain 모델에 대해 실험을 수행하였는데, 18 layer의 얕은 plain 모델에 비해 34 layer의 더 깊은 plain 모델에서 높은 validation error가 나타났다고 함. training / validation error 모두를 비교한 결과, 34 layer plain 모델에서 training error도 높았기 때문에 degradation 문제가 있다고 판단.
- 이러한 최적화 문제는 vanishing gradient 때문에 발생하는 것은 아니라 판단하였는데, plain 모델은 batch normalization이 적용되어 순전파 과정에서 모든 신호의 variance는 0이 아니며, 역전파 과정에서의 기울기 또한 healthy norm을 보였기 때문.
- 따라서 순전파, 역전파 신호 모두 사라지지 않았으며, 실제로 34-layer의 정확도는 경쟁력 있게 높은 수준.
- deep plain model은 exponentially low convergence rate를 가지기 때문에 training error의 감소에 좋지 못한 영향을 끼쳤을 것이라 추측.
> #### exponentially low convergence rate
> - 모델의 깊이나 복잡성이 증가함에 따라 학습 과정에서의 수렴 속도가 기하급수적으로 느려지는 것
#### Residual Networks
- 다음으로, 18 layer 및 34 layer ResNet을 plain 모델과 비교. 
- in the first comparizon, 모든 shortcut은 identity mapping을 사용하고, 차원을 키우기 위해 zero padding을 사용하였기에 파라미터 수는 증가하지 않음. (option A)
- residual learning으로 인해 상황이 역전되어 34-layer ResNet이 18-layer ResNet보다 2.8%가량 우수한 성능을 보임. 특히, 34-layer ResNet에서 상당히 낮은 training error를 보였고, validation data에도 일반화 됨. 이는 degradation 문제가 잘 해결되었으며, depth가 증가하더라도 좋은 정확도를 얻을 수 있음을 의미.
- 34 layer plain net과 ResNet을 비교해도 ResNet이 top-1 error를 3.5% 감소시킴, which verifies the effectiveness of residual learning on extremely deep systems.
- 18-layer ResNet과 plain net을 비교했을 때 성능이 거의 유사했지만, 18-layer의 ResNet이 더 빨리 수렴. 즉, 모델이 과도하게 깊지 않은 경우 (18-layer), 현재의 SGD Solver는 여전히 plain net에서도 좋은 solution을 찾을 수 있지만, ResNet eases the optimization by providing faster convergence at the early stage.
#### Identity vs Projection Shortcuts
- 위에서 parameter-free, identity shortcut이 학습을 돕는다는 것 확인.
- 다음은 projection shortcut을 3가지 옵션에 대해 비교
A) zero-padding shortcuts used for increasing dimensions, and all shortcuts parameter-free
B) projection shortcut을 사용한 경우. (dimension을 키워줄 때에만 사용) 다른 모든 shortcut은 identity.
C) 모든 shortcut으로 projection shortcut을 사용한 경우.
- 이때, 3가지 옵션 모두 plain model보다 좋은 성능을 보였고, 그 순위는 A < B < C 순. 
- B is better than A because the zero-padded dimensions in A indeed have no residual learning.
- C is better than B because extra parameters are introduced by many (thirteen) projection shortcuts.
- 3가지 옵션의 성능차가 미미했기에 projection shortuct이 degradation 문제를 해결하는데 필수적이지는 않다는 것을 확인.
- memory/time complexity와 model size를 줄이기 위해 이 논문에서는 C 옵션을 사용하지 않음. Identity shortcut은 bottleneck 구조의 복잡성을 높이지 않는 데에 매우 중요하기 때문.
#### Deeper Bottleneck Architectures
- ImageNet에 대하여 학습을 진행할 때 training time이 매우 길어질 것 같아 building block(모델의 기본 구성 요소)을 bottleneck design으로 수정하였다고 함.
- 따라서 각각의 residual function인 F는 3-layer stack 구조로 바뀌었는데, 이는 1x1, 3x3, 1x1 conv로 구성.
![](https://velog.velcdn.com/images/heayounchoi/post/048c83e3-6a32-4dc1-b790-cad03aad2aca/image.png "기존 ResNet building block과 bottleneck design이 적용된 building block")
###### 기존 ResNet building block과 bottleneck design이 적용된 building block
- 여기서 parameter-free 한 identity shortcut은 bottleneck 구조에서 특히 중요함. 만약 identity shortcut이 projection shortcut으로 대체되면, shortcut이 2개의 high-dimensional 출력과 연결되어 time complexity와 model size가 2배로 늘어남. 따라서 identity shortcut은 bottleneck design을 더 효율적인 모델로 만들어줌.
##### 50-layer ResNet
- 34-layer ResNet의 2-layer block을 3-layer bottleneck block으로 대체하여 50-layer ResNet을 구성.
- dimension matching을 위해 B 옵션을 사용.
##### 101-layer and 152-layer ResNets
- 더 많은 3-layer block을 사용하여 101-layer 및 152-layer ResNet을 구성.
- depth가 상당히 증가하였음에도 VGG-16/19 모델보다 더 낮은 복잡성을 가짐.
- 50/101/152 layer ResNets are more accurate than the 34-layer ones by considerable margins, without the degradation problem.
#### Comparisons with State-of-the-art Methods
- ILSVRC 2015에서 1등한 모델은 an ensemble with six models of different depth combined(only two 152-layer ones).
------
- 참고한 블로그에서 잘못된 해석들이 좀 있어서 수정함.
#### 추가적으로 공부할 것
- degradation problem에 대한 최신 논문


Reference: 
https://phil-baek.tistory.com/entry/ResNet-Deep-Residual-Learning-for-Image-Recognition-%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0
[Deep Residual Learning for Image Recognition 논문](https://arxiv.org/pdf/1512.03385.pdf)
