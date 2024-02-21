# MobileNetV2: Inverted Residuals and Linear Bottlenecks
[Introduction]
- modern SOTA image recognition networks require high computational resources beyond the capabilities of many mobile and embedded applications
- this paper introduces architecture that is specifically tailored for mobile and resource constrained environments
- main contribution of this paper is a novel layer module: the inverted residual with linear bottleneck
- this module takes as an input a low-dimensional compressed representation which is first expanded to high dimension and filtered with a lightweight depthwise convolution
- features are subsequently projected back to a low-dimensional representation with a linear convolution
- this module allows to significantly reduce the memory footprint needed during inference by never fully materializing large intermediate tensors

[Related Work]
- AlexNet, VGGNet, GoogLeNet, ResNet
- hyperparameter optimization, network pruning, connectivity learning(뉴런이나 레이어 간의 연결 구조 최적화)
- ShuffleNet
- sparsity
- 최근에는 genetic algorithms과 reinforcement learning으로 architectural search를 진행하고 있다고 함
- 하지만 이렇게 하면 resulting networks end up very complex

[Preliminaries, discussion and intuition]
{Depthwise Separable Convolutions}
- basic idea is to replace a full conv operator with a factorized version that splits conv into two separate layers
- first layer is called a depthwise convolution, it performs lightweight filtering by applying a single convolutional filter per input channel
- second layer is a 1x1 convolution, called a pointwise convolution, which is responsible for building new features through computing linear combinations of the input channels

<img width="473" alt="Pasted Graphic 49" src="https://github.com/heayounchoi/Paper-Study/assets/118031423/ed3e350f-01b7-49b0-a2c8-c9e0d3d1204f">

- standard convolution은 입력 이미지 또는 특성 맵에서 모든 채널을 한꺼번에 고려해 합성곱을 수행함. 그래서 계산 비용이 매우 높음
- depthwise separable convolution은 먼저 입력의 각 채널에 대해 독립적으로 커널을 적용해 각 채널 내에서의 공간적 정보를 추출하고, 그 결과를 결합해 채널 간 정보를 각 위치에서 모든 채널을 통과하는 1x1 커널로 채널 간의 관계를 학습함. 
- k=3일 경우, 계산 비용이 8~9배 작아짐
{Linear Bottlenecks}
- linear bottleneck 구조는 고차원의 활성화 공간에서 저차원으로 정보를 압축하여 전달하는 역할을 함.
- 이 과정에서 중요한 정보는 보존되면서 데이터의 차원을 줄임으로써 계산 비용을 감소시킴
- bottleneck 구조 내에서 비선형 변환을 제한적으로 사용함. 특정 정보를 파괴할 수 있기 때문.
- linear bottleneck layer 다음에 ReLU와 같은 비선형 함수는 정보의 손실을 최소화하면서 필요한 비선형성을 제공하여 모델이 복잡한 패턴과 관계를 학습할 수 있음
{Inverted residuals}
- inspired by the intuition that the bottlenecks actually contain all the necessary information, shortcuts are directly used between the bottlenecks
- 작동 방식: 먼저 입력 x의 채널 수를 확장함. 이 확장된 데이터에 대해 어떤 변환 F(x)를 적용함. 그 후, 이 변환된 데이터의 채널 수를 다시 줄여서 원래의 입력 크기에 맞춤. 최종적으로, 이 줄어든 출력과 원래 입력 x를 더해 최종 출력을 생성함
{Information flow interpretation}
- MobileNetV2의 특징은 network의 capacity(bottleneck layers)와 expressiveness(layer transformation)를 분리해준다는 점. 

[Model Architecture]

<img width="586" alt="Pasted Graphic 51" src="https://github.com/heayounchoi/Paper-Study/assets/118031423/0efaff15-4990-4cd3-9ff4-ce8c4c56706c">

- initial fully convolution layer with 32 filters, followed by 19 residual bottleneck layers described in Table 2

<img width="561" alt="Pasted Graphic 50" src="https://github.com/heayounchoi/Paper-Study/assets/118031423/358c3580-fe7b-475c-a082-a2c7c4f626ef">

- dropout
- batch normalization
- network 크기랑 expansion rate는 비례함. 5에서 10 사이의 expansion rate는 비슷하다고 함
- expansion factor는 6을 쓰는데, 예를 들어 64-channel input tensor produces a tensor with 128 channels, the intermediate expansion layer is 384 channels(64*6)

[Experiments]
{ImageNet Classification}
- Tensorflow
- RMSPropOptimizer decay, momentum: 0.9
- batch norm after every layer
- weight decay: 0.00004
- initial learning rate 0.045
- learning rate decay rate: 0.98 per epoch
- batch size: 96

[Conclusions and Future Work]
- 여기서 나온 convolutional block이 network expressiveness를 capacity로부터 분리시키고 있음. 이 부분에 대해 리서치를 해보라고 함

논문에 나온 다른 모델들 찾아보기
