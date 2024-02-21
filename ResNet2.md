# ResNet / Deep Residual Learning for Image Recognition
[Introduction]
- instead of learning unreferenced functions, learning residual functions with reference to the layer inputs can make networks easier to optimize
- 네트워크가 깊어지면 학습이 어려워짐
    - vanishing/exploding gradients 문제:
        - normalized initialization, intermediate normalization layers 등으로 해결 가능
    - degradation 문제:
        - with the network depth increasing, accuracy gets saturated and then degrades rapidly
        - not caused by overfitting(adding more layers leads to higher training error)
        - added layers가 identity mapping이라면, shallower counterpart보다 higher training error를 발생시키지 않을것
- this paper addresses the degradation problem by introducing a deep residual learning framework
- desired underlying mapping: H(x)
- residual mapping: F(x) = H(x) - x => H(x) = F(x) + x
- the idea is that it is easier to optimize the residual mapping than to optimize the original, unreferenced mapping
- if an identity mapping were optimal, it would be easier to push the residual to zero than to fit an identity mapping by a stack of nonlinear layers
- F(x)+x can be realized by feedforward neural networks with “shortcut connections”
- shortcut connections will perform identity mapping, and their outputs will be added to the outputs of the stacked layers

[Related Work]
- VLAD(Vector of Locally Aggregated Descriptors): representation that encodes by the residual vectors with respect to a dictionary
    - for vector quantization, encoding residual vectors is shown to be more effective than encoding original vectors
    - vector quantization: 고차원 벡터 공간에서의 데이터 포인트를 작은 수의 코드북 또는 사전에 정의된 벡터로 대체하는 과정. 데이터 압축을 가능하게 하며, 효율적인 저장과 전송, 빠른 검색을 위한 데이터의 근사화를 목적으로 함
- Fisher Vector: probabilistic version of VLAD
- in low-level vision and computer graphics, for solving Partial Differential Equations(PDEs), Multigrid method reformulates the system as subproblems at multiple scales, where each subproblem is responsible for the residual solution between a coarser and a finer scale
- An alternative to Multigrid is hierarchical basis preconditioning, which relies on variables that represent residual vectors between two scales
- these solvers converge much faster than standard solvers that are unaware of the residual nature of the solutions
- shortcut connection의 경우 원래 MLP 학습은 linear layer로 구성돼 있었고, vanishing/exploding gradients를 해결하기 위해서 few intermediate layers를 보조 classifiers에 직접 연결하는 연구도 있었음.
- layer responses, gradients, propagated errors를 중심화하기 위해 shorcut connections가 사용되기도 했으며, inception layer에서도 shorcut branch가 있었음
- highway networks에서도 shortcut connections를 사용하는데, gating functions를 활용함. 이 gates는 data-dependent하고 parameters를 가지며, when a gated shorcut is “closed”, the layers in highway networks represent non-residual functions
- in ResNet, residual functions are always learned since shortcuts are never closed
- highway networks have not demonstrated accuracy gains with extremely increased depth(over 100 layers)

[Deep Residual Learning]
- If multiple nonlinear layers can asymtotically approximate complicated functions, then they can also asymptotically approximate the residual functions
- degradation problem suggests that the solvers might have difficulties in approximating identity mappings by multiple nonlinear layers
- residual learning is adopted to every few stacked layers

￼<img width="187" alt="Pasted Graphic 44" src="https://github.com/heayounchoi/Paper-Study/assets/118031423/1967aa36-05b2-4f37-81b8-7269a509be0f">

- dimensions of x and F must be equal, but if this is not the case, linear projection can be performed by the shorcut connections to match the dimensions

<img width="215" alt="Pasted Graphic 45" src="https://github.com/heayounchoi/Paper-Study/assets/118031423/73457467-0bc1-4fca-9ac4-a403eaf2f561">

- Plain Network:
    - VGG net을 기반으로 함
    - 3x3 filters for conv layers
    - for the same output feature map size, the layers have the same number of filters
    - if the feature map size is halved, the number of filters is doubled so as to preserve the time complexity per layer
        - 특성 맵의 크기를 줄임으로써 발생할 수 있는 정보 손실을 보상하기 위해 필터의 수를 늘려서 더 많은 특성을 추출할 수 있도록 함
    - **downsampling directly by conv layers with a stride of 2**
        - 추가적인 풀링 계층을 사용하는 것에 비해 계산 비용이 낮음
    - network ends with a global average pooling layer and 1000-way fully-connected layer with softmax
    - total number of weighted layers is 34
    - has fewer filters and lower complexity than VGG nets

<img width="315" alt="Pasted Graphic 46" src="https://github.com/heayounchoi/Paper-Study/assets/118031423/09f96573-0105-4452-995d-875ef424043d">

- Residual Network:
    - insert shortcut connections on the plain network
    - identity shortcuts can be directly used when the input and output are of the same dimensions(solid line)
    - when dimensions increase(dotted line):
        - (A) zero padding for incresasing dimensions(no extra parameter)
        - (B) projection shortcut
        - botch options performed with a stride of 2
- implementation:
    - image is resized with its shorter side randomly sampled in [256, 480] for scale augmentation
    - 224x224 crop randomly sampled from an image or its horizontal flip, with the per-pixel mean subtracted
    - standard color augmentation
    - batch normalization right after each conv and before activation
    - weight initialization
    - SGD optimizer
    - batch size 256
    - learning rate 0.1, factor 0.1 when error plateaus
    - **60X10^4 iterations**
    - weight decay 0.0001
    - momentum 0.9
    - no dropout
    - 10-crop testing
    - average scores at multiple scales {224,256,384,480,640}

[Experiments]
- 레이어가 깊어지면 training error가 높아지는 현상이 있음
- 이게 레이어가 깊어질수록 필요해지는 학습 횟수가 기하급수적으로 늘어나는 것으로 예상했지만, 학습 횟수를 늘려도 문제가 해결되지 않았다고 함
- 이런 현상은 shortcut connection을 추가해 resnet으로 바꾸면서 해결됨. 레이어가 깊어질수록 에러율이 낮아짐
- 레이어가 overly deep 하지 않은 경우에는 plain net에서도 학습이 잘 이루어지지만, resnet을 사용할 경우 convergence가 더욱 빨라짐(identity mapping)
- projection shortcut의 경우 (A) zero-padding shortcuts for increasing dimensions, (B) projection shortcuts for increasing dimensions, (C) all shortcuts are projections 세 가지 경우를 실험했음. C > B > A 가 결과였지만 세 경우 모두 크게 성능 차이가 나지 않았기 때문에 C 보다 parameter가 적은 B를 사용하기로 함

￼<img width="503" alt="Pasted Graphic 47" src="https://github.com/heayounchoi/Paper-Study/assets/118031423/50ca6eb8-243e-4be9-b6e6-06aa067932ed">

- 효율성을 위해 왼쪽과 같은 building block을 오른쪽의 bottleneck design으로 사용함
- 50, 101, 152 layer resnets는 위 building block 디자인으로 설계됨
- 레이어를 엄청나게 늘렸음에도 불구하고 VGG 16/19보다 lower complexity를 가짐
- CIFAR-10:
    - 32x32 input with per-pixel mean subtracted
    - stack of 6n layers with 3x3 convs on the feature maps of sizes {32, 16, 8}, with 2n layers for each feature map size. numbers of filters are {16, 32, 64}
    - subsampling with a stride of 2
    - network ends with global average pooling, a 10-way fc layer, and softmax

￼<img width="430" alt="Pasted Graphic 48" src="https://github.com/heayounchoi/Paper-Study/assets/118031423/50684cc2-0d71-4dd7-8db7-20b342b56384">

	- shortcut connections are connected to the pairs of 3x3 layers (totally 3n shortcuts)
	- option A identity shortcuts used in all cases
	- weight decay: 0.0001
	- momentum: 0.9
	- weight initialization
	- BN
	- no dropout
	- mini batch size: 128
	- base learning rate: 0.1, divided by 10 at 32k and 48k iterations and training terminated at 64k iterations
	- data augmentation: 4 pixels padded on each side, 32x32 crop randomly sampled from the padded image or its horizontal flip
	- testing: only the single view of the original 32x32 image is used
	- compare n = {3, 5, 7, 9}, leading to 20, 32, 44, 56 layer networks
	- n = 18, 110 layer ResNet에서는 base learning rate of 0.1이 너무 커서 0.01을 쓴 다음 0.1로 다시 돌아갔다고 함. 근데 처음부터 0.1로 했을때랑 accuracy 차이가 별로 없었다고 함
	- layer response를 확인함으로써 residual function이 non-residual function보다 zero에 더 가까울 것이라는 motivation도 확인했다고 함. 레이어가 깊어질수록 layer response의 magnitude도 작아졌음
	- n = 200, 1202 layer도 테스트 해봤는데 training error는 110 layer일때와 비슷함에도 불구하고 test 성능은 110 layer보다 안좋게 나옴. 저자는 overfitting 때문일 것이라고 생각하며, regularizaiton을 더 강하게 하면 성능 개선이 있을것이라고 함.
