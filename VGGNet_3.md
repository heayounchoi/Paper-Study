# VGGNet: paper study &amp; implementation
https://arxiv.org/pdf/1409.1556.pdf

- 논문의 핵심 주제는 deep convolution network가 성능이 좋다는 것
- 3x3 filter, 16~19 layers 사용
- Localisation, classification task에서 좋은 성능
- Generalization 잘됨
- 이전에 image classification 분야에서 사용되던 방법은 high-dimensional shallow feature encodings였음
    - 고차원 이미지에서 얕은 계층 구조로 특징을 추출하던 전통적인 방법(SIFT, HOG)
- 과거엔 small filter size, small stride, training densely over the whole image and over multiple scales 등 다양한 방법들로 성능을 올리는 시도를 했었음
- Convnet configurations
    - Input: 224x224 RGB image
    - preprocessing: subtract mean RGB value from each pixel(training set)
    - Filter size: 3x3, 1x1(linear transformation followed by non-linearity)
    - Stride: 1 pixel
    - Padding: 1 pixel
    - Five max-pooling layers(2x2 pixel window, stride 2)
    - 3 fc layers(4096, 4096, 1000)
    - Final layer is softmax layer
    - ReLU after all hidden layers
    - Did not use local response normalisation(increases memory comsumption, computation time)
    - Width of conv layers(number of channels) starts from 64 and increases by a factor of 2 after pooling layers, until it reaches 512
    - 6 models, differ only in depth
  <img width="721" alt="Pasted Graphic" src="https://github.com/heayounchoi/VGGNet/assets/118031423/953b5d55-cca4-4c52-b0c2-72505002a6fd">
  <img width="459" alt="Pasted Graphic 1" src="https://github.com/heayounchoi/VGGNet/assets/118031423/36ca9440-8383-4131-a495-1244246677c3">

- 기존 연구에서는 first conv layer에 큰 filter를 사용했는데, 여기서는 3x3을 사용하는 이유
    - 3x3을 여러번 사용하면 큰 필터를 사용한 것과 같은 결과가 나오는데, 모든 layer이 ReLU를 거치면서 더욱 discriminative 해짐
    - Parameter 개수가 줄어듬
- 1x1 conv layer를 사용하는 이유(configuration C)
    - To increase the non-linearity of the decision function without affecting the receptive fields of the conv layers
- training
    - batch size: 256
    - momentum: 0.9
    - weight decay(L2 penalty multiplier 5*10^-4)
    - dropout ratio: 0.5 (first two fc layers)
    - learning rate: 10^-2 -> decrease by factor of 10 when the validation set accurcy stops improving(in paper, learning rate decreased 3 times and stopped after 370,000 iterations(74 epochs))
    - 기존 연구보다 빨리 converge 한 이유(아마도)
        - implicit regularisation imposed by greater depth and smaller conv filter sizes. implicit regularisation은 모델 설계의 특정 측면이 학습 과정에서 자동적으로 과적합을 방지하는 효과를 내는 것을 의미함
        - pre-initialisation of certain layers
            - net A로 먼저 학습시키기(shallow enough to be trained with random initialisation)
            - train deeper architectures with the layers of net A(first four conv layers and last three fc layers only. randomly initialise intermediate layers)
            - did not decrease learning rate for pre-initialised layers, allowing them to change during learning(일반적으로는 사전 학습된 레이어들의 매개변수가 이미 유용한 정보를 담고 있기 때문에 학습 과정에서 크게 변화시키지 않는다고 함)
            - for random initialisation, weights were sampled from a normal distribution with the zero mean and 10^-2 variance, and biases were initialised with zero.
            - 하지만 연구가 끝나고 나서 사전 학습없이 가중치 초기화하는 방법을 찾았다고 한다..(Xavier 초기화)
    - input image
        - randomly cropped from rescaled training images (one crop per image per SGD iteration)
            - single-scale training: first trained network using training scale 256, then used the weights to initialise training network with training scale 384(also used smaller initial learning rate of 10^-3)
            - multi-scale training: rescale by randomly sampling training scale from a certain range [256, 512]. all layers fine-tuned with single-scale pre-trained model with training scale 384
        - random horizontal flipping and random RGB colour shift on crops
- testing
    - rescale image size(test scale), where test scale does not have to equal to the training scale. several values of test scale for each training scale leads to improved performance
    - fc layers are converted to conv layers(first fc layer 7x7 conv, last two fc layers 1x1 conv)
        - convolution layer로 바꾸는 이유: 입력 이미지의 크기에 대해 유연함. fully connected layer는 고정된 크기의 이미지만 처리할 수 있음
        - therefore, cropping is not necessary
    - input size가 다 다르니까 output size도 다 다름. 그래서 class score map spatially average 함
    - augment test set by horizontal flipping
    - average soft-max class posteriors of the original and flipped images to obtain final score
    - crop evaluation이 정확도는 더 높을 수 있지만, computational cost 있음
- implementation details
    - multi-GPU training 했는데, four NVIDIA Titan Black GPUs로 학습시켰을때 single net training이 2~3주 걸렸다고 함..
- evaluation
    - top-1 error: multi-class classification error
    - top-5 error: proportion of images such that the correct category is outside the top-5 predicted categories
    - for majority of experiments, validation set was used as test set
    - single-scale evaluation:
        - A랑 A-LRN 비교했을때 성능 차이가 없어서 LRN 사용하지 않았음
        - additional non-linearity가 도움이 되긴 하지만 spatial context를 capture 하는게 더 성능에 좋음
        - 19 layers에서 error rate은 saturate하지만 데이터셋이 더 크다면 더 깊은 레이어도 좋을 수 있음
        - shallow net with 5x5 conv filters와 deep net with 3x3 conv filters를 비교했을때 deep net이 더 성능이 좋았음
        - test 시 single scale image를 사용하더라도 training 시에는 scale jittering을 하는게 성능이 더 좋았음
  <img width="740" alt="Pasted Graphic 2" src="https://github.com/heayounchoi/VGGNet/assets/118031423/911720ea-4bb7-4d43-bb23-5bc895814213">

	- multi-scale evaluation:
		- testing scale을 다양하게 해서 평균내기 실험
		- fixed training scale일 경우 training scale에서 크게 차이가 나면 성능이 안좋아지기 때문에 +/-32로 했고, training 때 scale jittering을 사용했을 경우 {Smin, 0.5(Smin + Smax), Smax}로 했음
		- deep net, scale jittering on training set, multi-scale on test set 콤보일때 성능이 젤 좋았음
  <img width="762" alt="Pasted Graphic 3" src="https://github.com/heayounchoi/VGGNet/assets/118031423/37d43f56-69fc-4328-953d-627584ba77d8">

	- multi-crop evaluation
		- multi-crop & dense가 가장 성능이 좋은 이유는 convolution boundary condition이 다양하기 때문으로 추정한다고 함
  <img width="804" alt="Pasted Graphic 4" src="https://github.com/heayounchoi/VGGNet/assets/118031423/c0ce644f-66ca-4ab3-9883-8c3bfc671fda">

	- ConvNet fusion
		- submission 당시에는 single-scale network랑 multi-scale D 까지밖에 학습을 못시켜서 7개 모델을 앙상블 해서 제출했음
  <img width="826" alt="Pasted Graphic 5" src="https://github.com/heayounchoi/VGGNet/assets/118031423/387138fd-5d34-4d9b-b15c-4837e04a13b1">

	- comparison with the state-of-the-art
		- 1등이 GoogLeNet ensemble, 2등이 VGG였음
		- submission 후 모델 2개만 써서 좋은 성능을 달성했다는건 remarkable 함 (구글넷이 7개 썼으니께)
		- network 1개만 썼을때는 VGGNet 성능이 제일 좋음
		- classical ConvNet 구조에서 깊이만 늘렸다는게 포인트
  <img width="826" alt="Pasted Graphic 8" src="https://github.com/heayounchoi/VGGNet/assets/118031423/de24ca78-b642-4742-8f94-39740dbe6daa">

#### VGGNet Appendix
A. Localisation
- 본문에서는 classification task 얘기했고, 여기서는 localisation task 우승한 건에 대해 이야기할것
- task: predict a single object bounding box for each of the top-5 classes, irrespective of the actual number of objects of the class
A.1. Localisation ConvNet
- fc layer에서 class score 대신 bounding box location을 예측해야함
- bounding box: 4-D vector storing its center coordinates, width, and height
- convnet D architecture 사용
- SCR(single-class regression): 하나의 공통된 회귀 모델로 모든 객체 클래스에 대한 바운딩 박스 예측
- PCR(per-class regression): 각 객체 클래스마다 별도의 바운딩 박스 예측 회귀 모델 사용
- training
    - replace the logistic regression objective with a Euclidean loss
    - trained two models, each on a single scale: 256, 384
    - initialised with the corresponding classification models
    - learning rate: 10^-3
    - last fc layer initialised randomly and trained from scratch
- testing
    - 2가지 테스트 방법
        - network modification 비교를 위해 validation set에서 클래스 별 bounding box prediction
        - 전체 이미지에 대해 객체 위치를 찾는 방법
A.2. Localisation Experiments
- Settings comparison
    - PCR 성능이 SCR 보다 좋음(기존 연구와 다름)
    - fine-tuning all layers > fine-tuning only the fc layers
- Fully-fledged evaluation
    - center crop을 사용하는것보다 top-5 localisation error가 낮음
    - classification task처럼 multi-scale testing이랑 fusion도 성능 향상을 시켜줌
- Comparison with the state of the art
    - 이전 대회 우승자 Overfeat보다 scale을 적게 사용하고, resolution enhancement technique을 사용하지 않았음에도 불구하고 결과가 좋았음
    - simple한 method로 powerful한 representation을 가질 수 있다는 것이 의미 있음
B. Generalisation of Very Deep Features
- ILSVRC dataset으로 pre-trained 된 VGG가 다른 작은 데이터셋에 대해서 잘 generalise 되는지(feature extractor)
- Net D랑 Net E 사용
- pre-trained net의 weights는 고정된 상태로 유지됨. 마지막 classifier layer만 제거하고, 마지막에서 두번째 fc layer에서 얻은 값을 L2 정규화한 후 대상 데이터셋에서 훈련된 linear SVM classifier와 결합함
    - L2 정규화하는 이유: 벡터의 크기를 표준화하므로 데이터의 스케일이 다를때도 일관된 특징 벡터를 얻을 수 있음
- Image Classification on VOC
    - aggregating image descriptors computed at multiple scales와 aggregation by stacking가 비슷한 결과를 냄(stacking은 computational cost가 큼)
    - VOC dataset의 objects가 다양한 스케일로 이루어져있어서 scale-specific semantics를 classifier가 exploit 하지 못했기 때문으로 추정
    - range of scales는 많던 적든 크게 차이 없음
- Image Classification on Caltech
    - VOC와 달리 stacking이 averaging이나 max-pooling보다 좋았음
    - Caltech 이미지는 object가 이미지 전체를 차지하고 있어서, multi-scale 했을 경우 whole object vs object parts가 되기 때문
- Action Classification on VOC
    - two training settings:
        - on whole image and ignoring bounding box
        - on whole image and bounding box(성능 더 좋음)
- Other Recognition Tasks
    - semantic segmentation, image caption generation, texture and material recognition 등에서도 shallow architecture보다 좋은 performance를 보여줌



