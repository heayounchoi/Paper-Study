# ImageNet Classification with Deep Convolutional Neural Networks / AlexNet

- 2012년 이미지 분류 대회에서 우승을 차지한 모델
- CNN은 Feedforward Neural Network보다 더 적은 파라미터와 복잡성이 있어 학습하기 쉬움에도 불구하고, 이론상 최대 성능은 Feedforward Neural Network보다 약간만 떨어졌다.
- Dataset의 경우, 고정된 크기의 이미지가 필요해 256x256으로 이미지 크기를 줄이고, 정사각형 이미지를 얻기 위해 이미지 중앙을 crop 했다. 그리고 각 픽셀의 RGB  값을 전체 이미지의 RGB 평균으로 해서 이미지 데이터를 centering 했다.
	- Centering은 각 특성의 평균값을 0 주변으로 끌어들임으로써, 최적화 알고리즘이 손실 함수의 최소값을 찾는데 더 빠르게 도달할 수 있다.

### Architecture
1) ReLU Nonlinearity
- Saturating nonlinearities를 사용하는 것보다 non-saturating nonlinearities를 사용하는 것이 학습 에러율을 낮추는데 걸리는 시간이 6배 단축된다고 한다. Saturating nonlinearities의 경우 미분값이 작아지면 역전파시 vanishing gradient 문제가 발생할 수 있기 때문이다.

2) Multiple GPUs
- 특정 레이어에서 2개의 GPU를 병렬적으로 사용한다.

3) Local Response Normalization
- ReLU는 양수의 방향으로는 입력의 값을 그대로 사용하기 때문에, conv나 pooling 시 매우 높은 하나의 픽셀 값이 주변의 픽셀에 영향을 미치게 된다. 이런 부분을 방지하기 위해 activation map의 같은 위치에 있는 픽셀끼리 정규화를 했다.
- 깊은 레이어에서는 LRN의 효과가 덜 중요하기 때문에 정규화는 처음 몇 레이어에만 적용됐으며, ReLU 이후 강한 활성화된 뉴런들을 균형있게 조정하기 위해 ReLU를 거치고 난 결과값에 사용했다.
![](https://velog.velcdn.com/images/heayounchoi/post/5bb46d3c-20af-47a0-9771-2152dbf32232/image.png)
- 논문에서는 k=2, n=5, α=10^-4, β=0.75로 설정했다.
- 공식은 만약 주어진 뉴런이 주변 뉴런들에 비해 매우 강하게 활성화되면, 해당 뉴런의 활성화 값을 줄어들게 하고, 주변 뉴런들에 비해 활성화가 약하면, 해당 뉴런의 활성화 값을 그대로 둔다.

4) Overlapping Pooling
- 겹쳐서 pooling 연산 수행. 겹치는 영역으로 인해 같은 특징이 여러 pooling 유닛에 의해 캡처될 수 있고, 단일 유닛의 오류가 출력에 큰 영향을 미치기 어렵게 만들어, 모델의 robustness를 향상시키는 효과가 있다.
- stride=2, kernel=3x3 사용

5) Overall Architecture
- 총 8개의 레이어로, 5개는 Convolution Network, 3개는 Fully Connected Network
- 파라미터 개수는 약 6000만개
![](https://velog.velcdn.com/images/heayounchoi/post/b2497fa9-9d08-4fb5-b390-77f6f49d2235/image.png)

### Reduce Overfitting
1) Data Augmentation
- 좌우반전, 패치 사용
- RGB 채널의 색상 강도 조절

2) Dropout
- 사용자가 지정한 확률을 근거로 하여 특정 뉴런에 신호를 전달하지 않는 방법
- 1번째와 2번째 fc layer에 50% 확률로 dropout 적용

---
추가적으로 공부할 부분
- ~~GPU 병렬 처리 하는 방법~~ (https://wooono.tistory.com/331)
- ~~LRN 대신 Batch Normalization이 쓰이게 된 이유~~

Reference: https://killerwhale0917.tistory.com/14
