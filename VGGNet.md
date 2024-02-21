# Very Deep Convolutional Networks for Large-Scale Image Recognition / VGGNet

- 연구 초점은 CNN의 깊이와 매우 큰 이미지 데이터셋에 대한 성능.
- 메인 아이디어는 아주 작은 3X3 Conv layer를 이용하는 것. 이것이 16, 19층의 layer를 쌓으면서 성능 향상.
- 다른 데이터셋에 대해서도 일반화 시켜서 현재까지도 backbone으로 많이 쓰임.
- 매우 많은 메모리를 이용하여 연산한다는 단점이 있음.
- 1등한 GoogLeNet보다 fc layer 사용으로 인한 파라미터 개수가 너무 많음.
![](https://velog.velcdn.com/images/heayounchoi/post/8aaa8984-4bef-41c2-9a6c-e6604ec1e2d0/image.png)

## ConvNet Figurations
![](https://velog.velcdn.com/images/heayounchoi/post/cb546eb3-1219-4539-86af-7cf99c8e4040/image.png)
![](https://velog.velcdn.com/images/heayounchoi/post/b872ca57-2dbe-40bc-96e9-ac0446cc8d84/image.png)

#### 전처리
- 224X224 RGB 이미지를 입력으로 하고, 전처리는 Centering만 함.
> #### 224X224 크기의 이미지를 사용하는 이유
1) 다양한 모데 구조와 하드웨어에서 괜찮은 성능과 계산 효율성. 너무 작은 이미지는 정보를 잃을 위험이 있고, 너무 큰 이미지는 계산 비용이 높음.
2) GPU의 메모리 크기와 병렬 처리 능력 고려. 메모리 사용량을 최적화하는 동시에 충분한 정보 포함.
3) 연구자들이 표준화된 데이터 세트와 모델 구조를 사용하면, 연구 결과를 비교/분석하기 쉬움.
> #### 정사각형 이미지를 사용하는 이유
> - 연산 효율성, 메모리 사용 효율이 좋고, 컨볼루션 필터도 정사각형을 주로 사용함.
> #### 정사각형 필터를 사용하는 이유
> - 대칭성을 갖기 때문에 학습해야 하는 파라미터 수를 줄여줌.
> #### 홀수 크기의 필터를 사용하는 이유
> - 홀수 크기의 필터는 중심 픽셀을 가짐. 이 중심 픽셀을 중심으로 주변 픽셀들과의 관계를 해석하게 되므로, 특성 추출에서 직관적이고 대칭적인 연산을 할 수 있음. 이러한 대칭성은 필터의 반응이 좀 더 균일하게 되도록 도와줌.

#### Conv layer
```python
Conv2D(kernel_size=(3,3), padding='same', activation='relu', input_shape=(224, 224, 3))
```
- 대부분의 필터는 가장 작은 3X3 필터를 사용하고, 1X1 필터도 사용함.
- 1X1 필터는 채널을 줄여서 bottleneck으로 사이즈를 맞추기 위해 많이 쓰인다고 함.
- stride는 1로 고정.
- padding은 컨볼루션 후에 spatial resolution 보존(output feature map의 사이즈가 down 되는 걸 막기)을 위해 same으로 둠.
> #### 3X3 사용 이유
> - 작은 크기의 필터는 필요한 파라미터 수가 적으면서도 충분한 수용 영역을 가짐. 이로 인해 모델의 연산 복잡성을 줄이면서도 좋은 성능을 달성할 수 있음. 예를 들어 두 개의 연속적인 3X3 컨볼루션은 5X5의 수용 영역을 가짐. 더 적은 파라미터로 깊은 네트워크를 구성할 수 있음. 여러 레이어를 거쳐 만들어진 특징 맵은 더 추상적인 정보를 담게 됨.
> #### bottleneck
> - 정보나 피쳐맵의 차원을 임시로 축소하고, 나중에 다시 확장하는 구조. 계산 효율성 향상.

#### MaxPooling
```python
MaxPool2D(2X2, stride=2)
```
- 몇차례의 Conv layer 이후, 맥스풀링은 2X2 stride 2로 5번 들어가 다운사이즈.

#### Fc layer
```python
Dense(4096, activation='relu')
Dense(4096, activation='relu')
Dense(1000, activation='softmax')
```
- LRN은 결과가 좋지 않아 포함하지 않음.
> #### LRN 보다 Batch Normalization이 많이 쓰이게 된 이유
> - 다양한 네트워크 및 데이터셋에 대해 일관된 성능 향상.
> - 내부 공변량 변화 감소로 인한 안정적인 학습.
> - 가중치 초기화에 대한 의존성 감소.
> - 다양한 활성화 함수와 함께 사용 가능해 모델의 유연성 증가.
> #### 왜 LRN은 효과가 없었을까
> - 깊은 네트워크에서는 LRN의 효과가 상대적으로 약할 수 있음.

## Classification Framework
- 학습과정은 AlexNet을 따라가며 (input 이미지를 크롭하는 방식만 다름), multinomial logistic regression 문제를 미니 배치로 쪼개어 momentum gradient descent로 최적화 수행.
- 파라미터
    - batch size: 256
    - momentum: 0.9 (과거 그래디언트의 90% + 현재 그래디언트의 10%)
    - weight decay: 0.0005(L2 regularization)
    - drop out: 앞 2개의 fc layer 0.5 drop out 적용
    - learning rate: 처음 0.01 이후, validation 셋의 accuracy 향상이 멈출때마다 10배씩 감소. 전체적으로 3번 감소.
    - 최종적으로 370k iterations(74 epochs)에서 학습 끝.
    - AlexNet에 비해 학습 파라미터도 많고, 모델의 깊이가 깊지만, 오히려 converge 하는데에 epoch 수는 적다고 함. 이유는 1) 더욱 깊고, 작은 필터의 conv layer 사용 2) 특정 layer들의 pre-initialization 때문이라고 생각한다고 함.
> - pre-initialization: 작은 네트워크를 먼저 학습시킨 후, 그 결과로 얻은 가중치를 깊은 네트워크의 초기 가중치로 사용. 모델이 얕을때는 무작위 가중치 초기화만으로도 충분히 학습이 가능하지만, 깊은 네트워크에서는 잘못된 초기화로 gradient vanishing이나 gradient exploding 문제 발생 가능. 네트워크의 일부 레이어만 사전 학습된 가중치를 사용하면, 나머지 레이어는 무작위 초기화를 통해 다양한 특징을 학습할 수 있음. VGGNet에서는 1~4번째 conv layer와 fc layer 3개에 대해서만 pre-initialization 적용.
- 학습 이미지 크기
    - 224X224로 고정
    - training scale을 'S'로 표시하며, single-scale training과 multi-scale training 지원.
    - single-scale에서는 AlexNet과 마찬가지로 S=256, S=384 지원.
    - multi-scale의 경우 Smin=256, Smax=512 범위에서 무작위 선택. S=384로 미리 학습 시킨 후 S를 무작위로 선택해가며 fine tuning. S를 무작위로 바꿔가면서 학습 시킨다고 하여 scale jittering이라고 함.
    - 크기 조정한 이미지에서 무작위로 224X224 크기 선택해서 학습.
- Testing
    - 신경망의 마지막 3 fc layer를 convolution layers로 변환해서 사용.
    - 첫번째 fc layer는 7X7 Conv로, 2~3번째 layer는 1X1 conv로 변환. Fully-convolution Networks라고 부름.
    - 신경망이 convolution 레이어로만 구성될 경우 입력 이미지의 크기 제약이 없어짐. 이에 따라 하나의 입력 이미지를 다양한 스케일로 사용한 결과들을 앙상블하여 이미지 분류 정확도 개선이 가능해짐.
    
## Conclusion
- 실험에서 네트워크의 깊이를 최대 19 레이어까지만 사용한 이유는, 해당 실험의 데이터에서는 분류 오차율이 VGG-19에서 수렴했기 때문. 학습 데이터 세트가 충분히 많다면 더 깊은 모델이 더 유용할 수 있음.
---
추가적으로 공부할 부분
~~- 1등한 GoogLeNet 공부. fc layer를 사용하지 않았다고 하는데 어떤 방법으로 모델을 개선한건지.~~


Reference:
https://inhovation97.tistory.com/44
https://codebaragi23.github.io/machine%20learning/1.-VGGNet-paper-review/
