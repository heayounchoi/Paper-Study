### Single Shot MultiBox Detector
- R-CNN 계열의 2-stage detector는 region proposals와 같은 다양한 view를 모델에 제공해 높은 정확도를 보여줌
- 하지만 region proposals를 추출하고 이를 처리하는 과정에서 많은 시간이 걸려 detection 속도가 느림
- 반면 YOLO v1은 원본 이미지 전체에 통합된 네트워크로 처리하기 때문에 detection 속도가 매우 빠름
- 반면 grid cell별로 2개의 bounding box만을 선택하여 상대적으로 적은 view를 모델에 제공해 정확도가 떨어짐
- SSD는 다양한 view를 활용하면서 통합된 network 구조를 가진 1-stage detector로서 높은 정확도와 빠른 속도를 가짐
---
**Preview**
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FcEHNEN%2FbtqSa9Yb4qh%2FtxKuruXq2rNmYYQzTeXHn1%2Fimg.png">
- SSD는 VGG16을 base network로 사용하고 auxiliary network를 추가한 구조를 가짐
- 두 네트워크를 연결하는 과정에서 fc layer를 conv layer로 대체하면서 detection 속도가 향상됨
- SSD는 conv network 중간의 conv layer에서 얻은 feature map을 포함시켜, 총 6개의 서로 다른 scale의 feature map을 예측에 사용함
- 또한 feature map의 각 cell마다 서로 다른 scale과 aspect ratio를 가진 bounding box인 default box를 사용해 객체의 위치를 추정함

---
**Main Ideas**

**_1) Multiscale feature maps_**
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fd9o18C%2FbtqR6v80uKO%2FjLZ1aj0nkGJYGpk1bwktIK%2Fimg.png">
- SSD 모델의 핵심적인 아이디어는 다양한 scale의 feature map을 사용한다는 점
- 단일한 scale의 feature map을 사용할 경우, 다양한 크기의 객체를 포착하는 것이 어렵다는 단점이 있음
- 이러한 문제를 해결하기 위해 논문의 저자는 SSD network 중간에 존재하는 conv layer의 feature map들을 추출하여 detection 시 사용하는 방법을 제안함

**_2) Default boxes_**
- 원본 이미지에서 보다 다양한 크기의 객체를 탐지하기 위해 feature map의 각 cell마다 서로 다른 scale과 aspect ratio를 가진 default box 생성
- default box는 faster r-cnn 모델에서 사용하는 anchor box와 개념적으로 유사하지만 서로 다른 크기의 feature map에 적용한다는 점에서 차이가 있음
- default box의 scale = $s_k$
- $s_k$는 원본 이미지에 대한 비율을 의미함
<img src="https://velog.velcdn.com/images/heayounchoi/post/f0528edd-1eb0-492a-bfbe-76222f975141/image.png">

- aspect ratio $a_r ∈$ \[1, 2, 3, 1/2, 1/3]
- default box의 너비 $w^a_k=s_k\sqrt a_r$
- 높이 $h^a_k=s_k/\sqrt a_r$
- aspect ratio가 1:1인 경우 scale이 $s'_k=\sqrt {s_ks_{k+1}}$인 default box 추가적으로 사용
- feature map의 scale이 작아질수록 default box의 scale은 커짐
- feature map의 크기가 작아질수록 더 큰 객체를 탐지할 수 있음을 의미함
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2FFEYNv%2FbtqSgLQrtnT%2FErJBo41yKDjKNwFA2JC99K%2Fimg.png">

**_3) Predictions_**
- 최종 예측을 위해 서로 다른 scale의 feature map을 추출한 후 3x3(stride=1, padding=1) conv 연산 적용
- 이때 default box의 수를 k, 예측하려는 class의 수를 c라고 할때, output feature map의 channel 수는 k(4+c)가 되도록 설꼐
- 이는 각 feature map의 cell이 k개의 default box를 생성하고 각 box마다 4개의 offset과 class score를 예측한다는 것을 의미함
<img src="https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fblog.kakaocdn.net%2Fdn%2Fbpcmxo%2FbtqRXGXUIcS%2FqFfch5Xk8keVDixu3xMRM0%2Fimg.png">

**_4) Matching strategy_**
- ground truth box와 가장 큰 jaccard overlap(IoU와 같음)을 가지는 default box와 0.5 이상인 box는 모두 positive, 이외는 모두 negative로 label

**_5) Loss function_**
<img src="https://velog.velcdn.com/images/heayounchoi/post/b6fbc5a2-2090-4104-ac7e-b916d9fc759a/image.png">

---
**Training SSD**

**_1) 전체 network 구성하기(base network + auxiliary network)_**
- pre-trained된 VGG16 모델을 불러와 마지막 2개의 fc layer를 conv layer로 대체
- 최종 output feature map의 크기가 1x1이 되도록 auxiliary network를 설계

**_2) 이미지 입력 및 서로 다른 scale의 feature map 얻기_**

**_3) 서로 다른 scale의 feature map에 conv 연산 적용하기_**

**_4) 전체 feature map 병합하기_**
- 3)번 과정에서 얻은 모든 feature map을 8732 x (21+4) 크기로 병합
- 이를 통해 default box별로 bounding box offset 값과 class score를 파악할 수 있음

**_5) loss function을 통해 SSD network 학습시키기_**
- feature map과 ground truth 정보를 활용하여 localization loss를 계산
- 이후 negative sample에 대하여 Cross entropy loss를 구한 후 loss에 따라 내림차순으로 정렬
- negative sample에서 loss가 높은 순으로 positive sample의 3배만큼의 수를 추출
- 이 과정은 모델이 어떤 negative sample을 객체로 잘못 예측했는지, 즉 어떤 negative sample이 모델에게 가장 혼란을 주었는지 파악하기 위함
- 이러한 hard negative mining 과정을 통해 얻은 hard negative sample과 positive sample을 사용하여 confidence loss를 계산
- 앞서 얻은 localization loss와 confidence loss를 더해 최종 loss를 구한 후 backward pass를 수행하여 network를 학습

---
**Detection**
- detection 시, 마지막 예측에 대하여 Non maximum suppression을 수행을 통해 겹치는 default box를 적절하게 제거해 정확도 향상
---
- PASCAL VOC 데이터셋으로 시험했을 때, Faster R-CNN 모델은 7FPS, mAP=73.2%, YOLO v1 모델은 45FPS, mAP=63.4%의 성능을 보인 반면 SSD 모델은 59FPS, mAP=74.3%라는 놀라운 속도와 detection 정확도를 보임
