# SIFT(Scale-Invariant-Feature Transform)
- 이미지의 scale, rotation에 불변하는 특징점을 추출하는 알고리즘
- 이런 것들에 불변하는 특징을 추출하는 것은 이후 두 이미지에 대한 **[R|T] Matrix**를 구하는데 활용될 수 있음
- SIFT 알고리즘은 크게 아래 4가지 과정으로 요약할 수 있음
    - (1) Scale-space extrema detection: 이미지의 크기를 변화시켜 극대점, 극소점 detection -> keypoint candidates 찾음
    - (2) keypoint localization: feature matching 시 불안정할 keypoint 제거 및 keypoint를 정확하게 위치시킴
    - (3) orientation assignment: 각 keypoint들의 direction, magnitude 등의 orientation 결정
    - (4) keypoint descriptor: keypoint를 표현하기 위한 descriptor 생성
- (1)로 인해 scale-invariance한 성질을 갖게 되고, **(3)으로부터 계산한 keypoint의 orientation을 (4)에서 구한 keypoint descriptor에서 빼주면 rotation-invariance 성질을 갖게 됨**
- Scale-space extrema detection
    - 이 과정을 통해 scale-invariant한 feature를 추출할 수 있음
    - scale-space는 original 이미지의 scale을 변화시켜 여러 scale의 이미지를 모아놓는 집합체
    - scale이 높아질수록 detail이 blur 됨. 그래서 신호처리 관점에서는 image에 gaussian filter를 convolution 한 것 과 scale을 키우는 것은 완전히 동일한 과정
<img width="50%" alt="Pasted Graphic 10" src="https://github.com/heayounchoi/Paper-Study/assets/118031423/12cc6b3e-ddea-46bf-b754-e298cca6015c">
    - 점차적으로 블러된 이미지를 얻기 위해 시그마에 상수 k를 계속 곱해서 convolution 함. 이때 시그마가 2배가 될때까지 만들어진 사진들은 한 octave로 묻고, 그 다음 이미지를 1/2로 downsampling 한 다음 다시 시그마에 상수 k를 곱해서 convolution 함. 시그마가 2배가 됐을때 downsampling 하는 이유는 연산량을 줄이기 위함. 시그마가 2배가 됐다는건 이미지를 2배 멀리서 본 것과 동치. 그러므로 이미지 크기를 1/2로 줄여도 상관없으며 동시에 연산량이 줄어드는 이점을 얻음.
- Dog(Difference of Gaussian)
    - Laplacian of Gaussian(LoG)를 사용하면 이미지 내에서 keypoint들을 추출할 수 있음(edge, corner)
    - 하지만 LoG는 많은 연산을 요구하기 때문에 비교적 간단하면서도 비슷한 성능을 낼 수 있는 Difference of Gaussian(DoG)를 사용함. 전 단계에서 얻은 옥타브 내에서 인접한 두 개의 블러 이미지들끼지 빼주면 됨
<img width="50%" alt="Pasted Graphic 11" src="https://github.com/heayounchoi/Paper-Study/assets/118031423/4e99bde0-e949-4e22-9b05-a58e79cd3ee1">
<img width="50%" alt="Pasted Graphic 12" src="https://github.com/heayounchoi/Paper-Study/assets/118031423/0d438637-bb14-48f6-9409-e8f12ff70c8a">

- keypoint localization
    - DoG 이미지들에서 keypoint 찾기
    - 먼저 DoG 이미지들 내에서 극대값, 극소값들의 대략적인 위치를 찾음
        - 각각의 octave에서 scale이 다른 3개의 이미지를 겹쳐놓음
        - 1개의 x는 검사하는 point, 녹색 circle은 26개의 neighbor pixels
        - x의 값이 26개의 neighbor pixels와 비교해서 가장 크거나 가장 작으면 keypoint로 간주
        - 하지만 이렇게 구한 값은 대략적인 것이고, 실제 극값의 위치는 pixels의 사이 공간에 있을 확률이 높음
        - 진짜 극값의 위치에 접근할 수 없기 때문에 수학적으로 계산해야 함
        - D는 DoG 이미지, X는 (x, y, standard deviation)을 나타냄
        - 식을 X에 대해 미분해서 0이 되는 지점이 극값
        - 이러한 방법은 알고리즘이 좀 더 안정적인 성능을 낼 수 있게 도와줌
  
<img width="50%" alt="Pasted Graphic 13" src="https://github.com/heayounchoi/Paper-Study/assets/118031423/8f51432c-ecc8-482b-84c7-ca3e72d4b1c0">

- bad keypoint 제거
    - 수학적으로 계산된 keypoints 중에서 특정 threshold 보다 낮은 keypoint들을 제거함
    - DoG는 edge를 좀 더 민감하게 찾아내기 때문에 noise가 edge 위에 위치할 경우 keypoint로 오인할 위험이 있음
    - 그래서 edge 위에 위치한 keypoints 또한 제거하여 코너에 위치한 keypoints만 남겨놓음
- rotation invariant
    - 사진을 어떻게 뒤집든 그 점이 여기에 있다는 것을 특징할 수 있어야 함
    - 이 정보를 만들기 위해 먼저 keypoint의 gradient를 구하고(orientation assignment) 그 다음 keypoint 주변 픽셀들의 gradient를 구한 다음, 주변 픽셀들의 gradient에서 keypoint gradient를 빼줌. 이렇게 만들어진 정보를 나열한 vector를 keypoint descriptor이라고 함
    - keypoint의 gradient와 주변 pixel의 gradient는 이미지의 rotation에 따라 값이 변함. 다만 변하지 않는 것은 주변 pixel과 keypoint의 상대적인 방향. 이 상대적인 gradient 값으로 descriptor를 만들 수 있고 이 descriptor는 rotation에 대해 invariant 함
- orientation assignment
<img width="50%" alt="Pasted Graphic 14" src="https://github.com/heayounchoi/Paper-Study/assets/118031423/7ee80192-56a2-4edc-b9a6-778b1f306b40">

    - gradient의 크기와 방향을 구하고 가로축이 방향, 세로축이 크기인 histogram을 그림. histogram에서 가장 큰 값을 가지는 방향(각도)을 keypoint의 방향으로 설정. 만약 가장 큰 keypoint 방향의 80%보다 큰 각도가 존재한다면, 그 각도도 keypoint의 orientation으로 설정.
<img width="50%" alt="Pasted Graphic 15" src="https://github.com/heayounchoi/Paper-Study/assets/118031423/6a673439-6791-401e-8b29-dc37af411aaa">

- keypoint descriptor
    - keypoint를 중심으로 16x16 window를 세팅하고, 이 윈도우를 4x4의 크기를 가진 16개의 작은 윈도우로 구성
<img width="50%" alt="Pasted Graphic 16" src="https://github.com/heayounchoi/Paper-Study/assets/118031423/175eaa90-75a0-47d5-8765-33d731ba0a90">

	- 16개의 작은 윈도우에 속한 pixel들의 gradient의 크기와 방향을 계산. 
	- 그리고 8개의 bin을 가진 histogram을 그림. 
	- 이전과 마찬가지의 방법이지만 bin의 수만 8개. 
	- 결국 16개의 윈도우에 8개 방향으로 세팅이 됐기 때문에 16x8=128개의 숫자(feature vector)를 가진 descriptor를 만들 수 있음
	- 이미지가 회전하면 모든 gradient의 방향이 바뀌기 때문에 feature vector도 변하게 됨.
	- 따라서 회전된 이미지에서 feature vector가 변하지 않도록 하기 위해 16개 각각의 4x4 윈도우에서 keypoint의 방향을 빼줌
