# Region Proposal Network
- object detection의 핵심 역할
- RPN은 feature map을 input으로, RP를 output으로 하는 네트워크
- selective search의 역할을 온전히 대체함
- RPN은 Faster R-CNN에서 활용됐는데, Faster R-CNN의 목표는 selective search 없이 RPN을 학습하는 구조로 모델을 만드는 것
- 요약
> - 다양한 사이즈의 이미지를 입력 값으로 object score와 object proposal을 출력함
> - Faster R-CNN과 CNN을 공유함
> - feature map의 마지막 conv layer를 작은 네트워크가 sliding 하여 저차원으로 매핑함
> - regression과 classification을 수행함
---

**RPN 구성**
- RPN의 input 값은 이전 CNN 모델에서 뽑아낸 feature map
- region proposal을 생성하기 위해 feature map 위에 NxN window를 sliding window 시킴

1. 기본 Anchor Box
- anchor box: 미리 정의된 형태를 가진 경계박스
- faster R-CNN에서는 3개의 스케일과 3개의 비율을 사용하여 k=9개의 앵커를 사용함
<img src="https://velog.velcdn.com/images/heayounchoi/post/51bb395a-c9bc-4375-8f7f-259768879b65/image.png" width="50%">

2. Delta
- 기본 anchor의 크기와 위치를 조정하기 위한 값들
- 모델을 통해 학슴됨
- anchor 하아네 delta가 하나씩 대응함
- anchor 하나의 값의 구성은 (Y1, X1, Y2, X2)처럼 bounding box 형식의 구조를 갖고, delta 하나의 값의 구성은 (deltaCenterY, deltaCenterX, deltaHeight, deltaWidth)로 구성됨

3. Probability
- 각 anchor 내부에 객체가 존재할 확률
- RPN의 output 값은 객체가 존재할 것이라는 확률이 높고, 중복이 제거된 bounding box들임(ROI, Regions of Interest)
---
**RPN 내부에서 이루어지는 동작**
1. 입력
- CNN에서 추출된 이미지의 특징 맵
- anchors

2. bounding box 계산
- anchor는 delta와 결합해서 값들을 조정해야 함
- delta 값이 deep network를 통해 산출된 값이므로, 이 조정 과정을 거쳐야 실제 객체의 위치를 정확하게 표현하게 됨

3. sort 단계
- 입력으로 설정된 anchor box는 갯수도 너무 많고, 확률이 0인 부분도 bounding box 정보를 가지고 있음
- 따라서 계산 과정은 확률이 높은 객체에 대해서만 작업을 진행할 필요가 있음
- 그래서 2번에서 산출된 bounding box들 중에서 확률이 높은 객체를 사용함

4. Non Maximum Suppression (NMS)
- 동일한 클래스에 대해 높은-낮은 confidence 순서로 정렬
- 가장 confidence가 높은 bounding box와 IOU가 일정 이상인 bounding box는 동일한 물체를 detect 했다고 판단해 지움

<NMS 시행 전후>
<div style="display: flex; align-items: center; justify-content: center;">
	<img src="https://velog.velcdn.com/images/heayounchoi/post/30344497-49ea-4c19-8a28-41675f65236f/image.png" width="30%">
  <span>====>></span>
	<img src="https://velog.velcdn.com/images/heayounchoi/post/e0c13a58-b2b0-40cb-a424-f16589a15bd9/image.png" width="30%">
</div>

5. Merge
- NMS에서 여러 개의 후보 중 최적의 바운딩 박스만 남기고 나머지는 제거하기 때문에 메모리 상에서 구멍이 뚫린 형태가 됨(연속적이지 않은 형태)
- 연속된 형태로 모아주어야 다음에 이어지는 convolution과 같은 표준화된 절차를 진행할 수 있음
<br>
RPN은 이와 같이 이미지 내에서 객체 후보 영역(proposals)을 식별한 후, ROI Pooling이라는 과정을 통해 고정된 크기의 특징 맵(feature map)으로 변환하여 실제 객체 탐지 및 분류에 사용함
