### Spatial Pyramid Pooling in Deep Convolutional Networks for Visual Recognition
---
**Abstract**
- 당시 CNN은 고정된 사이즈의 224x224 해상도의 이미지만 입력 받았음
- fc layer에서 고정길이 벡터만 받을 수 있기 때문
- 하지만 arbitrary한 이미지를 일정 사이즈로 제한하면, 성능에도 영향을 미침
- 따라서 해당 논문에서는 이러한 조건을 없애기 위한 spatial pyramid pooling이라는 풀링 기법을 제안함
- 이를 이용하면 이미지의 사이즈나 스케일에 상관없이 고정된 길이의 벡터를 생성할 수 있음
- SPPNet은 object detection에서 매우 유용한데, 이를 이용하면 CNN에 이미지를 딱 1번만 입력하면 됨
- R-CNN에서는 생성된 RoI 2000개를 전부 입력했었음
---
**기존 CNN의 문제점**

<img src="https://velog.velcdn.com/images/heayounchoi/post/5d8a650c-8c54-4d1f-aa48-bcd368d6bdda/image.png" width="50%">

---
**Spatial Pyramid Pooling Layer 작동원리**

<img src="https://velog.velcdn.com/images/heayounchoi/post/4e59c2cd-0177-4c54-8ef0-d0047ad81dc5/image.png" width="50%">

- 최종 feature map의 사이즈를 a x a라고 할때, 적용하려는 피라미드 풀링 사이즈가 n x n이면 윈도우 사이즈는 ceiling(a/n), stride는 floor(a/n)

---
**R-CNN 적용**
1) 이미지에 대해 Selective Search를 적용해 RoI 후보군 추출
2) 이미지를 crop/warp 하지 않고, 그대로 ConvNet에 입력 (2000 -> 1번의 입력)
3) 컨볼루션으로 나온 최종 feature map에 원본 이미지에서 압축된 비율과 동일하게 2000개의 RoI도 전부 축소시켜(projection) 이를 feature map에 적용
4) feature map으로부터 추출한 RoI feature에 SPP 적용
5) 압축한 벡터를 SVM에 입력해 class 분류
6) b-box 회귀 진행
7) NMS로 객체별 최종 b-box 예측
