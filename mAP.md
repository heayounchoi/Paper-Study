<img src="https://velog.velcdn.com/images/heayounchoi/post/e72bb1cc-72cf-4958-b792-99ab0657e720/image.png" width="50%">

---
**IoU (Intersection over Union)**
<img src="https://velog.velcdn.com/images/heayounchoi/post/f1bb040e-0838-40f7-9357-90bc5857827f/image.png" width="50%">
- IoU threshold가 0.5일 경우 IoU의 계산 결과 값이 0.5 이상이면 true positive, 미만이면 false positive로 판단

---
**Precision**
- 정확도
- '검출 결과들' 중 옳게 검출한 비율
<img src="https://velog.velcdn.com/images/heayounchoi/post/50ee5569-0fd6-4c6b-82d1-d3196c091e41/image.png" width="50%">

---
**Recall**
- 재현율
- '실제 옳게 검출된 결과물' 중에서 옳다고 예측한 것의 비율
<img src="https://velog.velcdn.com/images/heayounchoi/post/e0d2dbce-6bc3-4410-8f50-689d5e54db72/image.png" width="50%">

---
- 일반적으로 precision과 recall은 서로 반비례 관계를 가짐
- 따라서 precision과 recall의 성능 변화 전체를 확인해야 함
- 대표적인 방법이 precision-recall 그래프

---
**Precision-Recall graph**
- 물체를 검출하는 알고리즘의 성능을 평가하는 방법 중 하나로 confidence 레벨에 대한 threshold 값의 변화에 따라 precision과 recall 값들도 달라는 것을 그래프로 나타낸 것
- confidence는 검출한 것에 대해 알고리즘이 얼마나 정확하다고 생각하는지 알려주는 값
<img src="https://velog.velcdn.com/images/heayounchoi/post/c61b8112-16bf-4ce7-83b4-51393bf29062/image.png" width="50%">
<img src="https://velog.velcdn.com/images/heayounchoi/post/cc773ffa-cb49-4969-b5f9-346de6705fc3/image.png" width="50%">

---
**Average Precision (AP)**
- precision-recall graph 선 아래 쪽의 면적
- 높을수록 성능이 전체적으로 우수하다는 의미
- 보통 계산 전에 그래프를 단조적으로 감소하는 그래프로 변경해줌
<img src="https://velog.velcdn.com/images/heayounchoi/post/294923cc-8e10-4aeb-b4dd-30491431dc6d/image.png" width="50%">
- mAP는 물체 클래스가 여러 개인 경우 각 클래스당 AP를 구한 다음에 그것을 모두 합한 후 클래스의 개수로 나눈 것
