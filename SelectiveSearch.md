# Selective Search
- sliding window 방식은 일정 크기의 window가 이미지의 모든 영역을 sliding 하면서 객체를 찾기 때문에 계산량이 매우 많다는 단점이 있음
- 이 단점을 보완하기 위해서 객체가 있을만한 후보 영역을 미리 찾고 그 영역 내에서만 객체를 찾는 방식을 region proposal 방식이라고 함
- selective search는 region proposal의 대표적인 방법 중 하나
---
**Selective Search 과정**
- selective search 이전에 물체가 있을만한 영역을 모두 조사해보는 exhaustive search 방법이 있었음
- selective search는 이에 segmentation을 결합해 exhaustive search를 개선시킨 방법
<img src="https://velog.velcdn.com/images/heayounchoi/post/d6921472-a930-49a7-9185-7e58f56267c3/image.png" width="70%">

1) 초기에는 원본 이미지로부터 각각의 object들이 1개의 개별 영역에 담길 수 있도록 수많은 영역들을 생성함(이때 object들을 놓치지 않기 위해서 Over Segmentation을 해줌)

2) 아래 그림의 알고리즘에 따라서 유사도가 높은 것들을 하나의 segmentation으로 합쳐줌
<img src="https://velog.velcdn.com/images/heayounchoi/post/907f391f-7ffd-4e14-a937-63c0c2add6a5/image.png" width="70%">

> - R = 최초 segmentation을 통해서 나온 초기 n개의 후보 영역들
> - S = 영역들 사이의 유사도 집합
> - 색상, 무늬, 크기, 형태를 고려하여 각 영역들 사이의 유사도를 계산
> - 유사도가 가장 높은 r들의 영역을 합쳐 새로운 r 영역을 생성
> - 합쳐진 기존 영역과 관련된 유사도는 S 집합에서 삭제
> - 새로운 r 영역과 나머지 영역의 유사도를 계산해 r의 유사도 집합 S 생성
> - 새로운 영역의 유사도 집합 S와 영역 r을 기존의 S, R 집합에 추가

3) 2번 과정을 여러번 반복해 최종 후보 영역 도출
<img src="https://velog.velcdn.com/images/heayounchoi/post/9fb9d0e7-07a9-4611-a1ba-1f07a8a71e0c/image.png" width="70%">
최종 후보 영역들에 대해서 CNN을 통한 classification과 bounding box regression을 해주면 object detection이 수행되는 것
