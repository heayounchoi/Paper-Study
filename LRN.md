# LRN(Local Response Normalization)
- LRN layer implements the lateral inhibition
- lateral inhibition(측면 억제): 신경생리학 용어로, 한 영역에 있는 신경 세포가 상호 간 연결되어 있을 때 한 그 자신의 축색이나 자신과 이웃 신경세포를 매개하는 중간신경세포를 통해 이웃에 있는 신경 세포를 억제하려는 경향
- AlexNet에서 LRN을 사용했는데, ReLU를 사용했기 때문
- ReLU는 양수의 방향으로는 입력의 값을 그대로 사용해서 conv나 pooling 시 매우 높은 하나의 픽셀값이 주변의 픽셀에 영향을 미치게 됨. 이런 부분을 방지하기 위해 다른 activation map의 같은 위치에 있는 픽셀끼리 정규화를 함
