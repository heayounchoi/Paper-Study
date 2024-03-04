- 학습하는 과정 자체를 전체적으로 안정화하여 학습 속도를 가속 시킬 수 있는 방법
- 정규화를 하는 이유는 학습을 더 빨리 하기 위해서 또는 local optimum 문제에 빠지는 가능성을 줄이기 위해서임

<img src="https://velog.velcdn.com/images/heayounchoi/post/2c566ea4-00b6-42d3-8ef2-a95bba440fe0/image.png" width="50%">

- 배치 정규화 논문에서는 학습에서 불안정화가 일어나는 이유를 internal covariance shift라고 주장하는데, 이는 네트워크의 각 레이어나 activation마다 입력값의 분산이 달라지는 현상을 뜻함
- covariate shift: 이전 레이어의 파라미터 변화로 인하여 현재 레이어의 입력의 분포가 바뀌는 현상
- internal covariate shift: 레이어를 통과할때마다 covariate shift가 일어나면서 입력의 분포가 약간씩 변하는 현상

<img  src="https://velog.velcdn.com/images/heayounchoi/post/cfccc4ed-4b2a-46e2-889c-2e63427a369e/image.png" width="50%">

- 이 현상을 막기 위해서 간단하게 각 레이어의 입력의 분산을 평균 0, 표준편차 1인 입력값으로 정규화 시키는 방법을 생각해볼 수 있음 (Whitening)
- 이 방법은 들어오는 입력값의 특징들을 uncorrelated 하게 만들어주고, 각각의 분산을 1로 만들어주는 작업임
- 이런 작업은 covariance matrix를 계산하고, 그 역행렬을 구하는 등의 복잡한 계산을 요구해 전체 학습 과정의 계산량이 크게 증가할 수 있음 (상관관계 제거를 위한 과정)
- 또한 특정 파라미터의 영향을 무시하게 될 수 있음
- 그리고 화이트닝은 역전파와 독립적으로 진행되기 때문에 특정 파라미터가 계속해서 커지는 문제가 발생할 수 있음
- 이러한 화이트닝의 문제점을 해결하도록 한 트릭이 배치 정규화임
- 배치 정규화는 평균과 분산을 조정하는 과정이 별도의 과정으로 떼어진 것이 아니라, 신경망 안에 포함되어 학습 시 평균과 분산을 조정하는 과정 역시 같이 조절됨

<img src="https://velog.velcdn.com/images/heayounchoi/post/c1fb3450-3328-484c-9ecd-bc71ce2f09bc/image.png" width="50%">

- 미니배치의 평균과 분산을 이용해서 정규화 한 뒤에, scale 및 shift를 감마값, 베타값을 통해 실행함
- 감마와 베타는 학습 가능한 파라미터로, 역전파를 통해 학습됨
- 이들은 데이터를 계속 정규화하게 되면 활성화 함수의 비선형 같은 성질을 잃게 되는데, 이러한 문제를 완화하기 위함
- inference 시에는 학습 과정에서 미리 계산해 둔 미니 배치들의 평균과 분산의 이동평균을 사용함
- 배치 정규화를 적용하면 베타 파라미터가 바이어스의 역할을 수행해서 바이어스가 제거될 수 있는 것임
