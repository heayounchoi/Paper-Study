# Weights Initialization
- 신경망을 학습할 때 손실 함수에서 출발 위치를 결정하는 방법이 모델 초기화
- 특히 가중치는 모델의 파라미터에서 가장 큰 비중을 차지함. 가중치 초기화 방법에 따라 학습 성능이 크게 달라짐
- 신경망의 가중치를 모두 0으로 초기화했다고 가정하면, 가중합 결과가 항상 0이 되어 activation function은 가중 합산 결과인 0을 입력 받아서 늘 같은 값을 출력하게 됨

![image](https://github.com/heayounchoi/Paper-Study/assets/118031423/5d6ad46c-f9b1-4ccb-abe6-196cd93dfa8b)

- 결과적으로 의미 없는 결과가 만들어지며, 가중치가 0이면 학습도 진행되지 않음
- 가중치를 0.1로 초기화한다고 가정했을때, 모든 은닉 뉴런에 대해 입력과 가중치가 같기 때문에 가중 합산 결과도 같고, 활성 함수의 실행 결과도 같음

![image](https://github.com/heayounchoi/Paper-Study/assets/118031423/689e62fa-086e-4c9c-b1ef-e326fb1f97c1)

- 실제 뉴런은 2개지만 1개만 있는 것과 똑같은 의미임. 정보가 반으로 줄어들고, 연산 결과도 부정확해짐
- 이런걸 신경망에 symmetry(대칭성)이 생긴다고 함
- 가우시안 분포를 따르는 난수를 이용해서 초기화한다고 하면, 아주 작은 난수로 초기화할 경우 계층이 깊어질수록 출력이 점점 0으로 변화하고, 아주 큰 난수로 초기화할 경우 점점 -1이나 1로 변화함

![image](https://github.com/heayounchoi/Paper-Study/assets/118031423/bb3cd21e-c5e5-4d02-b0b1-69584f8498f4)
![image](https://github.com/heayounchoi/Paper-Study/assets/118031423/e0399bcd-9011-42ab-9789-718ff168c91f)

- 따라서 데이터가 계층을 통과하더라도 데이터의 크기를 줄이거나 늘리지 않고, 유지해주는 가중치로 초기화해야함
- xavier initialization은 sigmoid 계열의 활성 함수를 사용할 때, 가중치를 초기화하는 방법
- 입력과 출력의 분산을 같게 만들기 위해 식을 유도해보면 가중치의 분산이 입력 데이터 개수 n에 반비례하도록 1/n이라는 식이 나옴

![image](https://github.com/heayounchoi/Paper-Study/assets/118031423/79af001d-4fcf-4fce-a0c1-943a3fc6abc7)

- xavier 초기화 했을 경우 레이어를 통과해도 분산이 잘 유지되는 모습을 확인할 수 있음
- 하지만 활성 함수가 ReLU일때 xavier 초기화를 사용하면 데이터의 크기가 점점 작아짐
- xavier는 활성 함수를 선형 함수로 가정하기 때문

![image](https://github.com/heayounchoi/Paper-Study/assets/118031423/13c8805c-f258-4697-a68f-54607d4584e9)

- ReLU는 비활성화 구간때문에 분산이 절반으로 줄어들게 됨

![image](https://github.com/heayounchoi/Paper-Study/assets/118031423/9fba9997-acef-4a61-a10f-32545832fca6)

- ReLU 사용시 이처럼 출력의 분산이 절반으로 줄어들기 때문에, 가중치의 분산을 두배로 키움 (2/n이 됨)

![image](https://github.com/heayounchoi/Paper-Study/assets/118031423/5a0e7e70-73af-42a8-9e8a-432f3c4d7360)

- ReLU 특성상 0에 몰려있지만, 나머지 데이터는 양수 구간에 골고루 퍼져 있는 것을 확인할 수 있음
