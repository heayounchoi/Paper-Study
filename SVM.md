# SVM(Support Vector Machine)
SVM(Support Vector Machine)
- 기존 분류 방법들이 ‘오류율을 최소화’하려는 목적으로 설계되었다면, SVM은 두 부류 사이에 존재하는 ‘여백을 최대화’하려는 목적으로 설계됨

￼<img width="319" alt="Pasted Graphic 28" src="https://github.com/heayounchoi/Paper-Study/assets/118031423/69e34f13-4c74-4e99-a676-51fb797f0885">

- 두 데이터를 구분하는 선을 decision boundary라고 함
- margin을 최대화하면 robustness도 최대화 됨. robust 하다는건 outlier의 영향을 받지 않는다는 뜻
- 무작정 margin을 크게 하는 구분선을 택하는 것이 아니라, 데이터를 정확히 분류하는 범위를 먼저 찾고, 그 범위 안에서 margin을 최대화하는 구분선을 택함
- outlier가 있을 경우 어느정도 outlier를 무시하고 최적의 구분선을 찾음
- kernel trick

￼<img width="531" alt="Pasted Graphic 29" src="https://github.com/heayounchoi/Paper-Study/assets/118031423/4ca61eb0-35c4-4374-8219-57a6e540fccf">

	- 왼쪽 좌표의 경우 빨간 포인트와 파란 포인트를 구분할 수 있는 linear line이 없음
	- low dimensional space에서 high dimensional space로 매핑해(kernel trick) 오른쪽 좌표가 되면, linearly seperable해짐
	- 이 line을 다시 저차원 공간으로 매핑하면 non linear separable line이 구해짐
