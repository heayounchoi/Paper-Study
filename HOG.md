# HOG(Histogram of Oriented Gradient)
- gradient vectors
    - 기울기 벡터란 영상 내 하나의 픽셀을 기준으로 주변 픽셀에 대한 기울기를 나타내는 벡터
<img width="473" alt="Pasted Graphic 20" src="https://github.com/heayounchoi/Paper-Study/assets/118031423/627678fd-e0b7-4edf-8fc5-862a899c6839">

	- 빨간 점으로 표시된 픽셀을 기준으로 왼쪽의 gray scale 값은 56이고, 오른쪽의 값은 94임
	- gray scale의 경우 0이면 검은색이고 255면 흰색. 따라서 gray scale 값 94가 56보다 밝음
	- 이때 빨간 점으로 표시된 픽셀 입장에서 x축 방향의 변화량은 (94-56) = 38
	- y축 방향의 기울기 변화량은 (93-55) = 38
<img width="412" alt="Pasted Graphic 21" src="https://github.com/heayounchoi/Paper-Study/assets/118031423/5d57eefa-c4a6-4247-9d65-d16723336238">

 	- x축 방향의 기울기 변화량, y축 방향 기울기 변화량을 함께 표현한 값이 gradient vector.
<img width="100" alt="Pasted Graphic 22" src="https://github.com/heayounchoi/Paper-Study/assets/118031423/0be684c2-b614-4e93-a1c5-756726329cbd">
<img width="482" alt="Pasted Graphic 23" src="https://github.com/heayounchoi/Paper-Study/assets/118031423/3dde9cf5-4ec1-4388-8555-ad2887227e73">
<img width="367" alt="Pasted Graphic 24" src="https://github.com/heayounchoi/Paper-Study/assets/118031423/0c152a0b-36a4-4779-a027-d28b0a04b050">

- Pixels, Cells, Blocks
    - Pixels: 픽셀
    - Cells: 픽셀들을 몇 개 묶어서 소그룹으로 만든 것
    - Blocks: 셀을 몇개 묶어서 그룹으로 만든 것
- HOG
    - HOG는 보행자 검출을 위해 만들어진 특징 디스크립터
    - 이미지 경계의 기울기 벡터 크기와 방향을 히스토그램으로 나타내 계산함
    - 디스크립터를 만들기 위해서는 영상 속에서 검출하고자 하는 영역을 잘라내야 함 (window)
    - 일반적으로 보행자 검출을 위한 윈도 사이즈를 64x128 픽셀 크기로 하고, 셀의 크기를 일반적으로 8x8 pixels로 함
    - 해당 윈도우에 소벨 필터를 적용해 경계의 기울기 gx, gy를 구하고, 기울기의 방향과 크기를 계산함
    - 하나의 셀을 기준으로 히스토그램 계산
<img width="472" alt="Pasted Graphic 25" src="https://github.com/heayounchoi/Paper-Study/assets/118031423/c2817b8d-6104-48d7-908c-cc7470d2fb0e">

    - 비슷한 이미지는 서로 비슷한 히스토그램을 그림. 히스토그램의 유사성을 바탕으로 비슷한 이미지를 검출함
    - 히스토그램 계산을 마치면 정규화 과정을 거침. 경계 값 기울기는 밝기에 민감하기 때문에 민감성을 없애주기 위해 정규화를 하는 것
    - 정규화를 블록 단위로 나눔. 이때 블록의 크기는 일반적으로 셀 크기의 2배
    - 각 블록은 전체 윈도우를 순회하면서 정규화를 함
