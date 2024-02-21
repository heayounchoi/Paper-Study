# Gaussian filter
- 평균 필터는 대상 점을 주변 픽셀들의 평균값으로 대체하기 때문에 이미지를 blurring하는 효과를 가짐
- 평균 필터는 필터의 모든 값이 동일함
<img width="50%" alt="Pasted Graphic 37" src="https://github.com/heayounchoi/Paper-Study/assets/118031423/6c3bfb22-11ae-408f-ad25-09dbb2e936a8">

- 이러한 특성은 대상 점과 가까운 픽셀이 먼 픽셀보다 더 연관이 있다는 사실을 반영하지 못함
- 가까운 픽셀에 더 많은 가중치를 줄 필요가 있음

<img width="50%" alt="Pasted Graphic 38" src="https://github.com/heayounchoi/Paper-Study/assets/118031423/982d95d9-f189-4918-bb76-82da4fd403fd">

- 가우스 함수는 대상 점의 값이 가장 크고, 대상 점에서 멀어질수록 값이 작아지는 특징이 있음
- 가우스 함수를 이용해서 커널을 만들면 가우시안 필터가 됨

<img width="50%" alt="Pasted Graphic 39" src="https://github.com/heayounchoi/Paper-Study/assets/118031423/7b1808f5-02a3-4393-9feb-0a352d1d5e2c">
<img width="50%" alt="Pasted Graphic 40" src="https://github.com/heayounchoi/Paper-Study/assets/118031423/8fbed21f-4f58-4384-8dbf-609e629e2f20">

- 가우스 함수의 σ값은 표준편차를 나타냄
- 표준편차의 값이 클수록 분포는 완만해지고, 표준편차의 값이 작을수록 분포는 뾰족해짐
- σ값이 클수록 주변 픽셀들의 영향을 적게 받고, 블러링 효과가 크게 나타남
- 반대로 표준편차가 작을수록 주변 픽셀들의 영향을 많이 받고, 블러링 효과가 작아짐

<img width="50%" alt="Pasted Graphic 41" src="https://github.com/heayounchoi/Paper-Study/assets/118031423/58c2916b-53ef-4780-89ef-ab5c68e10068">

- 가우시안 필터는 low-pass filter
- low-pass filter는 이미지로부터 “high-frequency”를 제거하는 필터임
- 이미지에 가우시안 필터를 convolve 하면 블러링된 이미지를 얻을 수 있음
- 블러링된 이미지를 원래 이미지에서 빼면 이미지의 디테일한 부분을 얻을 수 있음

<img width="50%" alt="Pasted Graphic 42" src="https://github.com/heayounchoi/Paper-Study/assets/118031423/7104b1f3-1da2-42f5-88b9-7b6e166308e3">

- 이렇게 이미지의 디테일한 부분을 추출하는 연산을 high-pass filter라고 함
- 원본 이미지에 디테일한 부분을 더하면 선명한 이미지를 얻을 수 있음

<img width="50%" alt="Pasted Graphic 43" src="https://github.com/heayounchoi/Paper-Study/assets/118031423/e75897ea-5712-4473-9b05-376d980e53ca">
