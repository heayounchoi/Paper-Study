### Deformable Part Model

1. Sliding window
- 이미지 픽셀마다 (일정 간격의 픽셀을 건너뛰고) bounding box(window)를 그림
<img src="https://velog.velcdn.com/images/heayounchoi/post/3aa4c729-84ee-4cf4-aa3d-5c7186a27c4d/image.png" width="50%">

2. block-wise operation
- 하나의 이미지를 block 단위로 나눔
<img src="https://velog.velcdn.com/images/heayounchoi/post/12a8ad03-29a0-48f8-b10a-8794e871123f/image.png" width="50%">

3. SIFT or HOG block-wise orientation histogram
<img src="https://velog.velcdn.com/images/heayounchoi/post/7d1ae752-75da-4604-b8b2-4ecdbbf0333d/image.png" width="50%">
<img src="https://velog.velcdn.com/images/heayounchoi/post/633e8866-7b1d-4833-b309-89ea3c89c5eb/image.png" width="50%">

4. classification
- 다양한 template filters를 활용해서 template matching -> 합산 -> SVM classification
- bounding box의 features가 사람이 갖고 있는 특징(여러 template filter)을 포함하고 있다고 판단하면 해당 bounding box를 사람이라고 detect 함
<img src="https://velog.velcdn.com/images/heayounchoi/post/556fa140-ed2c-43d4-895e-1c3baf9e4199/image.png" width="50%">
<img src="https://velog.velcdn.com/images/heayounchoi/post/b171a6e7-fd7a-4b74-9d6a-bc98f58af1f5/image.png" width="50%">
