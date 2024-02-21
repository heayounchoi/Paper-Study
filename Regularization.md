# Regularization: L1, L2 penalty term (weight decay)
- regularization(규제화)란, 지도학습 모델의 성능 평가 단계에서 overfitting이 일어나는 경우에 이를 해결하기 위해서 사용하는 방법
- 과적합이 일어나는 이유
    - 1) 모형이 너무 복잡한 경우, 즉 모형의 파라미터의 갯수가 너무 많은 경우
    - 2) 학습을 통해서 도출된 수학적 모형이 독립변수의 값의 변화에 너무 민감하게 반응하는 경우, 즉 독립변수에 붙어있는 파라미터의 값이 너무 큰 경우
- 규제화는 기존의 비용함수에 penalty term을 더해서 새로운 비용함수로 사용하는 방법
- **Lp norm**

<img width="311" alt="Pasted Graphic 30" src="https://github.com/heayounchoi/Paper-Study/assets/118031423/dc335f44-d8b5-47d8-ba01-ea6f49ff8af1">

- L1 norm은 p에 1을, L2 norm은 p에 2를 대입해줌

<img width="356" alt="Pasted Graphic 31" src="https://github.com/heayounchoi/Paper-Study/assets/118031423/ca4c1efd-e33a-4e49-b7e8-311f80a6b4b0">
<br>
<img width="258" alt="Pasted Graphic 34" src="https://github.com/heayounchoi/Paper-Study/assets/118031423/6142c236-8a67-466a-9daf-2f720e0411e7">
<br>
<img width="356" alt="Pasted Graphic 32" src="https://github.com/heayounchoi/Paper-Study/assets/118031423/a73a119b-50f5-4fc6-8ad2-3936017580da">

- L1 penalty term을 사용하면 이렇게 되는데, 여기서 람다는 하이퍼파라미터로 페널티의 강도를 의미함. 람다의 크기를 키우면, 키울수록 페널티항이 커지기 때문에 기존의 파라미터 절댓값이 더 많이 줄어들게 됨

<img width="561" alt="Pasted Graphic 35" src="https://github.com/heayounchoi/Paper-Study/assets/118031423/3820077e-e3fe-4627-8276-652078d8b027">

- L1 페널티 항의 특징은 람다의 값을 일정 값 이상으로 높이면, 파라미터의 최적값이 0까지도 줄어들 수 있다는 점. 이 뜻은 모델의 민감도뿐만 아니라, 복잡도까지도 줄일 수 있다는 뜻이고, 복잡도를 줄인다는 것은 과적합이 일어나는 확률을 줄일 수 있다는 뜻 -> 모델의 성능을 높일 수 있음
<img width="350" alt="Pasted Graphic 36" src="https://github.com/heayounchoi/Paper-Study/assets/118031423/9e8c8b56-c8ef-405c-b810-f6f985c258f7">

- L2 페널티 항은 파라미터의 최적값이 0까지 줄어들지 않음
- 그래서 모델의 복잡도까지 줄이기 위해서는 L1 페널티 항을 사용해야 함
- 둘 중 어떤걸 사용해야 하는지는 일단 두개 다 사용해보고 더 성능이 좋은 걸 선택해서 적용해야 함
