### Singular Value Decomposition, SVD

- 특이값 분해(Singular Value Decomposition, SVD)가 말하는 것: 직교하는 벡터 집합에 대하여, 선형 변환 후에 그 크기는 변하지만 여전히 직교할 수 있게 되는 그 직교 집합은 무엇인가? 그리고 선형 변환 후의 결과는 무엇인가?
> - 벡터 집합이 직교한다는 것은 집합에 속한 벡터들의 dot product가 0이라는 것을 의미함
---
**특이값 분해의 정의**
- SVD는 임의의 $m×n$ 차원의 행렬 $A$에 대하여 다음과 같이 행렬을 분해할 수 있다는 ‘행렬 분해(decomposition)’ 방법 중 하나

> $A=UΣV^T$
> - $A: m×n$ rectangular matrix
> - $U: m×m$ orthogonal matrix
> - $Σ: m×n$ diagonal matrix
> - $V: n×n$ orthogonal matrix

> $U$가 orthogonal matrix라고 한다면,
> $UU^T=U^TU=I$

> $Σ$가 diagonal matrix라고 한다면 $Σ$의 대각성분을 제외한 나머지 원소의 값은 모두 0

<div style="display: flex; justify-content: space-between;">
  <img src="https://velog.velcdn.com/images/heayounchoi/post/c6c3cde5-d238-490b-b8ea-ff1cf799daae/image.png" width="25%">
  <img src="https://velog.velcdn.com/images/heayounchoi/post/4cb2b18c-e0c0-41d8-bf54-d91c5b749186/image.png" width="25%">
  <img src="https://velog.velcdn.com/images/heayounchoi/post/a7547032-3e75-44c7-a420-96efab7c72e9/image.png" width="25%">
</div>

---

- 2차원 실수 벡터 공간에서 하나의 벡터가 주어지면 언제나 그 벡터에 직교하는 벡터를 찾을 수 있음
- 하지만 직교하는 두 벡터에 대해 동일한 선형 변환 A를 취해준다고 했을 때, 그 변환 후에도 여전히 직교한다고 보장할 수는 없음

- $A\vec{x}$와 $A\vec{y}$는 $A$라는 행렬(즉, 선형변환)을 통해 변환되었을 때, 길이가 조금씩 변함. 이 값들을 scaling factor라고 할 수 있지만, 일반적으로는 singular value라고 하고 크기가 큰 값부터 $σ_1$, $σ_2$,⋯ 등으로 부름
- 선형 변환 전의 직교하는 벡터 $\vec{x}$, $\vec{y}$는 다음과 같이 열벡터의 모음으로 생각할 수 있으며 이것이 $A=UΣV^T$에서 $V$행렬에 해당
<img src="https://velog.velcdn.com/images/heayounchoi/post/52e36119-3b4f-4935-992d-f7a756c47149/image.png" width="25%">

- 선형 변환 후의 직교하는 벡터 $A\vec{x}$, $A\vec{y}$에 대하여 각각의 크기를 1로 정규화한 벡터를 $\vec{u_1}$, $\vec{u_2}$라 한다면 이 둘의 열 벡터의 모음이 $A=UΣV^T$에서 $U$ 행렬에 해당
<img src="https://velog.velcdn.com/images/heayounchoi/post/1b2bd3fa-e4ff-47a1-b674-9fe8e457d79b/image.png" width="25%">

- singular value(즉, scaling factor)는 다음과 같이 $Σ$ 행렬로 묶어서 생각할 수 있음
<img src="https://velog.velcdn.com/images/heayounchoi/post/e96d4c9a-a57f-4946-9224-a4f83b7b062e/image.png" width="25%">

$AV=UΣ$

- " $V$에 있는 열벡터를 행렬 $A$를 통해 선형변환 할 때, 그 크기는 $σ_1$, $σ_2$만큼 변하지만, 여전히 직교하는 벡터들을 $U$를 찾을 수 있는가?"
<img src="https://velog.velcdn.com/images/heayounchoi/post/623cf17d-ff98-4723-a759-58e385bef0e1/image.png" width="25%">

- 차원이 변하는 선형변환의 경우에도 선형 변환 전 벡터들 중 하나의 singular value를 0으로 만듬으로써 특이값 분해 가능

---

**특이값 분해의 목적**
- SVD라는 방법을 이용해 $A$라는 임의의 행렬을 여러개의 $A$ 행렬과 동일한 크기를 갖는 여러개의 행렬로 분해해서 생각할 수 있는데, 분해된 각 행렬의 원소의 값의 크기는 $σ$의 값의 크기에 의해 결정됨

---
**특이값 분해의 활용**
- 특이값 분해는 분해되는 과정보다는 분해된 행렬을 다시 조합하는 과정에서 그 응용력이 빛을 발함
- 기존의 $U$, $Σ$, $V^T$로 분해되어 있던 $A$행렬을 특이값 p개만을 이용해 A’라는 행렬로 ‘부분 복원’ 할 수 있음
- 특이값의 크기에 따라 $A$의 정보량이 결정되기 때문에 값이 큰 몇개의 특이값들을 가지고도 충분히 유용한 정보를 유지할 수 있음
<img src="https://velog.velcdn.com/images/heayounchoi/post/7cbdbca9-f365-4e22-9c0b-26bb15cde4fe/image.png" width="75%">
<img src="https://velog.velcdn.com/images/heayounchoi/post/f9dd7b1e-41b0-45eb-99a7-311070aec290/image.png" width="50%">
