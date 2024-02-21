# EdgeBoxes

EdgeBoxes는 이미지 내에서 객체 후보(object proposals) 영역을 식별하기 위한 알고리즘입니다. 이 방법은 객체의 경계 상자(bounding box)를 제안함에 있어서, 경계(edge) 정보를 활용합니다. EdgeBoxes 알고리즘의 기본 원리는 이미지 내에서 경계가 많이 나타나는 영역이 객체의 경계일 가능성이 높다는 것입니다. 즉, 객체는 일반적으로 내부보다 경계에서 더 많은 에지 정보를 포함하고 있기 때문에, 에지 기반의 접근 방식이 객체 검출에 효과적일 수 있습니다.

EdgeBoxes 알고리즘의 주요 단계는 다음과 같습니다:

1. **에지 검출**: 먼저 이미지에서 에지를 검출합니다. 이를 위해 보통 Structured Edge Detection과 같은 강력한 에지 검출 알고리즘이 사용됩니다.
    - Structured Edge Detection은 기존의 에지 검출 알고리즘들이 개별 픽셀 또는 작은 패치(patch) 단위로 에지를 검출하는 데 반해, 이미지 내의 구조적인 패턴과 관계를 파악하여 에지를 검출하는 알고리즘입니다. 이 접근 방식은 특히 머신 러닝, 특히 결정 트리 기반의 학습 모델을 사용하여 이미지의 구조적인 정보를 학습하고 이를 에지 검출에 활용합니다. (지도 학습)
2. **지향성 그래디언트 계산**: 검출된 에지의 지향성(orientation)을 계산합니다.
    - 에지 검출에 있어 지향성을 계산하는 것은 에지가 이미지 상에서 어떤 방향을 가지고 있는지를 결정하는 과정입니다. 예를 들어, 수직 에지는 주로 수평 방향의 변화를 나타내며, 그래디언트 벡터는 수직 방향을 가리키게 됩니다. 반면에, 수평 에지는 수직 방향의 변화를 나타내고, 그래디언트 벡터는 수평 방향을 가리키게 됩니다.
3. **에지 그룹핑**: 에지 픽셀을 그룹핑하여, 각 그룹이 객체의 일부를 형성할 수 있는지를 평가합니다. 이는 일종의 클러스터링으로 볼 수 있습니다.
4. **상자 점수 계산**: 이 과정에서 각각의 경계 상자에 대해 "객체성 점수"를 계산합니다. 이 점수는 상자 내의 에지들이 얼마나 잘 정렬되어 있는지, 즉, 상자의 경계에 집중되어 있는지를 평가합니다.
5. **정렬 점수 계산**: 상자 내부의 에지 픽셀이 상자의 경계와 얼마나 잘 정렬되어 있는지를 계산하여 각 상자에 점수를 매깁니다. 이 점수는 후보 영역이 실제 객체를 포함할 가능성과 관련이 있습니다.
6. **Non-Maximum Suppression (NMS)**: 서로 중첩되는 상자들 중에서 점수가 낮은 상자들을 제거합니다.

EdgeBoxes는 그 효율성 때문에 매우 빠른 영역 제안 알고리즘으로 간주됩니다. 그러나 최근에는 Region Proposal Network (RPN)을 사용하는 Faster R-CNN과 같은 방법들이 더 우수한 성능을 보여주면서 주목을 받고 있습니다. 그럼에도 불구하고, EdgeBoxes는 계산 비용이 낮은 영역 제안 방법으로 여전히 유용합니다.