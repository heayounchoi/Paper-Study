# R-CNN

### Rich feature hierarchies for accurate object detection and semantic segmentation

• semantic segmentation

- a computer vision task that involves partitioning an image into segments where each segment represents a specific class or category
- Unlike object detection, which provides a bounding box around objects of interest, semantic segmentation classifies each pixel in the image, resulting in a pixel-wise mask for the entire image. This mask indicates the category to which each pixel belongs.
- Semantic segmentation is a crucial step towards enabling machines to have a detailed and comprehensive understanding of visual scenes, which is vital for many AI-driven applications.
- Introduction
    - SIFT (Scale-Invariant Feature Transform)
        - SIFT는 객체의 크기와 회전에 불변하는 특징을 추출하는 데 사용됨
        - 작동 원리:
            1. Scale-space Extrema Detection: 이미지에서 관심 지점(이미지의 다양한 스케일에서 동일한 특징을 가진 지점)을 찾기 위해, 가우시안 차이(Difference of Gaussian, DoG)를 사용하여 스케일-스페이스 극값을 찾음
            2. Keypoint Localization: 각 극값 주변에서 키포인트의 위치와 스케일을 정확하게 결정
            3. Orientation Assignment: 각 키포인트에 방향을 할당하여 회전에 불변성을 부여
            4. Keypoint Descriptor: 키포인트 주변의 지역 그래디언트 정보를 사용하여 키포인트 디스크립터를 생성. 이 디스크립터는 키포인트의 지역적인 모양에 대한 정보를 담고 있으며, 변형에 강한 특성을 가짐
            5. Matching: SIFT 특징은 다른 이미지의 특징과 비교하여 매칭할 수 있으며, 이를 통해 객체 인식이나 이미지 스티칭 등에 사용됨
    - HOG (Histogram of Oriented Gradients)
        - HOG는 주로 보행자 감지에 사용됨. HOG는 이미지의 지역적 그래디언트 방향의 분포를 특징으로 사용함
        - 작동 원리:
            1. Gradient Computation: 이미지의 각 픽셀에서 그래디언트의 방향과 크기를 계산
            2. Cell Division: 이미지를 작은 연결된 영역인 셀로 나눔
            3. Histogram Generation: 각 셀에 대해 그래디언트 방향의 히스토그램을 생성. 이 히스토그램은 셀의 그래디언트 정보를 요약함
            4. Block Normalization: 인접한 셀들을 블록으로 그룹화하고, 이 블록의 히스토그램을 정규화하여 조명 변화와 그림자에 강하게 만듬
            5. Descriptor Formation: 모든 블록의 정규화된 히스토그램을 연결하여 최종 HOG 특징 디스크립터를 형성
            6. Detection: HOG 디스크립터는 분류기(예: SVM)에 입력되어 객체 감지에 사용됨
    - blockwise orientation histograms
        - 이미지를 여러 개의 작은 영역(블록)으로 나누고, 각 블록 내에서 픽셀의 방향성(그래디언트의 방향)에 대한 통계를 수집하는 방식
    - complex cells in V1
        - 시각 정보 처리에 있어서 중요한 역할을 하는 뇌의 시각 피질 내의 특정 유형의 신경 세포. V1 영역의 복잡한 세포들은 특정 방향의 에지나 선에 반응하며, 이는 SIFT와 HOG가 추출하는 방향성 정보와 유사한 처리를 뇌가 수행한다는 것을 암시
    - Support Vector Machine (SVM) (지도 학습)
        - 이미지 설명: [https://bkshin.tistory.com/entry/머신러닝-2서포트-벡터-머신-SVM](https://bkshin.tistory.com/entry/%EB%A8%B8%EC%8B%A0%EB%9F%AC%EB%8B%9D-2%EC%84%9C%ED%8F%AC%ED%8A%B8-%EB%B2%A1%ED%84%B0-%EB%A8%B8%EC%8B%A0-SVM)
        - 지도 학습 모델 중 하나로, 주로 분류(classification)와 회귀(regression) 문제에 사용됨.
        - SVM의 기본 아이디어는 데이터를 분류하는 최적의 결정 경계(decision boundary), 즉 '최대 마진 분류기(maximum margin classifier)'를 찾는 것
        - SVM의 주요 개념:
            1. 결정 경계(Decision Boundary): SVM은 두 클래스를 구분하는 결정 경계를 찾음. 이 경계는 데이터 포인트(샘플)들을 분류하는 데 사용됨.
            2. 서포트 벡터(Support Vectors): 결정 경계에 가장 가까이 있는 데이터 포인트들을 서포트 벡터라고 함. 이 포인트들은 결정 경계를 정의하는 데 중요한 역할을 하며, 이들 사이의 거리를 최대화하는 것이 SVM의 목표.
            3. 마진(Margin): 서포트 벡터와 결정 경계 사이의 거리를 마진이라고 함. SVM은 이 마진을 최대화하여, 새로운 데이터 포인트에 대한 분류 오류를 최소화하려고 .
            4. 커널 트릭(Kernel Trick): 비선형 분류 문제를 해결하기 위해, SVM은 커널 트릭을 사용하여 저차원의 데이터를 고차원 공간으로 매핑. 이를 통해 선형 분류기가 작동할 수 있는 고차원에서 데이터를 분류할 수 있음.
            5. 소프트 마진(Soft Margin): 실제 데이터는 종종 완벽하게 분류할 수 없기 때문에, SVM은 일부 오분류를 허용하는 소프트 마진 접근 방식을 사용. 이를 통해 모델의 일반화 능력을 향상시킬 수 있음.
        - SVM의 작동 방식:
            1. **모델 구축**: SVM 모델은 학습 데이터를 사용하여 결정 경계를 찾음. 이 과정에서 서포트 벡터와 마진이 결정됨.
            2. **분류**: 모델이 구축되면, 새로운 데이터 포인트는 결정 경계를 기반으로 분류됨. 데이터 포인트가 경계의 어느 쪽에 위치하는지에 따라 해당 클래스로 분류됨.
            3. **최적화**: SVM은 결정 경계와 서포트 벡터 사이의 마진을 최대화하는 방향으로 모델을 최적화함. 이는 종종 복잡한 최적화 문제로 이어지며, 적절한 알고리즘을 사용하여 해결.
        - Linear SVM
            - Linear SVM은 데이터가 선형적으로 구분될 수 있는 경우에 사용됨. 즉, 데이터를 분류하는 결정 경계(decision boundary)가 직선(또는 고차원에서는 평면 또는 초평면)인 경우
            - Linear SVM은 주어진 데이터 포인트들을 두 클래스로 나누는 최적의 선형 경계를 찾는 것을 목표로 함. 이 최적의 경계는 두 클래스 사이의 마진(margin)을 최대화하는 것으로, 마진은 결정 경계와 가장 가까운 훈련 데이터 포인트(서포트 벡터) 사이의 거리.
        - Nonlinear SVM
            - Nonlinear SVM은 선형적으로 구분되지 않는 데이터에 사용
            - 이 모델은 커널 트릭(kernel trick)이라는 기법을 사용하여 입력 데이터를 더 높은 차원으로 매핑함으로써, 원래의 비선형 데이터에 대해 선형 분류가 가능한 새로운 특성 공간을 찾음. 예를 들어, 원형 패턴이나 복잡한 구조를 가진 데이터는 원래의 특성 공간에서는 선형적으로 분리할 수 없지만, 커널 함수를 통해 변환된 공간에서는 선형 분리가 가능해짐
    - High-capacity model
        - 많은 양의 데이터를 학습하고, 복잡한 패턴을 모델링할 수 있는 능력을 가진 모델을 의미
    - receptive field
        - 입력 이미지 내에서 특정 뉴런 또는 뉴런의 집합이 '보는' 입력 데이터의 영역
    - Affine image warping
        - 이미지 처리에서 사용되는 기법으로, 이미지를 변형하는 데에 있어서 선형 변환을 적용하고, 그 결과로 이미지의 위치, 크기, 회전, 그리고 기울기(경사) 등을 변경. Affine 변환은 원본 이미지의 모든 평행선이 변환 후에도 여전히 평행을 유지한다는 특성을 가지고 있음. (이미지의 기본적인 기하학적 구조를 보존한다는 것을 의미)
        - 이미지의 평행선: 원근감(perspective)을 분석할 때 평행선은 원근의 소실점(vanishing point)으로 수렴하는 경향이 있음. 이러한 성질을 이용하여 3D 공간에서의 객체의 위치나 방향을 추정할 수 있음.
    - Bag-of-Visual-Words (BoVW) Approach
        - Bag-of-Visual-Words는 자연어 처리에서 영감을 받은 이미지 표현 방법
        - 이 방법은 이미지 내의 특징들을 '단어'로 간주하고, 이미지 전체를 이러한 '시각적 단어들'의 '가방'으로 표현
        - BoVW 접근 방식의 주요 단계:
            1. 특징 추출: 이미지에서 키포인트(keypoints)를 찾고, 각 키포인트 주변의 지역적인 특징을 추출. 이를 위해 SIFT, SURF, ORB 등의 알고리즘이 사용될 수 있음
            2. 코드북 생성: 추출된 특징들을 클러스터링하여 각 클러스터의 중심을 '시각적 단어'로 정의. 이 과정은 종종 k-means 알고리즘을 사용하여 수행됨
            3. 히스토그램 구축: 각 이미지에 대해, 추출된 특징들이 어떤 '시각적 단어'에 해당하는지를 기반으로 히스토그램을 구축. 이 히스토그램은 이미지가 각 '시각적 단어'를 얼마나 포함하고 있는지를 나타냄
            4. 분류기 학습: 생성된 히스토그램을 사용하여 이미지 분류기를 학습. SVM이나 다른 머신러닝 알고리즘이 이용될 수 있음
    - Spatial Pyramid
        - 이미지의 공간적인 구조를 보다 잘 포착하기 위한 BoVW의 확장
        - 이 방법은 이미지를 여러 레벨의 해상도로 분할하고, 각 레벨에서 BoVW 히스토그램을 계산. 이렇게 하면 이미지의 다양한 해상도에서 특징을 포착할 수 있으며, 이미지의 공간적인 배치 정보도 어느 정도 보존됨
        - Spatial Pyramid의 주요 단계:
            1. 이미지 분할: 이미지를 여러 격자(grid)로 분할. 이 격자는 다양한 크기와 위치를 가질 수 있음
            2. 레벨별 특징 계산: 각 격자 내에서 BoVW 히스토그램을 계산. 이는 이미지의 다른 부분에서 추출된 특징들을 포함
            3. 특징 통합: 모든 격자의 히스토그램을 결합하여 최종적인 이미지 표현을 생성. 이 표현은 이미지의 공간적인 구조를 반영
            4. 분류기 학습: 통합된 특징을 사용하여 이미지 분류기를 학습
        - Fine-grained sub-categorization
            - 매우 세밀하고 구체적인 카테고리로 객체를 분류하는 과정
        - Domain adaptation
            - 한 도메인(domain) 또는 분포에서 학습된 모델을 다른 도메인 또는 분포에 적용할 때 발생하는 성능 저하를 최소화하는 과정. 즉, 모델이 한 환경에서 학습되었지만, 다른 환경에서도 잘 작동하도록 만드는 것이 목표
        - greedy non-maximum suppression
            - 객체 탐지에서 여러 후보 중 가장 높은 점수를 가진 경계 상자를 선택하고, 나머지 중복되는 상자들을 제거하는 과정
- Object detection with R-CNN
    - Module design
        - Region proposals
            - 다양한 region proposal 방법들:
                1. Objectness: 이미지 내의 각 픽셀이 객체의 일부일 확률을 평가. 이 방법은 객체처럼 보이는 영역을 찾아내기 위해 여러 가지 시각적 단서를 사용
                2. Selective Search: 이미지의 색상, 질감, 크기 및 모양 일관성과 같은 다양한 전략을 사용하여 이미지를 세분화하고, 이 세분화된 영역들을 점차적으로 병합하여 객체 후보를 생성
                3. Category-Independent Object Proposals: 특정 카테고리에 의존하지 않고 객체 후보를 생성. 이는 다양한 객체 카테고리에 걸쳐 일반적으로 적용될 수 있는 방법으로, 객체가 될 수 있는 영역을 식별하는 데 초점을 맞춤
                4. Constrained Parametric Min-Cuts (CPMC): 그래프 컷(graph cut) 기반 방법을 사용하여 이미지를 여러 영역으로 분할. 이 방법은 이미지 내의 객체를 분리하기 위해 최적화된 경계를 찾음
                5. Multi-Scale Combinatorial Grouping (MCG): 여러 스케일에서 세분화된 영역을 생성하고, 이를 조합하여 객체 후보를 형성. 이 방법은 다양한 크기와 모양의 객체를 탐지하는 데 유용함
                6. Ciresan et al.: 정기적으로 배치된 정사각형 크롭에 CNN을 적용하여 미토시스(mitotic cells)를 탐지하는 방법을 제안. 이는 정사각형 크롭이 region proposal의 한 형태로 사용됨을 보여줌
            - R-CNN은 선택적 탐색(Selective Search) 사용
                
                ![img.png](R-CNN%200b2f76f464394891bb64050503bb0c79/img.png)
                
        - Feature extraction
    - Test-time detection
        - Run-time analysis
            - Hashing
                - the process of transforming any given key or a string of characters into another value
            - linear predictor function
                - a linear function (linear combination) of a set of coefficients and explanatory variables (independent variables), whose value is used to predict the outcome of a dependent variable
    - Training
        - Supervised pre-training
        - Domain-specific fine-tuning
        - Object category classifiers
            - hard negative mining
                - positive 샘플과 negative 샘플의 개수를 균일하게 만드는 방법. 신뢰도 점수(confidence score)를 활용해 negative 샘플을 선정하는 것. 이미지 안에 배경 영역은 굉장히 넓어서, negative 샘플이 될 경계 박스가 많음. negative 샘플이 지나치게 많으면 객체 탐지 모델의 성능이 떨어질 우려가 있으니, 신뢰도 점수가 가장 높은 경계 박스순으로 negative 샘플을 선정한다는 의미
    - Results on PASCAL VOC 2010-12
        - Bounding box Regression
            - Selective search를 통해 찾은 박스 위치는 정확하지 않음. Predicted box와 ground truth box와의 차이를 줄여주는 bounding box regression이 필요. Linear regression model로 볼 수 있음.
            - ground truth box와 predict box의 IoU가 threshold 이상인 값에 대해서만 진행
    - Results on ILSVRC2013 detection
- Visualization, ablation, and modes of error
    - Visualizing learned features
        - Deconvolution: convolution이 feature map의 크기를 줄이는 것과 반대로, deconvolution은 feature map의 크기를 증가시킴
    - Ablation studies
        - Performance layer-by-layer, without fine-tuning
        - Performance layer-by-layer, with fine-tuning
        - Comparison to recent feature learning methods
    - Network architectures
    - Detection error analysis
    - Bounding-box regression
    - Qualitative results
- The ILSVRC2013 detection dataset
    - Dataset overview
    - Region proposals
    - Training data
    - Validation and evaluation
    - Ablation study
    - Relationship to OverFeat
- Semantic segmentation
    - CNN features for segmentation
    - Results on VOC 2011
- Conclusion
    
    

참고: [https://bkshin.tistory.com/entry/논문-리뷰-R-CNN-톺아보기](https://bkshin.tistory.com/entry/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-R-CNN-%ED%86%BA%EC%95%84%EB%B3%B4%EA%B8%B0)

공부할 것

- SIFT, HOG 원리
- object detection 모델 코드
- OverFeat(sliding window approach)
- 다양한 region proposal 방법들
- Caffe CNN library
- deconvolution