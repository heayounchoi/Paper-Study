# sobel operator
- 이미지의 edge를 검출하는데 사용되는 기술 중 하나로, 이미지에서 밝기의 변화율이 높은 지점을 찾아 경계로 식별함
    - vs Prewitt operator: 둘 다 수평 및 수직 방향의 경계를 검출하기 위한 커널을 사용함. 주요 차이점은 커널 내 가중치 분포. Prewitt은 모든 방향에 동일한 가중치를 적용하는 반면, sobel은 중앙 행이나 열에 더 큰 가중치를 부여하여 보다 정밀한 경계 검출이 가능함. 동시에 노이즈에 더 민감할 수도 있음
    - vs Canny edge detector: canny는 다단계 프로세스를 통해 이미지에서 경계를 검출함. 노이즈 제거, 경계 강도 계산, 비최대 억제, 이력 임계값 처리 등을 포함하여 보다 정확하고 섬세한 경계 검출이 가능함. 하지만 sobel에 비해 계산 복잡도가 더 높고, 여러 매개변수의 조정이 필요함
    - vs Laplacian of Gaussian: LoG는 sobel 보다 더 섬세한 경계 검출이 가능하며, 노이즈에 강함. 동시에 계산 복잡도가 더 높고, 처리 시간이 더 길음
- 간단해서 쓰기 좋다는 뜻~