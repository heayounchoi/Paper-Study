### [Going deeper with convolutions](https://arxiv.org/pdf/1409.4842.pdf) / Inception

**Abstract**
- main hallmark of this architecture is the improved utilization of the computing resources inside the network
- achieved by a carefully crafted design that allows for increasing the depth and width of the network while keeping the computational budget constant
- to optimize quality, the architectural decisions were based on the Hebbian principle and the intuition of multi-scale processing
---

**Introduction**
- GoogLeNet submission to ILSVRC 2014 uses 12x fewer parameters than Alexnet, while being significantly more accurate
- for most of the experiments, the models were designed to keep a computational budget of 1.5 billion multiply-adds at inference time, so that they could be put to real world use at a reasonable cost
---

**Related Work**
- When applied to conv layers, the Network-in-Network method could be viewed as additional 1x1 conv layers followed typically by the rectified linear activation
- this approach is heavily used in GoogLeNet
- 1x1 conv have dual purpose in GoogLeNet setting: 
> 1) dimension reduction modules to remove computational bottlenecks
> 2) allows for increasing the width without significant performance penalty
---

**Motivation and High Level Considerations**
- the most straightforward way of improving the performance of deep neural networks is by increasing their size
- this includes both increasing the depth - the number of levels - of the network and its width: the number of units at each level
- bigger size typically means a larger number of parameters, which makes the enlarged network more prone to overfitting, especially if the number of labeled examples in the training set is limited
- another drawback of uniformly increased network size is the dramatically increased use of computational resources
- The fundamental way of solving both issues would be by ultimately moving from fully connected to sparsely connected architectures.
- **if the probability distribution of the data-set is representable by a large, very sparse deep neural network, then the optimal network topology can be constructed layer by layer by analyzing the correlation statistics of the activations of the last layer and clustering neurons with highly correlated outputs**
- on the downside, todays computing infrastructures are very inefficient when it comes to numerical calculation on non-uniform sparse data structures
- the vast literature on sparse matrix computations suggests that clustering sparse matrices into relatively dense submatrices tends to give state of the art practical performance for sparse matrix multiplication
- **the Inception architecture started out as a case study of the first author for assessing the hypothetical output of a sophisticated network topology construction algorithm that tries to approximate a sparse structure for vision networks and covering the hypothesized outcome by dense, readily available components**
---

**Architectural Details**

<img src="https://velog.velcdn.com/images/heayounchoi/post/b782efa4-97ab-4240-981f-8e0281512077/image.png" width="50%">

- Inception 아키텍처의 핵심 아이디어는 컨볼루션 비전 네트워크에서 최적의 local sparse structure를 어떻게 근사하고, 이미 사용 가능한 dense 컴포넌트로 커버할 수 있는지를 찾아내는데 있음
- 이전 레이어의 각 단위가 입력 이미지의 어떤 영역과 대응된다고 가정하며, 이 단위들은 필터 뱅크로 그룹화됨
- 출력 필터 뱅크가 단일 출력 벡터로 연결되어 다음 단계의 입력을 형성함
- 더 높은 레이어에서는 더 추상적인 특성이 포작되고, 공간적 집중도는 감소할 것이기 때문에 더 높은 레이어로 이동함에 따라 3x3 및 5x5 컨볼루션의 비율이 증가해야 함
- 문제는 5x5 컨볼루션 필터들이 소수만 사용되더라도, 많은 필터를 가진 컨볼루션 레이어 위에서 계산 비용이 매우 높을 수 있다는 점 (a archtecture)
- 그래서 두번째 아이디어는, wherever the computational requirements would increase too much, dimension reductions and projections are applied
- for memory efficiency during training, Inception modules are used only at higher layers and the lower layers are kept in traditional convolutional fashion
- 성능을 좀 덜하게 만들어 computational cost를 줄이면 비슷한 네트워크들보다 속도가 빠르다고 함
---

**GoogLeNet**

<img src="https://velog.velcdn.com/images/heayounchoi/post/e99bc4fd-8d9c-4787-a0e5-dc91ad2b5387/image.png" width="50%">
  
<img src="https://velog.velcdn.com/images/heayounchoi/post/182a8fc0-ee28-4167-9642-58e607ac39d5/image.png">

- the network is 22 layers deep when counting only layers with parameters (or 27 layers if pooling is also counted)
<img src="https://velog.velcdn.com/images/heayounchoi/post/370dba26-d80e-46fd-929a-2b9dccd372b5/image.png">

- 네트워크가 깊어서 backpropagation이 잘 될지가 문제였음
- 얕은 네트워크에선 이게 잘 되는데, 그러려면 중간 레이어의 특징들이 discriminative 해야함
- by adding auxiliary classifiers connected to these intermediate layers, we would expect to encourage discrimination in the lower stages in the classifier, increase the gradient signal that gets propagated back, and provide additional regularization
- during training, their loss gets added to the total loss of the network with a discount weight (0.3)
- at inference time, these auxiliary networks are discarded

<img src="https://velog.velcdn.com/images/heayounchoi/post/c5516ad4-8cf1-4c1d-8cd2-8b2062a222f1/image.png">
<img src="https://velog.velcdn.com/images/heayounchoi/post/57ed1729-0d30-45d2-a1ea-7a41b364ca29/image.png">


